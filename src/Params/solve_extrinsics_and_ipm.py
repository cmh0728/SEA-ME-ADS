import sys, math, yaml
import numpy as np
import cv2

# ---------- 사용자 설정 ----------
# 단일 체커보드 프레임으로 외부 파라미터와 IPM을 추정하는 스크립트
IMAGE_PATH = "frame0004.jpg"      # 바닥 체스보드가 보이는 한 장
# 내부 코너 수(가로 x 세로) - 인쇄물 내부코너 기준으로 맞춰주세요!
BOARD_COLS, BOARD_ROWS = 9, 7
SQUARE_SIZE_M = 0.011  # 1.1 cm

# IPM 영역/스케일
X_MIN, X_MAX = 0.0 , 0.2 # 차량 앞쪽 20cm
Y_MIN, Y_MAX = -0.2 , 0.2 # 차량 좌우 40cm
W_target, H_target = 1280, 720  # 목표 IPM 크기 (픽셀)
INTERVAL_X = (X_MAX - X_MIN) / W_target
INTERVAL_Y = (Y_MAX - Y_MIN) / H_target

# ---------------------------------

def rodrigues_to_rpy(R):
    # Rodrigues → roll/pitch/yaw 변환 (디버그용)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0
    return roll, pitch, yaw

def build_ipm_homography_from_plane(K, R, t):
    # 바닥(z=0) 기준 호모그래피 계산
    n = np.array([[0.0],[0.0],[1.0]])   # ground normal
    d = -t[2,0]                         # distance to plane (sign!)
    Kinv = np.linalg.inv(K)
    H = K @ (R - (t @ n.T)/d) @ Kinv
    return H

def main():
    # === K, D 하드코딩 ===
    D = np.array([
        -0.08518303109375061, 0.09162271169535907,
         0.0031898210475882326, -0.005419073450784245, 0.0
    ], dtype=np.float64)

    K = np.array([
        [671.3594253991467, 0.0,                 644.3611380949407],
        [0.0,               630.607858047108,    350.24879773346635],
        [0.0,               0.0,                  1.0]
    ], dtype=np.float64)

    print("K=\n", K)
    print("D=", D)

    # 1) 이미지 로드
    # 왜곡 보정 → 전처리(그레이/CLAHE/블러)
    img = cv2.imread(IMAGE_PATH)

    undistorted = cv2.undistort(img, K, D)

    # 2) 전처리 (그레이 → CLAHE → 블러)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 3) 체스보드 코너 검출 (SB → 구형 폴백)
    # 좌표계를 objp와 일치시키기 위해 reshape/transpose 수행
    pattern_size = (BOARD_COLS, BOARD_ROWS)  # 내부 코너 수!
    ok = False
    try:
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
        if ok:
            corners = corners.reshape(BOARD_ROWS, BOARD_COLS, 1, 2)
            corners = corners.transpose(1, 0, 2, 3).reshape(-1, 1, 2)
            vis = undistorted.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite("corners_debug.png", vis)

    except Exception:
        ok = False

    if not ok:
        flags_old = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FILTER_QUADS)
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags_old)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners = corners.reshape(BOARD_ROWS, BOARD_COLS, 1, 2)
            corners = corners.transpose(1, 0, 2, 3).reshape(-1, 1, 2)
            vis = undistorted.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite("corners_debug.png", vis)

    if not ok:
        print("Chessboard not found. 조명/반사/거리/내부코너 수 확인해줘.")
        sys.exit(1)

    # 4) 3D 보드 코너 (Z=0 평면)
    # 체커보드 행(row)이 차량 앞(+X), 열(col)이 좌(+Y)을 향하도록 좌표계를 정의
    # 체커보드 월드 좌표 생성: 행(row) → 차량 +X, 열(col) → 차량 +Y(왼쪽)
    objp = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x = row * SQUARE_SIZE_M                # 체커보드 세로(row) -> 차량 전방(+X)
            y = -(col * SQUARE_SIZE_M)             # 체커보드 가로(col) -> 차량 좌측(+Y), 왼쪽 양수 유지
            objp.append([x, y, 0.0])
    objp = np.array(objp, dtype=np.float32)
    print("Mapped checkerboard object points preview (first 10 rows):")
    print(objp[:10]) # 체크보드 좌표계 확인 

    # 5) PnP → R, t
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP failed"); sys.exit(1)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    roll, pitch, yaw = rodrigues_to_rpy(R)
    print("\n=== Extrinsics (Z=0=ground) ===")
    print("R=\n", R)
    print("t(m)=\n", t)
    print(f"roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°")
    print(f"camera height ~= {-t[2,0]:.3f} m")

    # 6) H 만들고 IPM 생성
    H = build_ipm_homography_from_plane(K, R, t)  # 바닥 → 이미지 호모그래피

    W  = int(round((X_MAX - X_MIN)/INTERVAL_X))
    Hh = int(round((Y_MAX - Y_MIN)/INTERVAL_Y))

    ground_corners = np.float32([
        [X_MIN, Y_MIN, 1.0],
        [X_MAX, Y_MIN, 1.0],
        [X_MAX, Y_MAX, 1.0],
        [X_MIN, Y_MAX, 1.0],
    ]).T  # 3x4

    img_corners_h = H @ ground_corners
    img_corners = (img_corners_h[:2] / img_corners_h[2]).T.astype(np.float32)
    # 여기서 img_corners를 원하는 값으로 직접 수정해도 됨.
    # 예: img_corners = np.array([[x0,y0],[x1,y1],...], dtype=np.float32)
    # 각 꼭짓점이 영상 안에 있는지 확인하고 수정해 주세요.

    # Debug: IPM 대상 영역이 영상 안에 들어오는지 빨간 폴리라인으로 확인
    debug = undistorted.copy()
    cv2.polylines(
        debug,
        [img_corners.reshape(-1, 1, 2).astype(int)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3
    )
    cv2.imwrite("ipm_region.png", debug)

    dst_corners = np.float32([
        [0, 0],
        [W-1, 0],
        [W-1, Hh-1],
        [0, Hh-1],
    ])

    G = cv2.getPerspectiveTransform(img_corners, dst_corners)
    ipm = cv2.warpPerspective(undistorted, G, (W, Hh))
    cv2.imwrite("ipm.png", ipm)
    print(f"\nSaved IPM to ipm.png  ({W}x{Hh})")

    out = {
        "R": R.tolist(),
        "t": t.reshape(-1).tolist(),
        "roll_deg": float(math.degrees(roll)),
        "pitch_deg": float(math.degrees(pitch)),
        "yaw_deg": float(math.degrees(yaw)),
        "camera_height_m": float(-t[2,0]),
        "H_ground_to_image": H.tolist()
    }
    with open("extrinsics_and_h.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print("Wrote extrinsics_and_h.yaml")

if __name__ == "__main__":
    main()
