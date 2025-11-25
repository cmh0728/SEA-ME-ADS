import sys, math, yaml
import numpy as np
import cv2

# ---------- 사용자 설정 ----------
# 단일 체커보드 프레임으로 외부 파라미터와 IPM을 추정하는 스크립트
IMAGE_PATH = "frame0000.jpg"      # 바닥 체스보드가 보이는 한 장
# 내부 코너 수(가로 x 세로) - 인쇄물 내부코너 기준으로 맞춰주세요!
BOARD_COLS, BOARD_ROWS = 9, 7
SQUARE_SIZE_M = 0.011  # 1.1 cm

# IPM 영역/스케일 (체커보드 평면 기준 [X(앞), Y(좌/우)])
X_MIN, X_MAX = -0.25 , 0.1 # max가 차량 뒤쪽, min이 앞쪽
Y_MIN, Y_MAX = -0.28 , 0.21   # max가 차량 왼쪽 
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
    """
    Z=0(체커보드/바닥) 평면에서 [X, Y, 1]^T → 이미지 픽셀 [u, v, 1]^T 로 가는 H
    s [u, v, 1]^T = K [ r1 r2 t ] [X, Y, 1]^T
    """
    H = K @ np.hstack([R[:, 0:2], t])   # 3x3
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
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fail to read image:", IMAGE_PATH)
        sys.exit(1)

    # 2) 전처리 (그레이 → CLAHE → 블러)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 3) 체스보드 코너 검출 (SB → 구형 폴백)
    pattern_size = (BOARD_COLS, BOARD_ROWS)  # (cols, rows)
    ok = False

    try:
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
        if ok:
            # corners: (N, 1, 2), row-major 순서 그대로 사용
            vis = img.copy()
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
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite("corners_debug.png", vis)

    if not ok:
        print("Chessboard not found. 조명/반사/거리/내부코너 수(BOARD_COLS, BOARD_ROWS) 확인해줘.")
        sys.exit(1)

    print(f"Detected {len(corners)} corners.")

    # 4) 3D 보드 코너 (Z=0 평면)
    # OpenCV 코너 순서: row-major (row 0..ROWS-1, 각 row마다 col 0..COLS-1)
    # row 방향 -> 차량 +X(앞), col 방향 -> 차량 +Y(왼쪽 양수)
    objp = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            X =  row * SQUARE_SIZE_M       # 앞(+X)
            Y = -col * SQUARE_SIZE_M       # 왼쪽(+Y) 되도록 col 증가에 -부호
            objp.append([X, Y, 0.0])
    objp = np.array(objp, dtype=np.float32)

    print("Mapped checkerboard object points preview (first 10):")
    print(objp[:10])

    # 5) PnP → R, t
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        print("solvePnP failed")
        sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    # ★ 카메라 중심 (월드 좌표계) 계산
    #   X_cam = R X_world + t  →  X_world = R^T (X_cam - t)
    #   카메라 중심은 X_cam = 0 → C_world = -R^T t
    C_world = -R.T @ t
    cam_height = abs(C_world[2,0])

    roll, pitch, yaw = rodrigues_to_rpy(R)
    print("\n=== Extrinsics (Z=0=ground, 체커보드 평면) ===")
    print("R=\n", R)
    print("t(m) [world origin in camera frame]=\n", t)
    print("Camera center in world frame C_w = -R^T t :\n", C_world)
    print(f"roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°")
    print(f"camera height ~= {cam_height:.3f} m")

    # 6) H 만들고 IPM 생성
    H = build_ipm_homography_from_plane(K, R, t)  # 바닥(Z=0) → 이미지 호모그래피

    # IPM 해상도 계산
    W  = int(round((X_MAX - X_MIN)/INTERVAL_X))
    Hh = int(round((Y_MAX - Y_MIN)/INTERVAL_Y))
    print(f"IPM target size: {W} x {Hh}")

    # 가로(W) 방향: Y_MIN → Y_MAX (좌/우)
    # 세로(Hh) 방향: X_MIN → X_MAX (앞/뒤)
    ground_corners = np.float32([
    [X_MIN, Y_MAX, 1.0],   # 좌측 근처 (왼쪽, 가까운)
    [X_MIN, Y_MIN, 1.0],   # 우측 근처 (오른쪽, 가까운)
    [X_MAX, Y_MIN, 1.0],   # 우측 먼 쪽
    [X_MAX, Y_MAX, 1.0],   # 좌측 먼 쪽
    ]).T  # 3x4


    img_corners_h = H @ ground_corners    # 3x4
    img_corners = (img_corners_h[:2] / img_corners_h[2]).T.astype(np.float32)  # 4x2

    print("Projected IPM region corners on image:")
    print(img_corners)

    # Debug: IPM 대상 영역이 영상 안에 들어오는지 빨간 폴리라인으로 확인
    debug = img.copy()
    cv2.polylines(
        debug,
        [img_corners.reshape(-1, 1, 2).astype(int)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3
    )
    cv2.imwrite("ipm_region.png", debug)
    print("Wrote ipm_region.png (red polygon on original image).")

    # IPM 목적지 좌표 (픽셀 공간)
    dst_corners = np.float32([
        [0,     0],        # 좌측-가까운
        [W-1,   0],        # 우측-가까운
        [W-1, Hh-1],       # 우측-먼 쪽
        [0,   Hh-1],       # 좌측-먼 쪽
    ])

    # 이미지 상의 네 점(img_corners) → IPM 평면(dst_corners) 호모그래피
    G = cv2.getPerspectiveTransform(img_corners, dst_corners)
    ipm = cv2.warpPerspective(img, G, (W, Hh))
    cv2.imwrite("ipm.png", ipm)
    print(f"\nSaved IPM to ipm.png  ({W}x{Hh})")

    out = {
        "R": R.tolist(),
        "t_world_origin_in_cam": t.reshape(-1).tolist(),
        "camera_center_world": C_world.reshape(-1).tolist(),
        "roll_deg": float(math.degrees(roll)),
        "pitch_deg": float(math.degrees(pitch)),
        "yaw_deg": float(math.degrees(yaw)),
        "camera_height_m": float(cam_height),
        "H_ground_to_image": H.tolist()
    }
    with open("extrinsics_and_h.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print("Wrote extrinsics_and_h.yaml")

if __name__ == "__main__":
    main()

