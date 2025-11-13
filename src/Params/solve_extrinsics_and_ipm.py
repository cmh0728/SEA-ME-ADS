import sys, math, yaml
import numpy as np
import cv2

# ---------- 사용자 설정 ----------
IMAGE_PATH = "frame0003.jpg"      # 바닥 체스보드가 보이는 한 장
# 내부 코너 수(가로 x 세로) - 인쇄물 내부코너 기준으로 맞춰주세요!
BOARD_COLS, BOARD_ROWS = 9, 7
SQUARE_SIZE_M = 0.011  # 1.1 cm

# IPM 영역/스케일
X_MIN, X_MAX = -0.05, 0.088 + 0.05
Y_MIN, Y_MAX = -0.05, 0.066 + 0.05
W_target, H_target = 640, 480
INTERVAL_X = (X_MAX - X_MIN) / W_target
INTERVAL_Y = (Y_MAX - Y_MIN) / H_target

# ---------------------------------

def rodrigues_to_rpy(R):
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
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fail to read image:", IMAGE_PATH); sys.exit(1)

    # 2) 전처리 (그레이 → CLAHE → 블러)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 3) 체스보드 코너 검출 (SB → 구형 폴백)
    pattern_size = (BOARD_COLS, BOARD_ROWS)  # 내부 코너 수!
    ok = False
    try:
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
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

    if not ok:
        print("Chessboard not found. 조명/반사/거리/내부코너 수 확인해줘.")
        sys.exit(1)

    # 4) 3D 보드 코너 (Z=0 평면)
    objp = np.zeros((BOARD_COLS*BOARD_ROWS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1,2)
    objp *= SQUARE_SIZE_M

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
    H = build_ipm_homography_from_plane(K, R, t)

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

    dst_corners = np.float32([
        [0, 0],
        [W-1, 0],
        [W-1, Hh-1],
        [0, Hh-1],
    ])

    G = cv2.getPerspectiveTransform(img_corners, dst_corners)
    ipm = cv2.warpPerspective(img, G, (W, Hh))
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
