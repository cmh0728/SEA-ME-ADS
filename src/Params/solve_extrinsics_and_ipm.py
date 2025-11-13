import re, sys, math, yaml
import numpy as np
import cv2

# ---------- 사용자 설정 ----------
IMAGE_PATH = "frame.jpg"          # 바닥 체스보드가 보이는 한 장
CAMERA_YAML = "calibration.txt"       # oST v5 형식 (camera_calibration 결과)
# 체스보드 내부 코너 수(가로 x 세로)와 한 칸 길이[m]
BOARD_COLS, BOARD_ROWS = 9, 7
SQUARE_SIZE_M = 0.011 # 1.1 cm

# IPM 영역/스케일(원하는 맵 범위와 해상도)
# 예: X: -1.5~+1.5 m, Y: 0~8 m 범위로 만들고, 1픽셀=2cm
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = 0.0, 8.0
INTERVAL_X = 0.02   # 1픽셀 당 X 2 cm
INTERVAL_Y = 0.02   # 1픽셀 당 Y 2 cm
# ---------------------------------

def load_ost_yaml(path):
    """
    oST v5 형식(camera_calibration 패키지 출력)의 K,D 를 읽는다.
    """
    with open(path, "r") as f:
        text = f.read()

    # K
    m = re.search(r"camera matrix\s+([\d\.\-\s]+)\n([\d\.\-\s]+)\n([\d\.\-\s]+)", text)
    if not m:
        raise RuntimeError("camera matrix(K) not found in YAML")
    row1 = [float(x) for x in m.group(1).split()]
    row2 = [float(x) for x in m.group(2).split()]
    row3 = [float(x) for x in m.group(3).split()]
    K = np.array([row1, row2, row3], dtype=np.float64)

    # D
    m = re.search(r"distortion\s+([-\d\.\s]+)", text)
    if not m:
        raise RuntimeError("distortion(D) not found in YAML")
    D = np.array([float(x) for x in m.group(1).split()], dtype=np.float64)
    return K, D

def rodrigues_to_rpy(R):
    """ 회전행렬 → roll, pitch, yaw (rad) """
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
    지면을 월드 좌표계의 Z=0 평면(n=[0,0,1]^T)으로 두었을 때,
    H = K * (R - t*n^T/d) * K^-1
    여기서 d = -t_z  (월드 평면이 Z=0이고, t는 카메라의 월드 좌표)
    """
    n = np.array([[0.0],[0.0],[1.0]])   # 지면 법선 (월드)
    d = -t[2,0]                         # 카메라의 z (평면까지 거리, 부호 주의)
    Kinv = np.linalg.inv(K)
    H = K @ (R - (t @ n.T)/d) @ Kinv
    return H

def main():
    # 1) K, D 로드
    K, D = load_ost_yaml(CAMERA_YAML)
    print("K=\n", K)
    print("D=", D)

    # 2) 이미지 로드 & 체스보드 코너 검출
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fail to read image:", IMAGE_PATH)
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (BOARD_COLS, BOARD_ROWS)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ok:
        print("Chessboard not found.")
        sys.exit(1)
    # 서브픽셀 정밀화
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # 3) 월드(지면) 좌표의 3D 점 (Z=0 평면)
    objp = np.zeros((BOARD_COLS*BOARD_ROWS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1,2)
    objp *= SQUARE_SIZE_M

    # 4) PnP → rvec, tvec
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP failed")
        sys.exit(1)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    roll, pitch, yaw = rodrigues_to_rpy(R)
    print("\n=== Extrinsics (world=board on ground, Z=0) ===")
    print("R=\n", R)
    print("t(m)=\n", t)
    print(f"roll={math.degrees(roll):.2f} deg, pitch={math.degrees(pitch):.2f} deg, yaw={math.degrees(yaw):.2f} deg")
    print(f"camera height ~= {-t[2,0]:.3f} m  (assuming Z=0 is ground)")

    # 5) 지면→이미지 호모그래피(H)와 IPM 생성
    H = build_ipm_homography_from_plane(K, R, t)

    # IPM 타겟 좌표계 구성 (실세계 X,Y 범위를 픽셀로)
    W = int(round((X_MAX - X_MIN)/INTERVAL_X))
    Hh = int(round((Y_MAX - Y_MIN)/INTERVAL_Y))
    # 타겟 평면 → 이미지 warp용 매핑 생성
    # 목적: 타겟 좌표 (u,v) 가 실세계 (x,y) 에 매핑되도록
    # dst 평면 점 집합을 만들어 inverse warp를 활용
    ipm = np.zeros((Hh, W, 3), dtype=np.uint8)

    # 타겟 평면 좌표계를 (x,y) -> 픽셀(u,v)로 정함: 
    # u축: X_MIN→X_MAX, v축: Y_MIN→Y_MAX (앞쪽이 +Y)
    # 이때 호모그래피는 "지면(X,Y,1) -> 이미지(xi, yi, w)" 이므로
    # 역변환 H_inv로 타겟 픽셀마다 소스로 샘플링
    H_inv = np.linalg.inv(H)

    # 더 빠르게 하려면 cv2.warpPerspective 사용:
    # 먼저 (x,y)를 픽셀 좌표로 선형 사상하는 3x3 행렬 A를 구성하여
    # A: (x,y,1) -> (u,v,1), 그 역행렬 A_inv와 H를 합성
    # 여기선 간단히 직접 warpPerspective 사용을 위해 dst 코너 4점을 만들자
    # 실세계에서의 사각형 코너 (좌상, 우상, 우하, 좌하) in meters:
    ground_corners = np.float32([
        [X_MIN, Y_MIN, 1.0],
        [X_MAX, Y_MIN, 1.0],
        [X_MAX, Y_MAX, 1.0],
        [X_MIN, Y_MAX, 1.0],
    ]).T  # 3x4

    # 지면 코너를 이미지로 보냄
    img_corners_h = H @ ground_corners
    img_corners = (img_corners_h[:2] / img_corners_h[2]).T.astype(np.float32)  # 4x2

    # 타겟 IPM 이미지상의 코너 픽셀 좌표
    dst_corners = np.float32([
        [0, 0],
        [W-1, 0],
        [W-1, Hh-1],
        [0, Hh-1],
    ])

    # 지면→이미지 사상으로부터, 이미지→IPM 사상 G를 추정
    # (img -> ipm) 이 되어야 warpPerspective에 바로 넣을 수 있음
    G = cv2.getPerspectiveTransform(img_corners, dst_corners)

    ipm = cv2.warpPerspective(img, G, (W, Hh))
    cv2.imwrite("ipm.png", ipm)
    print(f"\nSaved IPM to ipm.png  (size: {W}x{Hh})")

    # YAML로 extrinsics & H 저장(선택)
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
