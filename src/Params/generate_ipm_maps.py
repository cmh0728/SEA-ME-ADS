import yaml
import numpy as np

# extrinsics_and_h.yaml에서 바닥(Z=0) -> 이미지 호모그래피 H 읽기
with open("extrinsics_and_h.yaml", "r") as f:
    data = yaml.safe_load(f)

H = np.array(data["H_ground_to_image"], dtype=np.float64)

# --- IPM 영역 (바닥 좌표계: X=앞, Y=좌(+), 우(-)) ---
# 🔥 calib 스크립트에서 사용한 값과 동일하게 맞춤
X_MIN, X_MAX = -0.3, 0.05   # X_MIN: 카메라 기준 "가까운" 쪽, X_MAX: "먼" 쪽
Y_MIN, Y_MAX = -0.3, 0.3    # Y>0: 왼쪽, Y<0: 오른쪽

# --- camera.yaml 에서 사용할 IPM 해상도 ---
H_ipm = 320    # 세로 (X 축 방향: 가까운 -> 먼)
W_ipm = 400    # 가로 (Y 축 방향: 왼쪽 -> 오른쪽)

map_x = np.zeros((H_ipm, W_ipm), np.float32)
map_y = np.zeros((H_ipm, W_ipm), np.float32)

for r in range(H_ipm):
    # r = 0     → X_MIN (가까운)
    # r = H-1   → X_MAX (먼)
    X = X_MIN + (X_MAX - X_MIN) * r / (H_ipm - 1)

    for c in range(W_ipm):
        # c = 0     → Y_MAX (왼쪽)
        # c = W-1   → Y_MIN (오른쪽)
        Y = Y_MAX - (Y_MAX - Y_MIN) * c / (W_ipm - 1)

        ground = np.array([X, Y, 1.0], dtype=np.float64)
        img_h = H @ ground
        u = img_h[0] / img_h[2]
        v = img_h[1] / img_h[2]

        map_x[r, c] = u
        map_y[r, c] = v

# txt로 저장 (기존 camera 코드에서 읽어쓰는 포맷 그대로)
with open("ParamX.txt", "w") as fx:
    for r in range(H_ipm):
        for c in range(W_ipm):
            fx.write(f"{map_x[r, c]:.6f} ")
        fx.write("\n")

with open("ParamY.txt", "w") as fy:
    for r in range(H_ipm):
        for c in range(W_ipm):
            fy.write(f"{map_y[r, c]:.6f} ")
        fy.write("\n")

print("Saved ParamX.txt / ParamY.txt")
