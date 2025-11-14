import yaml
import numpy as np

with open("extrinsics_and_h.yaml", "r") as f:
    data = yaml.safe_load(f)

H = np.array(data["H_ground_to_image"], dtype=np.float64)

# --- IPM 영역 (바닥 좌표계: X=앞, Y=좌(+), 우(-)) ---
X_MIN, X_MAX = -0.3 , 0.1    # 네가 solve_extrinsics에서 썼던 값이랑 맞춰
Y_MIN, Y_MAX = -0.25 , 0.18

# --- camera.yaml 과 반드시 같게 ---
H_ipm = 600   # RemapHeight
W_ipm = 800   # RemapWidth

map_x = np.zeros((H_ipm, W_ipm), np.float32)
map_y = np.zeros((H_ipm, W_ipm), np.float32)

for r in range(H_ipm):
    # r=0 (위쪽)  -> X_MIN (가까운 쪽)
    # r=H_ipm-1   -> X_MAX (먼 쪽)
    X = X_MIN + (X_MAX - X_MIN) * r / (H_ipm - 1)

    for c in range(W_ipm):
        # c=0 (왼쪽) -> Y_MIN
        # c=W_ipm-1 -> Y_MAX
        Y = Y_MIN + (Y_MAX - Y_MIN) * c / (W_ipm - 1)

        ground = np.array([X, Y, 1.0], dtype=np.float64)
        img_h = H @ ground
        u = img_h[0] / img_h[2]
        v = img_h[1] / img_h[2]

        map_x[r, c] = u
        map_y[r, c] = v

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
