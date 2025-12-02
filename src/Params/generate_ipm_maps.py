import yaml
import numpy as np

with open("extrinsics_and_h.yaml", "r") as f:
    data = yaml.safe_load(f)

H = np.array(data["H_ground_to_image"], dtype=np.float64)

# --- IPM 영역 (바닥 좌표계: X=앞, Y=좌(+), 우(-)) ---
X_MIN, X_MAX = -0.25 , 0.05 # max가 차량 뒤쪽, min이 앞쪽
Y_MIN, Y_MAX = -0.26 , 0.26   # max가 차량 왼쪽 

# --- camera.yaml 과 동일 ---
H_ipm = 320
W_ipm = 400

map_x = np.zeros((H_ipm, W_ipm), np.float32)
map_y = np.zeros((H_ipm, W_ipm), np.float32)

for r in range(H_ipm):
    # (위=가까운, 아래=먼 쪽) or 반대로 – X는 너가 원하는 방향으로 선택
    X = X_MIN + (X_MAX - X_MIN) * r / (H_ipm - 1)

    for c in range(W_ipm):
        # *** 여기만 바꿈 ***
        # c=0 (왼쪽)  -> Y_MAX (차량 왼쪽)
        # c=W-1(오른쪽)-> Y_MIN (차량 오른쪽)
        Y = Y_MAX - (Y_MAX - Y_MIN) * c / (W_ipm - 1)
        # 또는 Y_MIN/Y_MAX를 바꾸고 기존 공식을 써도 됨

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
