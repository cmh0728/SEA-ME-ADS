import yaml
import numpy as np

# === 1) extrinsics_and_h.yaml 에서 H 읽기 ===
with open("extrinsics_and_h.yaml", "r") as f:
    data = yaml.safe_load(f)

H = np.array(data["H_ground_to_image"], dtype=np.float64)

# === 2) IPM 영역 (solve_extrinsics_and_ipm.py와 동일하게 맞춰야 함) ===
X_MIN, X_MAX = 0.0, 0.2   # 앞 0~20cm
Y_MIN, Y_MAX = -0.2, 0.2  # 좌우 -20~20cm

# === 3) camera.yaml 의 RemapHeight / RemapWidth와 통일 ===
# camera.yaml:
# RemapHeight   : 780
# RemapWidth    : 600
H_ipm = 780   # pst_CameraData->st_CameraParameter.s32_RemapHeight
W_ipm = 600   # pst_CameraData->st_CameraParameter.s32_RemapWidth

map_x = np.zeros((H_ipm, W_ipm), np.float32)
map_y = np.zeros((H_ipm, W_ipm), np.float32)

# === 4) 각 IPM 픽셀 (r,c)에 대해 지면 좌표(X,Y) -> 이미지 좌표(u,v) 계산 ===
for r in range(H_ipm):
    # 위쪽(r=0) = X_MIN(가까운 지점), 아래(r=H_ipm-1) = X_MAX(먼 지점)
    X = X_MIN + (X_MAX - X_MIN) * r / (H_ipm - 1)
    for c in range(W_ipm):
        # 왼쪽(c=0) = Y_MIN, 오른쪽(c=W_ipm-1) = Y_MAX
        Y = Y_MIN + (Y_MAX - Y_MIN) * c / (W_ipm - 1)

        ground = np.array([X, Y, 1.0], dtype=np.float64)
        img_h = H @ ground
        u = img_h[0] / img_h[2]
        v = img_h[1] / img_h[2]

        map_x[r, c] = u
        map_y[r, c] = v

# === 5) 텍스트 파일로 저장 (LoadMappingParam과 동일한 순서) ===
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
