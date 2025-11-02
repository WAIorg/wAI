import numpy as np

depth_img = np.load("./img_nov1/depth_raw_mm.npy")
print("Depth shape:", depth_img.shape)

sample_points = [(220, 40)]
for x, y in sample_points:
    print(f"Depth at ({x},{y}) = {depth_img[y, x]}")