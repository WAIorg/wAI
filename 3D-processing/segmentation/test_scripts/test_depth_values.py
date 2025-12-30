import numpy as np

depth_img = np.load("./img_nov1/depth_registered_3.npy")
print("Depth shape:", depth_img.shape)

sample_points = [(309, 317)]
for x, y in sample_points:
    print(f"Depth at ({x},{y}) = {depth_img[y, x]}")