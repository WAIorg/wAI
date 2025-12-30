import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# ==== USER PARAMETERS ====
# Adjust these paths to where your saved images are
IR_IMAGES_DIR = "images/ir4"       # IR checkerboard images
RGB_IMAGES_DIR = "images/rgb4"     # RGB checkerboard images

CHECKERBOARD = (9, 6)  # inner corners per chessboard row/col (NOT number of squares)
SQUARE_SIZE = 0.0225    # in meters

SAVE_RESULTS = False
OUTPUT_FILE = "kinect_ir_rgb_calibration.npz"
# ==========================


def calibrate_individual_cameras(image_files, checkerboard, square_size):
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for fname in image_files:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {fname}")
            continue

        ret, corners = cv2.findChessboardCorners(img, checkerboard, None)
        if ret:
            corners_refined = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
        else:
            print(f"[WARN] Chessboard not found in {fname}")

    if len(objpoints) < 5:
        raise ValueError("Not enough valid chessboard images for calibration!")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None
    )

    return {
        "objpoints": objpoints,
        "imgpoints": imgpoints,
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "image_shape": img.shape[::-1]
    }


def stereo_calibrate(ir_calib, rgb_calib, checkerboard):
    objpoints = []
    ir_points = []
    rgb_points = []

    # Match by index (assuming same number of valid pairs)
    n_pairs = min(len(ir_calib["objpoints"]), len(rgb_calib["objpoints"]))
    for i in range(n_pairs):
        objpoints.append(ir_calib["objpoints"][i])
        ir_points.append(ir_calib["imgpoints"][i])
        rgb_points.append(rgb_calib["imgpoints"][i])

    flags = (
        cv2.CALIB_FIX_INTRINSIC  # Fix intrinsics while optimizing extrinsics
    )

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        ir_points,
        rgb_points,
        ir_calib["camera_matrix"],
        ir_calib["dist_coeffs"],
        rgb_calib["camera_matrix"],
        rgb_calib["dist_coeffs"],
        ir_calib["image_shape"],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags,
    )

    return {"R": R, "T": T, "E": E, "F": F, "reprojection_error": ret}

def visualize_overlay(ir_image_path, rgb_image_path, ir_calib, rgb_calib, R, T):
    """
    Show the IR (or depth) image aligned to RGB using stereo rectification.
    Uses matplotlib instead of cv2.imshow for Linux/Wayland compatibility.
    """
    # Load images
    ir_img = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_image_path)

    h, w = ir_img.shape

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        ir_calib["camera_matrix"], ir_calib["dist_coeffs"],
        rgb_calib["camera_matrix"], rgb_calib["dist_coeffs"],
        (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1_ir, map2_ir = cv2.initUndistortRectifyMap(
        ir_calib["camera_matrix"], ir_calib["dist_coeffs"], R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(
        rgb_calib["camera_matrix"], rgb_calib["dist_coeffs"], R2, P2, (w, h), cv2.CV_32FC1
    )

    ir_rect = cv2.remap(ir_img, map1_ir, map2_ir, cv2.INTER_LINEAR)
    rgb_rect = cv2.remap(rgb_img, map1_rgb, map2_rgb, cv2.INTER_LINEAR)

    # Convert IR to color
    ir_color = cv2.applyColorMap(cv2.convertScaleAbs(ir_rect, alpha=4), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_rect, 0.6, ir_color, 0.4, 0)

    # Convert BGR to RGB for matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    rgb_rect_rgb = cv2.cvtColor(rgb_rect, cv2.COLOR_BGR2RGB)
    ir_color_rgb = cv2.cvtColor(ir_color, cv2.COLOR_BGR2RGB)

    # Plot with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ir_color_rgb)
    axes[0].set_title("IR Rectified")
    axes[0].axis("off")

    axes[1].imshow(rgb_rect_rgb)
    axes[1].set_title("RGB Rectified")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay IR → RGB")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def refined_overlay(ir_image_path, rgb_image_path, ir_calib, rgb_calib, R, T,
                    dx=0, dy=0, scale=1.0):
    """
    Refine IR → RGB overlay by applying small pixel offsets (dx, dy) and scale.
    """
    # Load images
    ir_img = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_image_path)
    h, w = ir_img.shape

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        ir_calib["camera_matrix"], ir_calib["dist_coeffs"],
        rgb_calib["camera_matrix"], rgb_calib["dist_coeffs"],
        (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1_ir, map2_ir = cv2.initUndistortRectifyMap(
        ir_calib["camera_matrix"], ir_calib["dist_coeffs"], R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(
        rgb_calib["camera_matrix"], rgb_calib["dist_coeffs"], R2, P2, (w, h), cv2.CV_32FC1
    )

    ir_rect = cv2.remap(ir_img, map1_ir, map2_ir, cv2.INTER_LINEAR)
    rgb_rect = cv2.remap(rgb_img, map1_rgb, map2_rgb, cv2.INTER_LINEAR)

    # Apply scale and translation refinement
    M = np.array([[scale, 0, dx],
                  [0, scale, dy]], dtype=np.float32)
    ir_refined = cv2.warpAffine(ir_rect, M, (w, h))

    # Convert IR to color and overlay
    ir_color = cv2.applyColorMap(cv2.convertScaleAbs(ir_refined, alpha=4), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_rect, 0.6, ir_color, 0.4, 0)

    # Plot with matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    rgb_rect_rgb = cv2.cvtColor(rgb_rect, cv2.COLOR_BGR2RGB)
    ir_color_rgb = cv2.cvtColor(ir_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ir_color_rgb)
    axes[0].set_title("IR Rectified + Refined")
    axes[0].axis("off")

    axes[1].imshow(rgb_rect_rgb)
    axes[1].set_title("RGB Rectified")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Overlay IR → RGB (dx={dx}, dy={dy}, scale={scale:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    ir_images = sorted(glob.glob(os.path.join(IR_IMAGES_DIR, "*.png")) +
                       glob.glob(os.path.join(IR_IMAGES_DIR, "*.jpg")) +
                       glob.glob(os.path.join(IR_IMAGES_DIR, "*.webp")))
    rgb_images = sorted(glob.glob(os.path.join(RGB_IMAGES_DIR, "*.png")) +
                        glob.glob(os.path.join(RGB_IMAGES_DIR, "*.jpg")) +
                        glob.glob(os.path.join(RGB_IMAGES_DIR, "*.webp")))

    print(f"Found {len(ir_images)} IR images and {len(rgb_images)} RGB images.")

    print("\n[INFO] Calibrating IR camera...")
    ir_calib = calibrate_individual_cameras(ir_images, CHECKERBOARD, SQUARE_SIZE)

    print("\n[INFO] Calibrating RGB camera...")
    rgb_calib = calibrate_individual_cameras(rgb_images, CHECKERBOARD, SQUARE_SIZE)

    print("\n[INFO] Performing stereo calibration (IR ↔ RGB)...")
    stereo = stereo_calibrate(ir_calib, rgb_calib, CHECKERBOARD)

    print("\n=== IR Camera Intrinsics ===")
    print(ir_calib["camera_matrix"])
    print("Distortion:", ir_calib["dist_coeffs"].ravel())

    print("\n=== RGB Camera Intrinsics ===")
    print(rgb_calib["camera_matrix"])
    print("Distortion:", rgb_calib["dist_coeffs"].ravel())

    print("\n=== Extrinsics (IR → RGB) ===")
    print("Rotation matrix:\n", stereo["R"])
    print("Translation vector:\n", stereo["T"].ravel())
    print(f"Stereo reprojection error: {stereo['reprojection_error']:.4f}")

    if len(ir_images) > 0 and len(rgb_images) > 0:
        visualize_overlay(ir_images[0], rgb_images[0],
                        ir_calib, rgb_calib,
                        stereo["R"], stereo["T"])
        
    refined_overlay(
        ir_images[0], rgb_images[0],
        ir_calib, rgb_calib,
        stereo["R"], stereo["T"],
        dx=-50, dy=-2, scale=1.002
    )
    
    if SAVE_RESULTS:
        np.savez(
            OUTPUT_FILE,
            ir_camera_matrix=ir_calib["camera_matrix"],
            ir_dist_coeffs=ir_calib["dist_coeffs"],
            rgb_camera_matrix=rgb_calib["camera_matrix"],
            rgb_dist_coeffs=rgb_calib["dist_coeffs"],
            R=stereo["R"],
            T=stereo["T"],
            E=stereo["E"],
            F=stereo["F"]
        )
        print(f"\nCalibration results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
