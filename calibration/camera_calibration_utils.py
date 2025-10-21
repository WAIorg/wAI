import cv2
import numpy as np
import glob


def rectify_kinect_disparity(disparity_raw, x_offset=-4.8, y_offset=-3.9, interpolation=cv2.INTER_LINEAR):
    """
    Rectify Kinect disparity image by shifting it to align with the IR camera frame.

    The Kinect device can return the IR image, as well as a depth image created from the IR image. 
    There is a small, fixed offset between the two, which appears to be a consequence of the correlation window size. 
    Looking at the raw disparity image below, there is a small black band, 8 pixels wide, on the right of the image.

    Parameters
    ----------
    disparity_raw : np.ndarray
        Raw disparity map (2D array from Kinect)
    x_offset : float
        Horizontal offset (in pixels). Negative = shift left → content moves right.
    y_offset : float
        Vertical offset (in pixels). Negative = shift up → content moves down.
    interpolation : int
        OpenCV interpolation flag (e.g. cv2.INTER_LINEAR, cv2.INTER_CUBIC)

    Returns
    -------
    disparity_rectified : np.ndarray
        Offset-corrected disparity map, same size as input
    """

    # Construct affine translation matrix
    M = np.float32([[1, 0, -x_offset],   # note: negative to move the content correctly
                    [0, 1, -y_offset]])

    h, w = disparity_raw.shape
    disparity_rectified = cv2.warpAffine(
        disparity_raw, M, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # fill missing edges with 0 (no depth)
    )

    return disparity_rectified

def calibrate_kinect_cameras(ir_image_dir, rgb_image_dir, chessboard_size=(9,6), square_size=0.0245):
    """
    Calibrate Kinect IR and RGB cameras using chessboard images.
    Returns intrinsic matrices, distortion coefficients, and focal lengths.
    
    Parameters
    ----------
    ir_image_dir : str
        Path to IR chessboard images (grayscale)
    rgb_image_dir : str
        Path to RGB chessboard images
    chessboard_size : tuple
        Number of inner corners per chessboard row and column
    square_size : float
        Size of each square in meters
    
    Returns
    -------
    dict
        Dictionary with 'IR' and 'RGB' entries containing:
        - camera_matrix
        - dist_coeffs
        - fx, fy, cx, cy
    """
    def calibrate_camera(images_path, zero_distortion=False):
        images = sorted(glob.glob(images_path + "/*.png"))
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        objp *= square_size

        objpoints = []
        imgpoints = []

        for fname in images:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
            if ret:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), term)
                imgpoints.append(corners2)
                objpoints.append(objp)

        if zero_distortion:
            flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + \
                    cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
        else:
            flags = 0

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None, flags=flags)

        # If zero distortion, explicitly set all coefficients to 0
        if zero_distortion:
            dist = np.zeros((5,1))

        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        return {'camera_matrix': K, 'dist_coeffs': dist, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'rvecs': rvecs, 'tvecs': tvecs}

    ir_calib = calibrate_camera(ir_image_dir, zero_distortion=True)
    rgb_calib = calibrate_camera(rgb_image_dir, zero_distortion=False)

    return {'IR': ir_calib, 'RGB': rgb_calib}

# Example usage
kinect_calib = calibrate_kinect_cameras('calib_ir', 'calib_rgb')

print("IR camera fx, fy:", kinect_calib['IR']['fx'], kinect_calib['IR']['fy'])
print("RGB camera fx, fy:", kinect_calib['RGB']['fx'], kinect_calib['RGB']['fy'])


def calibrate_kinect_3d(ir_image_dir, rgb_image_dir,
                            chessboard_size=(9,6), square_size=0.0245):
    """
    Calibrate Kinect IR and RGB cameras and compute the stereo transform.
    
    Returns:
        calib: dict with keys 'IR', 'RGB', 'R', 'T', 'fx', 'fy', 'cx', 'cy'
    """
    
    # --- Step 1: Prepare object points ---
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size

    # --- Step 2: Detect corners ---
    def detect_corners(image_dir):
        images = sorted(glob.glob(image_dir + "/*.png"))
        objpoints = []
        imgpoints = []
        for fname in images:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
            if ret:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), term)
                objpoints.append(objp)
                imgpoints.append(corners2)
        return objpoints, imgpoints, img.shape[::-1]

    objpoints_ir, imgpoints_ir, ir_shape = detect_corners(ir_image_dir)
    objpoints_rgb, imgpoints_rgb, rgb_shape = detect_corners(rgb_image_dir)

    # --- Step 3: Calibrate individual cameras ---
    # IR: zero distortion
    flags_ir = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    ret_ir, K_ir, dist_ir, _, _ = cv2.calibrateCamera(objpoints_ir, imgpoints_ir, ir_shape, None, None, flags=flags_ir)
    dist_ir = np.zeros((5,1))  # explicitly zero distortion

    # RGB: full distortion
    ret_rgb, K_rgb, dist_rgb, _, _ = cv2.calibrateCamera(objpoints_rgb, imgpoints_rgb, rgb_shape, None, None)

    fx_ir, fy_ir = K_ir[0,0], K_ir[1,1]
    cx_ir, cy_ir = K_ir[0,2], K_ir[1,2]
    fx_rgb, fy_rgb = K_rgb[0,0], K_rgb[1,1]
    cx_rgb, cy_rgb = K_rgb[0,2], K_rgb[1,2]

    # --- Step 4: Stereo calibration (compute extrinsics) ---
    # Fix intrinsics
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    flags_stereo = cv2.CALIB_FIX_INTRINSIC

    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints_ir, imgpoints_ir, imgpoints_rgb,
        K_ir, dist_ir, K_rgb, dist_rgb,
        ir_shape, criteria=criteria, flags=flags_stereo
    )

    # --- Step 5: Rotation magnitude in degrees ---
    rvec, _ = cv2.Rodrigues(R)
    rot_deg = np.degrees(np.linalg.norm(rvec))
    rot_axis = rvec.flatten() / np.linalg.norm(rvec)

    # --- Step 6: Summary dictionary ---
    calib = {
        'IR': {'K': K_ir, 'dist': dist_ir, 'fx': fx_ir, 'fy': fy_ir, 'cx': cx_ir, 'cy': cy_ir},
        'RGB': {'K': K_rgb, 'dist': dist_rgb, 'fx': fx_rgb, 'fy': fy_rgb, 'cx': cx_rgb, 'cy': cy_rgb},
        'R': R,
        'T': T,
        'rotation_deg': rot_deg,
        'rotation_axis': rot_axis,
        'baseline_m': np.linalg.norm(T)
    }

    return calib

# --- Example usage ---
kinect_calib = calibrate_kinect_stereo('calib_ir', 'calib_rgb')

print("IR fx, fy:", kinect_calib['IR']['fx'], kinect_calib['IR']['fy'])
print("RGB fx, fy:", kinect_calib['RGB']['fx'], kinect_calib['RGB']['fy'])
print("Stereo rotation (deg):", kinect_calib['rotation_deg'])
print("Stereo rotation axis:", kinect_calib['rotation_axis'])
print("Baseline (m):", kinect_calib['baseline_m'])
print("Translation vector T:\n", kinect_calib['T'])
print("Rotation matrix R:\n", kinect_calib['R'])