import freenect
import cv2
import numpy as np
import os

# Create folder to save images
save_folder = "images"
os.makedirs(save_folder, exist_ok=True)

def get_rgb():
    """Fetch RGB frame from Kinect"""
    frame, _ = freenect.sync_get_video()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def get_ir():
    """Fetch IR frame from Kinect"""
    ir_frame, _ = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)
    if ir_frame is not None:
        ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    return ir_frame

count = 0
while True:
    rgb = get_rgb()
    ir = get_ir()

    if rgb is None or ir is None:
        continue

    # Stack images side by side
    combined = np.hstack((rgb, ir))
    cv2.imshow("RGB | IR", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save both images
        rgb_path = os.path.join(save_folder, f"rgb_{count:03d}.png")
        ir_path = os.path.join(save_folder, f"ir_{count:03d}.png")
        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(ir_path, ir)
        print(f"Saved {rgb_path} and {ir_path}")
        count += 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
freenect.sync_stop()
