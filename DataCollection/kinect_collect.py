import freenect
import cv2
import numpy as np
import os
import sys

BASE_DIR = "data_collection_1"
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_video():
    frame, _ = freenect.sync_get_video()
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    if depth is None:
        return None
    return depth.astype(np.uint16)

def find_next_take_number():
    """Find the next available video take number."""
    take_number = 1
    while os.path.exists(os.path.join(VIDEOS_DIR, f"take_{take_number}")):
        take_number += 1
    return take_number

def find_next_image_number():
    """Find the next available image number."""
    img_number = 1
    while os.path.exists(os.path.join(IMAGES_DIR, f"img_{img_number}")):
        img_number += 1
    return img_number

def record_take(take_number):
    take_dir = os.path.join(VIDEOS_DIR, f"take_{take_number}")
    os.makedirs(take_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_out = cv2.VideoWriter(os.path.join(take_dir, "rgb.avi"), fourcc, 30.0, (640, 480))
    depth_out = cv2.VideoWriter(os.path.join(take_dir, "depth.avi"), fourcc, 30.0, (640, 480), False)

    depth_values = []

    print(f"Recording video take {take_number}... Press 'e' to stop.")

    while True:
        rgb = get_video()
        depth = get_depth()
        if rgb is None or depth is None:
            continue

        rgb_out.write(rgb)
        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
        depth_out.write(depth_vis)

        depth_values.append(depth.copy())

        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):  # stop recording
            break
        elif key == ord('q'):  # quit everything
            rgb_out.release()
            depth_out.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    rgb_out.release()
    depth_out.release()
    np.save(os.path.join(take_dir, "depth_raw.npy"), np.array(depth_values))
    print(f"Saved video recording to {take_dir}\n")

def save_image(img_number, rgb, depth):
    img_dir = os.path.join(IMAGES_DIR, f"img_{img_number}")
    os.makedirs(img_dir, exist_ok=True)

    cv2.imwrite(os.path.join(img_dir, "rgb.png"), rgb)

    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
    cv2.imwrite(os.path.join(img_dir, "depth.png"), depth_vis)

    np.save(os.path.join(img_dir, "depth_raw.npy"), depth)
    print(f"Saved image to {img_dir}\n")

def main():
    print("Press 'v' to start video recording, 'i' to take an image, 'q' to quit.")

    while True:
        rgb = get_video()
        depth = get_depth()
        if rgb is None or depth is None:
            continue

        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)

        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            take_number = find_next_take_number()
            record_take(take_number)
        elif key == ord('i'):
            img_number = find_next_image_number()
            save_image(img_number, rgb, depth)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
