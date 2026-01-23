"""
Run full wAI software pipeline:
1) Runs segmentation pipeline
2) Runs modelling pipeline
3) Calculates volume from mesh
4) Estimates weight from volume
"""

from modelling import obj_to_volume, modelling_script
from weight_calculation import weight_formula
from segmentation import segmentation_script
import yaml
import trimesh
import pymeshfix
import os  
from pathlib import Path
from typing import Optional
import glob
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust depth if needed
CONFIG_PATH = REPO_ROOT / "config.yaml"

def load_config(config_path: str):
    """Load and parse YAML configuration."""
    print("load yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(base_dir, rel_path))

    # Resolve all paths
    paths = {k: resolve_path(v) for k, v in config.get("paths", {}).items()}
    return paths, config

def get_most_recent_images(data_dir: str = "backend/data"):
    """
    Get the most recent RGB image and depth array from the data folder.
    Returns tuple of (rgb_path, depth_path) or (None, None) if not found.
    """
    print("getting recent images")
    data_path = Path(data_dir)
    if not data_path.exists():
        # Try relative to repo root
        data_path = REPO_ROOT / data_dir
        if not data_path.exists():
            return None, None
    
    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth_arrays"
    
    # Find most recent RGB image
    rgb_files = list(rgb_dir.glob("*_rgb.png")) if rgb_dir.exists() else []
    if not rgb_files:
        return None, None
    
    # Find most recent depth array
    depth_files = list(depth_dir.glob("*_depth.npy")) if depth_dir.exists() else []
    if not depth_files:
        return None, None
    
    # Sort by modification time and get most recent
    rgb_path = max(rgb_files, key=os.path.getmtime)
    depth_path = max(depth_files, key=os.path.getmtime)
    
    # Extract timestamps to match them
    rgb_timestamp = rgb_path.stem.replace("_rgb", "")
    depth_timestamp = depth_path.stem.replace("_depth", "")
    
    # Try to find matching timestamp pair
    if rgb_timestamp == depth_timestamp:
        return str(rgb_path), str(depth_path)
    
    # If timestamps don't match, return most recent of each
    return str(rgb_path), str(depth_path)

def main(
    visualize: bool = True, 
    save: bool = True,
    rgb_path: Optional[str] = None,
    depth_path: Optional[str] = None,
    sex: Optional[str] = None,
    height: Optional[float] = None,
    use_latest: bool = False
):
    """
    Run the full pipeline.
    
    Args:
        visualize: Whether to visualize intermediate results
        save: Whether to save intermediate results
        rgb_path: Path to RGB image (overrides config and use_latest)
        depth_path: Path to depth array (overrides config and use_latest)
        sex: Sex for weight calculation (overrides config)
        height: Height in cm for weight calculation (overrides config)
        use_latest: If True, use most recent images from data folder
    """
    paths, config = load_config(CONFIG_PATH)
    
    print("running full pipline")
    # Determine which paths to use
    if rgb_path and depth_path:
        # Use provided paths
        final_rgb_path = rgb_path
        final_depth_path = depth_path
    elif use_latest:
        # Get most recent from data folder
        final_rgb_path, final_depth_path = get_most_recent_images()
        if not final_rgb_path or not final_depth_path:
            raise ValueError("Could not find recent images in data folder. Ensure images are saved.")
    else:
        # Use config paths
        final_rgb_path = paths["rgb_img_path"]
        final_depth_path = paths["depth_img_path"]
    
    # Determine sex and height
    final_sex = sex if sex is not None else config["inputs"]["sex"]
    final_height = height if height is not None else config["inputs"]["height"]
    
    print(f"Using RGB: {final_rgb_path}")
    print(f"Using Depth: {final_depth_path}")
    print(f"Sex: {final_sex}, Height: {final_height}cm")
    
    # 1) Segmentation pipeline
    point_cloud, img_rgb, x1, y1, x2, y2 = segmentation_script.run_pipeline(
        frame_rgb=final_rgb_path, 
        depth_arr=final_depth_path,
        visualize=visualize, 
        save=save
        )
    # 2) Modelling pipeline
    mesh = modelling_script.main(
        img_rgb=img_rgb, x1=x1, y1=y1, x2=x2, 
        y2=y2, point_cloud=point_cloud, 
        visualize=visualize, save=save
        )

    # 3) Volume calculation
    vol = mesh.get_volume() * 1000
    print(f"Volume: {vol} cm³")

    # 4) Weight estimation
    print("Weight using Open3D volume:")
    weight_formula.able_body_weight_formula(sex=final_sex, volume=vol, height=final_height)
    
    return vol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wAI 3D processing pipeline")
    parser.add_argument("--rgb", type=str, help="Path to RGB image (overrides config)")
    parser.add_argument("--depth", type=str, help="Path to depth array (overrides config)")
    parser.add_argument("--sex", type=str, help="Sex for weight calculation (Male/Female)")
    parser.add_argument("--height", type=float, help="Height in cm for weight calculation")
    parser.add_argument("--use-latest", action="store_true", help="Use most recent images from data folder")
    parser.add_argument("--visualize", action="store_true", help="Visualize intermediate results")
    parser.add_argument("--save", action="store_true", default=True, help="Save intermediate results")
    
    args = parser.parse_args()
    
    # If no arguments provided, use config
    if not any([args.rgb, args.depth, args.use_latest]):
        paths, config = load_config(CONFIG_PATH)
        main(visualize=args.visualize, save=args.save)
    else:
        main(
            visualize=args.visualize,
            save=args.save,
            rgb_path=args.rgb,
            depth_path=args.depth,
            sex=args.sex,
            height=args.height,
            use_latest=args.use_latest
        )
