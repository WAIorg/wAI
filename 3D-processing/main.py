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
from bias_correction import bias_correction
import yaml
import trimesh
import pymeshfix
import os  
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust depth if needed
CONFIG_PATH = REPO_ROOT / "config.yaml"

def load_config(config_path: str):
    """Load and parse YAML configuration."""
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

def main(
    rgb_path: str = None,
    sex: str = None,
    height: float = None,
    visualize: bool = True,
    save: bool = True,
    progress_callback=None
):
    """
    Run full wAI software pipeline with specified paths and inputs.
    If paths are not provided, falls back to config.yaml.
    """
    if progress_callback:
        progress_callback(0, "Starting pipeline...")
    
    paths, config = load_config(CONFIG_PATH)
    
    # Use provided paths or fall back to config
    rgb_img_path = rgb_path or paths["rgb_img_path"]
    
    # Use provided inputs or fall back to config
    sex_value = sex or config["inputs"]["sex"]
    height_value = height or config["inputs"]["height"]
    
    # 1) Segmentation pipeline (0-40%)
    if progress_callback:
        progress_callback(5, "Starting segmentation pipeline...")
    img_rgb, person_segmentation_mask, x1, y1, x2, y2 = segmentation_script.run_pipeline(
        frame_rgb=rgb_img_path, 
        config=config,
        visualize=visualize, 
        save=save,
        progress_callback=lambda p, m: progress_callback(5 + p * 0.35, m) if progress_callback else None
        )
    
    # 2) Modelling pipeline (40-75%)
    if progress_callback:
        progress_callback(40, "Starting 3D modelling pipeline...")

    mesh = modelling_script.main(
        img_rgb=img_rgb, x1=x1, y1=y1, x2=x2, 
        y2=y2, person_segmentation_mask=person_segmentation_mask,
        visualize=visualize, save=save,
        progress_callback=lambda p, m: progress_callback(40 + p * 0.35, m) if progress_callback else None
        )

    # 3) Volume calculation (75-85%)
    if progress_callback:
        progress_callback(75, "Calculating volume...")
    vol = mesh.get_volume() * 1000
    print(f"Volume: {vol} cm³")

    # 4) Weight estimation (85-95%)
    if progress_callback:
        progress_callback(85, "Estimating weight...")
    print("Weight using Open3D volume:")
    weight_result = weight_formula.able_body_weight_formula(
        sex=sex_value, 
        volume=vol, 
        height=height_value
    )
    
    # 5) Bias Correction (95-100%)
    if progress_callback:
        progress_callback(85, "Applying Bias Correction ...")

    final_weight = bias_correction.apply_bias_correction(
        configs=config,
        weight=weight_result,
        sex=sex_value,
        height=height_value
    )

    if progress_callback:
        progress_callback(100, "Processing complete!")
    
    return {
        "volume": vol,
        "weight": final_weight,
        "sex": sex_value,
        "height": height_value
    }


if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    main(visualize=False, save=True)
