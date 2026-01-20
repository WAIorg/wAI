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
CONFIG_PATH = "/Users/adeleyounis/Desktop/Capstone/wAI/config.yaml"

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

def main(visualize: bool = True):
    paths, config = load_config(CONFIG_PATH)
    # 1) Segmentation pipeline
    point_cloud, img_rgb, x1, y1, x2, y2 = segmentation_script.run_pipeline(
        frame_rgb=paths["rgb_img_path"], 
        depth_arr=paths["depth_img_path"],
        visualize=visualize
        )
    # 2) Modelling pipeline
    mesh = modelling_script.main(
        img_rgb=img_rgb, x1=x1, y1=y1, x2=x2, 
        y2=y2, point_cloud=point_cloud, visualize=visualize
        )

    # 3) Volume calculation
    volume = obj_to_volume.main(mesh)

    # 4) Weight estimation
    weight_formula.able_body_weight_formula(sex=config["inputs"]["sex"], volume=volume, height=config["inputs"]["height"])

if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    main(visualize=False)
