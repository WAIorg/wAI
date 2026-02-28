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
import time
import pandas as pd
CONFIG_PATH = "/Users/mackenziesnyder/Desktop/Capstone/wAI/config.yaml"

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

def main(file_name: str = None, sex: str = None, height: float = None, visualize: bool = True, save: bool = True):
    # start time
    start_time = time.time()

    paths, config = load_config(CONFIG_PATH)
    # 1) Segmentation pipeline
    point_cloud, img_rgb, person_segmentation_mask, x1, y1, x2, y2 = segmentation_script.run_pipeline(
        frame_rgb=f"/Users/mackenziesnyder/Desktop/Capstone/wAI/data/data_feb12/rgb/{file_name}_rgb.png", 
        depth_arr=f"/Users/mackenziesnyder/Desktop/Capstone/wAI/data/data_feb12/depth/{file_name}_depth.npy",
        config=config,
        file_name=file_name,
        visualize=visualize, 
        save=save
        )
    seg_time = time.time()
    print(f"Segmentation pipeline completed in {seg_time - start_time:.2f} seconds.")

    # 2) Modelling pipeline
    mesh = modelling_script.main(
        img_rgb=img_rgb, x1=x1, y1=y1, x2=x2, 
        y2=y2, point_cloud=point_cloud, person_segmentation_mask=person_segmentation_mask,
        file_name=file_name,
        visualize=visualize, save=save)

    print(f"Modelling pipeline completed in {time.time() - seg_time:.2f} seconds.")

    # 3) Volume calculation
    # vol = mesh.get_volume() * 1000  # in liters
    vol = obj_to_volume.main(mesh)
    # 4) Weight estimation
    print("Weight using Open3D volume:")
    weight_formula.able_body_weight_formula(sex=sex, volume=vol, height=height)

    print(f"Full pipeline completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    data = pd.read_csv("../data-collection-processing/captures.csv")
    data["estimated_weight_lbs"] = data["estimated_weight_kg"] * 2.20462
    data["height_cm"] = (
        data["height"]
        .astype(str)
        .str.replace("cm", "", regex=False)
        .str.strip()
        .astype(float)
    )

    filenames = data["rgb_path"].str.replace("_rgb.png", "")
    print(filenames)
    for f, sex_i, height_i in zip(filenames, data["sex"], data["height_cm"]):
        print("----------------------------------")
        print(f"processing {f}")
        main(file_name=f, sex=sex_i, height=height_i, visualize=False, save=True)