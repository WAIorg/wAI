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

def main(file_name: str = None, visualize: bool = True, save: bool = True):
    # start time
    start_time = time.time()

    paths, config = load_config(CONFIG_PATH)
    # 1) Segmentation pipeline
    point_cloud, img_rgb, person_segmentation_mask, x1, y1, x2, y2 = segmentation_script.run_pipeline(
        frame_rgb=f"/Users/adeleyounis/Desktop/Capstone/wAI/3D-processing/data/rgb/{file_name}_rgb.png", 
        depth_arr=f"/Users/adeleyounis/Desktop/Capstone/wAI/3D-processing/data/depth/{file_name}_depth.npy",
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
    weight_formula.able_body_weight_formula(sex=config["inputs"]["sex"], volume=vol, height=config["inputs"]["height"])

    print(f"Full pipeline completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    filenames = [
    "20260212_082208",
    "20260212_082713",
    "20260212_083231",
    "20260212_083956",
    "20260212_094222",
    "20260212_094752",
    "20260212_095222",
    "20260212_095815",
    "20260212_100209",
    "20260212_100614",
    "20260212_101125",
    "20260212_101455",
    "20260212_101840",
    "20260212_102239",
    "20260212_102631",
    "20260212_103043",
    "20260212_103653",
    "20260212_104039",
    "20260212_104412",
    "20260212_105015",
    "20260212_110016",
    "20260212_110345",
    "20260212_110737",
    "20260212_111103",
    "20260212_111425",
    "20260212_111856",
    "20260212_113317",
    "20260212_113653",
    "20260212_114240",
    "20260212_114632",
    "20260212_115330",
    "20260212_115743",
    "20260212_122753",
    "20260212_123134",
    "20260212_123501",
    "20260212_123815",
    "20260212_124148",
    "20260212_124538",
    "20260212_125605",
    "20260212_130006",
    "20260212_130349",
    "20260212_130729",
    "20260212_131144",
    "20260212_135516",
    "20260212_140134",
    "20260212_142002"
    ]
    for f in filenames:
        print("----------------------------------")
        print(f"processing {f}")
        main(file_name=f, visualize=False, save=True)
