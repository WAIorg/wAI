from modelling import modelling_script
from segmentation import segmentation_script
import yaml
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

    # Step 1: Image Segmentation
    point_cloud = segmentation_script.run_pipeline(
        frame_rgb=paths["rgb_img_path"], 
        depth_arr=paths["depth_img_path"],
        visualize=visualize
        )
    
    print("Point cloud saved at:", paths["pt_cloud_ply_path"])
    print("✅ Image segmentation stage complete")

    # Step 2: 3D Modelling
    volume = modelling_script.main(visualize=visualize)
    print("✅ 3D modelling stage complete")

    # TODO:
    # Further steps: get weight estimate, combine meshes, etc.



if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    main(visualize=False)
