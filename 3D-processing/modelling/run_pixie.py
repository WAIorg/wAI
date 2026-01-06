import os
import sys
import yaml
import torch
import subprocess
import open3d as o3d
import glob

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
    return paths


def setup_pixie_environment(pixie_dir: str):
    """Ensure PIXIE is installed and available."""
    if not os.path.exists(pixie_dir):
        print("Cloning PIXIE repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/yfeng95/PIXIE.git", pixie_dir],
            check=True,
        )
    else:
        print("PIXIE already exists at:", pixie_dir)

    # --- Patch requirements ---
    req_file = os.path.join(pixie_dir, "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file, "r") as f:
            lines = f.readlines()
        with open(req_file, "w") as f:
            for line in lines:
                # Skip old torch/kornia restrictions
                if "torch" in line or "kornia" in line:
                    continue
                f.write(line)
        print("🧩 Patched PIXIE requirements to skip old Torch/Kornia versions")

    # --- Install modern dependencies ---
    print("Installing core dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "chumpy", "open3d", "opencv-python", "smplx", "kornia>=0.6.0"],
        check=True,
    )

    print("Installing remaining PIXIE requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        check=True,
    )

    sys.path.append(pixie_dir)
    print("✅ PIXIE environment setup complete.")


def run_pixie_on_image(rgb_path: str, output_dir: str, pixie_dir: str):
    """Run PIXIE demo on a given RGB image."""
    print(f"Running PIXIE on image: {rgb_path}")
    os.makedirs(output_dir, exist_ok=True)

    demo_script = os.path.join(pixie_dir, "demos", "demo_fit_body.py")

    if not os.path.exists(demo_script):
        raise FileNotFoundError(f"PIXIE demo script not found: {demo_script}")

    subprocess.run(
        [
            sys.executable,
            demo_script,
            "-i",
            rgb_path,
            "--saveObj", "True",
            "--saveVis", "False", 
            "-s", output_dir,
            "--device", "cpu",
            "--rasterizer_type", "standard",
        ],
        check=True,
    )
    print("PIXIE run complete. Output saved to:", output_dir)

def visualize_output_mesh(output_dir: str, visualize: bool = True):
    """Visualize first .obj mesh file in output directory using Open3D."""
    objs = glob.glob(os.path.join(output_dir, "*.obj"))
    if not objs:
        print("⚠️ No .obj files found in:", output_dir)
        return

    mesh_path = objs[0]
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.1, 0.7, 0.9])
    print("Successfully loaded mesh from:", mesh_path)

    if visualize:
        o3d.visualization.draw_geometries([mesh])
    return mesh

def main(setup_pixie: bool = False, visualize: bool = True):
    CONFIG_PATH = "/Users/adeleyounis/Desktop/Capstone/wAI/config.yaml"
    paths = load_config(CONFIG_PATH)

    # Setup PIXIE
    if setup_pixie:
        setup_pixie_environment(paths["pixie_dir"])

    run_pixie_on_image(
        rgb_path=paths["rgb_img_path"],
        output_dir=paths["pose_output_dir"],
        pixie_dir=paths["pixie_dir"],
    )

    mesh = visualize_output_mesh(paths["pose_output_dir"], visualize=visualize)
    return mesh

if __name__ == "__main__":
   mesh = main(visualize=True, setup_pixie=False)