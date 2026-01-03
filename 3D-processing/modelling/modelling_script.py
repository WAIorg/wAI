"""
Run full wAI geometry pipeline:
1) run_pixie.main()  -> generates PIXIE mesh (OBJ)
2) alpha_mesh.main() -> builds + saves alpha mesh (OBJ)
3) obj_to_volume.main()     -> loads OBJ and computes volume
"""

import os
import open3d as o3d
from modelling import alpha_mesh
from modelling import obj_to_volume
from modelling import run_pixie


def main(visualize: bool = True):
    # 1) Run PIXIE stage
    pixie_out = run_pixie.main(visualize=False, setup_pixie=False)
    print("✅ PIXIE stage complete")

    # 2) Run alpha meshing stage
    mesh = alpha_mesh.main(visualize=visualize, save=True)
    print("✅ Alpha meshing stage complete")

    # 3) Volume stage:
    #TODO: update once combining is working to use that .obj

    volume = obj_to_volume.main(pixie_out)
    print("✅ Volume computation stage complete")
    return volume

if __name__ == "__main__":
    main(visualize=False)
