import glob,os

from utils import render_gapart_object_with_sapien


MODES = ["RENDER_GAPARTNET_WITH_SAPIEN"]

modes = MODES[0]

# render gapartnet object with SAPIEN (point cloud)
if "RENDER_GAPARTNET_WITH_SAPIEN" in modes:
    urdf_paths = glob.glob(f"/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/*/mobility_annotation_gapartnet.urdf")
    render_gapart_object_with_sapien(urdf_paths, use_viewer=False, save_root = "sapien_render_output")


