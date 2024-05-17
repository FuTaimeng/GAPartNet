import glob,os
# paths = glob.glob('/home/haoran/Projects/Part/GAPartNet/manipulation/output_pred/*articulated-point_cloud.ply')
# for path in paths:
#     os.system(f"cp {path} /home/haoran/Projects/Part/GAPartNet/manipulation/output_new")

paths = glob.glob('/home/haoran/Projects/Part/GAPartNet/manipulation/gapartnet_obj_rm/*/*/*articulated-point_cloud.ply')
for path in paths:
    os.makedirs(f'/home/haoran/Projects/Part/GAPartNet/manipulation/output_new_', exist_ok=True)
    os.system(f"cp {path} /home/haoran/Projects/Part/GAPartNet/manipulation/output_new_")
    path_pred = path.replace('articulated-point_cloud.ply','output-350__.ply')
    # import pdb; pdb.set_trace()
    os.system(f"cp {path_pred} /home/haoran/Projects/Part/GAPartNet/manipulation/output_new_")