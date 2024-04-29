#############
# code name: articualted object manipulation
# description: articualted object manipulation, we put several random object in 
#              the scene we control the fixed franka arm to manipulate the part 
#              on the GAPartNet object. we use the annotation from GAPartNet to 
#              get the part information. If you feel the code useful, please 
#              cite the following paper:
#
#              @article{geng2022gapartnet,
#                title={GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts},
#                author={Geng, Haoran and Xu, Helin and Zhao, Chengyang and Xu, Chao and Yi, Li and Huang, Siyuan and Wang, He},
#                journal={arXiv preprint arXiv:2211.05272},
#                year={2022}
#              }
#
#              @misc{geng2023sage,
#              title={SAGE: Bridging Semantic and Actionable Parts for GEneralizable Articulated-Object Manipulation under Language Instructions},
#              author={Haoran Geng and Songlin Wei and Congyue Deng and Bokui Shen and He Wang and Leonidas Guibas},
#              year={2023},
#              eprint={2312.01307},
#              archivePrefix={arXiv},
#              primaryClass={cs.RO}
#              }
#
#              @article{geng2023partmanip,
#              title={PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations},
#              author={Geng, Haoran and Li, Ziming and Geng, Yiran and Chen, Jiayi and Dong, Hao and Wang, He},
#              journal={arXiv preprint arXiv:2303.16958},
#              year={2023}
#              }
# code author: Haoran Geng
#############
# exit()
from object_gym import ObjectGym
import numpy as np
from utils import read_yaml_config, prepare_gsam_model, images_to_video
import torch
import glob
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import sys
import tqdm
import os, imageio
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert
import random

sys.path.append(sys.path[-1]+"/gym")
torch.set_printoptions(precision=4, sci_mode=False)

# load arguments
args = gymutil.parse_arguments(description="Placement",
    custom_parameters=[
        {"name": "--mode", "type": str, "default": ""},
        {"name": "--task_root", "type": str, "default": "output"},
        {"name": "--config", "type": str, "default": "config"},
        {"name": "--device", "type": str, "default": "cuda"},
        # headless
        {"name": "--headless", "action": 'store_true', "default": False},
        {"name": "--save_video", "action": 'store_true', "default": False},
        {"name": "--save_info", "action": 'store_true', "default": False},
        ])

def init_gym(cfgs, task_cfg=None):
    '''
    function: init gym
    input: cfgs, task_cfg
    '''
    # init gsam
    if cfgs["INFERENCE_GSAM"]:
        grounded_dino_model, sam_predictor = prepare_gsam_model(device=args.device)
    else:
        grounded_dino_model, sam_predictor = None, None
        
    # load selected object information (not important for articulated object manipulation)
    selected_obj_names = task_cfg["selected_obj_names"]
    selected_obj_urdfs=task_cfg["selected_urdfs"]
    selected_obj_num = len(selected_obj_names)
    selected_ob_poses = task_cfg["init_obj_pos"]
    selected_ob_pose_rs = [pose[3:] for pose in selected_ob_poses]
    save_root = task_cfg["save_root"]
    cfgs["asset"]["position_noise"] = [0,0,0]
    cfgs["asset"]["rotation_noise"] = 0
    cfgs["asset"]["asset_files"] = selected_obj_urdfs
    cfgs["asset"]["asset_seg_ids"] = [2 + i for i in range(selected_obj_num)]
    cfgs["asset"]["obj_pose_ps"] = selected_ob_poses
    cfgs["asset"]["obj_pose_rs"] = selected_ob_pose_rs

    # init gym
    gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    
    # refresh observation and run steps to initialize the scene
    gym.refresh_observation(get_visual_obs=False)
    gym.run_steps(pre_steps = 10, refresh_obs=False, print_step=False)
    gym.refresh_observation(get_visual_obs=False)
    gym.save_root = save_root
    
    return gym, cfgs

if args.mode == "run_arti_open":
    '''
    function: init gym and run open demo
    '''
    
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    for path in tqdm.tqdm(paths, total=len(paths)):
        # get gapart id and anno
        gapart_id = path.split("/")[-2]
        gapart_anno_path = "/".join(path.split("/")[:-1]) + "/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path, "r"))
        for link_anno in gapart_anno:
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                pass
        
        # cfg loading and init gym
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        
        # load articualted object with the bottom at z = 0
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
            
        # set the save root and other configurations
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        
        if args.save_video:
            import datetime
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_video_root = f"output/{current_time_str}"
            os.makedirs(save_video_root, exist_ok=True)
        else:
            save_video_root = None
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)
        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if not cfgs["HEADLESS"] and True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        bbox_now_i = all_bbox_now[part_i]
                        for i in range(len(idx_set)):
                            gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                np.array([1, 0 ,0], dtype=np.float32))
        
        
        # manipulate the object with the last part, change it for other objects
        # TODO: change the bbox_id to manipulate parts using annotated semantics
        bbox_id = -1
        # get the part bbox and calculate the handle direction
        all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32).to(gym.device).reshape(-1, 8, 3)
        all_bbox_center_front_face = torch.mean(all_bbox_now[:,0:4,:], dim = 1) 
        handle_out = all_bbox_now[:,0,:] - all_bbox_now[:,4,:]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = all_bbox_now[:,0,:] - all_bbox_now[:,1,:]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = all_bbox_now[:,0,:] - all_bbox_now[:,3,:]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), 
                        handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        init_position = all_bbox_center_front_face[bbox_id].cpu().numpy()
        handle_out_ = handle_out[bbox_id].cpu().numpy()
        
        # move the object to the pre-grasp position
        pre_grasp_position = init_position + 0.2 * handle_out_
        interaction_infos = []
        delta = np.array([0,0,0,0,0,0,0])
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*pre_grasp_position,*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # move the object to the grasp position
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.2-0.1) * handle_out_),*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # close the gripper
        for i in range(3): info = gym.move_gripper(
            close_gripper = True, save_video=args.save_video, save_root = save_video_root, 
            ); interaction_infos+=info
        
        # move the object to the lift position
        for i in range(30): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.1+i*0.01) * handle_out_),*(rotations[bbox_id].cpu().numpy())]+delta), 
            close_gripper = True, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info

        info = {"interaction": interaction_infos,"cfg": cfgs,"task_cfg": task_cfg, "gapart_id": gapart_id}
        # save info
        if args.save_info:
            np.save(f"{save_video_root}/interaction_infos.npy", info, allow_pickle=True)
        if args.save_video:
            images_to_video(f"{save_video_root}/video", f"{save_video_root}/video.mp4")
        # run the simulation for more visualization, comment it if you don't need it
        print("Finish the manipulation, run the simulation 1000 steps for more visualization")
        gym.run_steps(pre_steps = 1000, refresh_obs=False, print_step=False)
        
        # clean up for the next object
        gym.clean_up()
        del gym
if args.mode == "run_arti_open_random_drawer":
    '''
    function: init gym and run open demo
    '''
    
    ROOT = "gapartnet"
    # read all paths
    # we choose one example object to show the demo, change the path
    arti_task_cfgs = json.load(open("assets/tasks_open_drawer_with_handle.json", "r"))
    random.shuffle(arti_task_cfgs)
    for arti_task_cfg in tqdm.tqdm(arti_task_cfgs, total=len(arti_task_cfgs)):
        gapart_id = arti_task_cfg["id"]
        if gapart_id in ["30666", "47711", "47712", "47713", "47714", "47715", "47716", "47717", "47718", "47719", "47720"]:
            continue
        part_anno = arti_task_cfg["part_anno"]
        drawer_link_name = part_anno["link_name"]
        handle_link_name = arti_task_cfg["child_anno"]["link_name"]
        gapart_anno_path = f"assets/{ROOT}/{gapart_id}/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path, "r"))
        for link_anno in gapart_anno:
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                pass
        
        # cfg loading and init gym
        cfgs = read_yaml_config(f"{args.config}.yaml")
        obj_task_root = args.task_root
        obj_task_cfgs_path = "task_config.json"
        with open(obj_task_cfgs_path, "r") as f: obj_task_cfg = json.load(f)
        
        # load articualted object with the bottom at z = 0
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
            
        # set the save root and other configurations
        obj_task_cfg["save_root"] = "/".join(obj_task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        
        if args.save_video:
            import datetime
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_video_root = f"/media/haoran/Elements/output_drawer/{gapart_id}-{drawer_link_name}-{handle_link_name}/{current_time_str}"
            os.makedirs(save_video_root, exist_ok=True)
        else:
            save_video_root = None
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.1
        cfgs["asset"]["arti_rotation_noise"] = 10
        cfgs["asset"]["arti_dof_noise"] = 0.1
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=obj_task_cfg)
        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        # import pdb; pdb.set_trace()
        # gym.arti_init_obj_pos_list
        # gym.arti_init_obj_rot_list
        
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if not cfgs["HEADLESS"] and True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        bbox_now_i = all_bbox_now[part_i]
                        for i in range(len(idx_set)):
                            gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                np.array([1, 0 ,0], dtype=np.float32))

        # manipulate the object with the last part, change it for other objects
        # TODO: change the bbox_id to manipulate parts using annotated semantics
        bbox_id = [i for i, link_name in enumerate(gym.gapart_link_names[0]) if link_name == handle_link_name][0]
        # get the part bbox and calculate the handle direction
        all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32).to(gym.device).reshape(-1, 8, 3)
        all_bbox_center_front_face = torch.mean(all_bbox_now[:,0:4,:], dim = 1) 
        handle_out = all_bbox_now[:,0,:] - all_bbox_now[:,4,:]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = all_bbox_now[:,0,:] - all_bbox_now[:,1,:]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = all_bbox_now[:,0,:] - all_bbox_now[:,3,:]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), 
                        handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        init_position = all_bbox_center_front_face[bbox_id].cpu().numpy()
        handle_out_ = handle_out[bbox_id].cpu().numpy()
        
        # move the object to the pre-grasp position
        pre_grasp_position = init_position + 0.2 * handle_out_
        interaction_infos = []
        delta = np.array([0,0,0,0,0,0,0])
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*pre_grasp_position,*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # move the object to the grasp position
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.2-0.1) * handle_out_),*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # close the gripper
        for i in range(3): info = gym.move_gripper(
            close_gripper = True, save_video=args.save_video, save_root = save_video_root, 
            ); interaction_infos+=info
        
        # move the object to the lift position
        for i in range(30): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.1+i*0.01) * handle_out_),*(rotations[bbox_id].cpu().numpy())]+delta), 
            close_gripper = True, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info

        gym.refresh_observation(get_visual_obs=False)
        parent_link_joint_name = part_anno["parents_joint"][0]
        link_id = gym.gym.get_actor_dof_dict(gym.envs[0], 2)[parent_link_joint_name]
        open_dof = gym.arti_obj_dof_qpos_qvel[0][link_id][0]
        SUCCESS_THRESHOLD = 0.3
        success = open_dof > gym.arti_obj_dof_lower[link_id] + SUCCESS_THRESHOLD*(gym.arti_obj_dof_upper[link_id] - gym.arti_obj_dof_lower[link_id])
        print("#"*20, "Success or NOT: ", success.item(), open_dof.item(), "#"*20)
        info = {"interaction": interaction_infos,"cfg": cfgs,"obj_task_cfg": obj_task_cfg, "gapart_id": gapart_id, "success": success}
        # save info
        if args.save_info:
            np.save(f"{save_video_root}/interaction_infos.npy", info, allow_pickle=True)
        if args.save_video:
            images_to_video(f"{save_video_root}/video", f"{save_video_root}/video.mp4")
        # run the simulation for more visualization, comment it if you don't need it
        print("Finish the manipulation, run the simulation 1000 steps for more visualization")
        # gym.run_steps(pre_steps = 1000, refresh_obs=False, print_step=False)
        
        # clean up for the next object
        gym.clean_up()
        del gym

elif args.mode == "run_arti_other":
    '''
    function: init gym and run open demo
    '''
    
    ROOT = "gapartnet"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/47711/mobility_annotation_gapartnet.urdf")
    # paths = [""]
    for path in tqdm.tqdm(paths, total=len(paths)):
        # get gapart id and anno
        gapart_id = path.split("/")[-2]
        gapart_anno_path = "/".join(path.split("/")[:-1]) + "/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path, "r"))
        for link_anno in gapart_anno:
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                pass
        
        # cfg loading and init gym
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        
        # load articualted object with the bottom at z = 0
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
            
        # set the save root and other configurations
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        
        if args.save_video:
            import datetime
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_video_root = f"output/{current_time_str}"
            os.makedirs(save_video_root, exist_ok=True)
        else:
            save_video_root = None
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)
        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if not cfgs["HEADLESS"] and False:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        part_i  = -1
                        bbox_now_i = all_bbox_now[part_i]
                        for i in range(len(idx_set)):
                            gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                np.array([1, 0 ,0], dtype=np.float32))
        # import pdb; pdb.set_trace()
        
        # manipulate the object with the last part, change it for other objects
        # TODO: change the bbox_id to manipulate parts using annotated semantics
        bbox_id = -1
        # get the part bbox and calculate the handle direction
        all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32).to(gym.device).reshape(-1, 8, 3)
        all_bbox_center_front_face = torch.mean(all_bbox_now[:,0:4,:], dim = 1) 
        handle_out = all_bbox_now[:,0,:] - all_bbox_now[:,4,:]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = all_bbox_now[:,0,:] - all_bbox_now[:,1,:]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = all_bbox_now[:,0,:] - all_bbox_now[:,3,:]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), 
                        handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        init_position = all_bbox_center_front_face[bbox_id].cpu().numpy()
        handle_out_ = handle_out[bbox_id].cpu().numpy()
        
        # move the object to the pre-grasp position
        pre_grasp_position = init_position + 0.2 * handle_out_
        interaction_infos = []
        delta = np.array([0.05,0,0,0,0,0,0])
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*pre_grasp_position,*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # move the object to the grasp position
        for i in range(5): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.2-0.1) * handle_out_),*(rotations[bbox_id].cpu().numpy())])+delta, 
            close_gripper = False, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info
        
        # close the gripper
        for i in range(3): info = gym.move_gripper(
            close_gripper = True, save_video=args.save_video, save_root = save_video_root, 
            ); interaction_infos+=info
        
        # move the object to the lift position
        for i in range(30): traj, info = gym.control_to_pose(
            np.array([*(init_position + (0.1+i*0.01) * handle_out_),*(rotations[bbox_id].cpu().numpy())]+delta), 
            close_gripper = True, save_video = args.save_video, save_root = save_video_root, 
            use_ik = True); interaction_infos+=info

        info = {"interaction": interaction_infos,"cfg": cfgs,"task_cfg": task_cfg, "gapart_id": gapart_id}
        # save info
        if args.save_info:
            np.save(f"{save_video_root}/interaction_infos.npy", info, allow_pickle=True)
        if args.save_video:
            images_to_video(f"{save_video_root}/video", f"{save_video_root}/video.mp4")
        # run the simulation for more visualization, comment it if you don't need it
        print("Finish the manipulation, run the simulation 1000 steps for more visualization")
        gym.run_steps(pre_steps = 1000, refresh_obs=False, print_step=False)
        
        # clean up for the next object
        gym.clean_up()
        del gym
  
elif args.mode == "run_arti_free_control":
    '''
    function: init gym and run free control
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    
    # we choose one example object to show the demo, change the path 
    # to the object you want to show!
    paths = ["../partnet_mobility_part/45661/mobility_annotation_gapartnet.urdf"]
    for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        cfgs["USE_CUROBO"] = True
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [0.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        print(gym.save_root)
        gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        
        ############################ change to desired pose ############################
        rotation = np.array([0, 1, 0, 0])
        position = np.array([0.2502,     -0.2000,     0.8517])
        move_pose = np.concatenate([position, rotation])
        ################################################################################
        
        traj = gym.control_to_pose(move_pose, close_gripper = True, save_video = False, save_root = None)
        
        gym.clean_up()
        del gym     

elif args.mode == "run_arti_circle":
    '''
    function: init gym and run free control
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    
    # we choose one example object to show the demo, change the path 
    # to the object you want to show!
    paths = ["../partnet_mobility_part/45661/mobility_annotation_gapartnet.urdf"]
    for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        task_cfg["init_obj_pos"][0][0] = 2
        task_cfg["init_obj_pos"][1][0] = 2
        cfgs["HEADLESS"] = args.headless
        
        if args.save_video:
            import datetime
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_video_root = f"output/{current_time_str}"
            os.makedirs(save_video_root, exist_ok=True)
        else:
            save_video_root = None
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [1.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        print(gym.save_root)
        gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        
        ############################ change to desired pose ############################
        rotation = np.array([1, 0, 0, 0])
        position = np.array([0.2502,     -0.2000,     0.8517])
        move_pose = np.concatenate([position, rotation])
        positions = [(-1.2246467991473532e-16, -2.0),
        (-0.3246994692046836, -1.9458172417006345),
        (-0.6142127126896679, -1.7891405093963935),
        (-0.8371664782625287, -1.5469481581224267),
        (-0.9694002659393304, -1.2454854871407992),
        (-0.9965844930066698, -0.9174206545276676),
        (-0.9157733266550575, -0.5983045753470306),
        (-0.7357239106731317, -0.322718428374259),
        (-0.47594739303707356, -0.12052624879351093),
        (-0.16459459028073392, -0.013638696597277677),
        (0.16459459028073392, 0.013638696597277677),
        (0.47594739303707356, 0.12052624879351093),
        (0.7357239106731314, 0.3227184283742587),
        (0.9157733266550573, 0.5983045753470302),
        (0.9965844930066698, 0.9174206545276674),
        (0.9694002659393305, 1.245485487140799),
        (0.8371664782625287, 1.5469481581224267),
        (0.6142127126896679, 1.7891405093963935),
        (0.3246994692046836, 1.9458172417006345),
        (1.2246467991473532e-16, 2.0)]
        positions = np.array(positions)
        positions = positions * 0.1 + np.array([0.4,0.65])
        ################################################################################
        
        interaction_infos = []
        for pos_i, posi in enumerate(positions):

            run_times = 0
            if pos_i ==0 :
                run_times = 5
            else:
                run_times = 2
            for i in range(run_times): traj, info = gym.control_to_pose(
                np.array([posi[0], 0.0, posi[1],*rotation]), 
                close_gripper = True, save_video = args.save_video, save_root = save_video_root, 
                use_ik = True); interaction_infos+=info
            print([posi[0], 0.0, posi[1]])
        info = {"interaction": interaction_infos,"cfg": cfgs,"task_cfg": task_cfg, "gapart_id": gapart_id}
        # save info
        if args.save_info:
            np.save(f"{save_video_root}/interaction_infos.npy", info, allow_pickle=True)
        if args.save_video:
            images_to_video(f"{save_video_root}/video", f"{save_video_root}/video.mp4")
        gym.clean_up()
        del gym     
         
elif args.mode == "run_arti_render":
    '''
    function: init gym and run render code, render the articulated object point cloud
    '''
    ROOT = "gapartnet"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/45661/remo*.urdf")
    arti_task_cfgs = json.load(open("assets/tasks_open_drawer_with_handle.json", "r"))
    random.shuffle(arti_task_cfgs)
    for arti_task_cfg in tqdm.tqdm(arti_task_cfgs, total=len(arti_task_cfgs)):
        gapart_id = arti_task_cfg["id"]
        if gapart_id in ["30666", "47711", "47712", "47713", "47714", "47715", "47716", "47717", "47718", "47719", "47720"]:
            continue
        part_anno = arti_task_cfg["part_anno"]
        drawer_link_name = part_anno["link_name"]
        paths = glob.glob(f"assets/{ROOT}/{gapart_id}/remo*{drawer_link_name}*.urdf")
        assert len(paths) == 1
        path = paths[0]
    # paths -= unused_paths
    # for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        if gapart_id in ["102278","103989","103560", "103863","103425", 
                         "103869", "47315", "47613", "48018", "47290", "49062",
                         "41003","46456","45203"]:
            continue
        save_dir = f"gapartnet_obj_rm_new/{gapart_id}"
        save_name = gapart_id
        rm_name = path.split("/")[-1].split(".")[0]
        fname = os.path.join(save_dir, f"{save_name}-{rm_name}-articulated-point_cloud.ply")
        print("processing ", gapart_id)
        if os.path.exists(fname):
            print("skip", fname)
            continue
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        cfgs["HEADLESS"] = args.headless
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_obj_pose_ps"] = [[0,0, 3]]
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 10.0
        cfgs["asset"]["arti_dof_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 1.0
        cfgs["asset"]["arti_rotation"] = 0
        
        cfgs["asset"]["arti_urdf_name"] = path.split("/")[-1].split(".")[0]
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["cam"]["point_cloud_bound"] = [            
            [-1, 1],
            [-1, 1],
            [1, 10.0]
        ]
        cfgs["cam"]["cam_poss"] = [
            [0.1, 0, 1],
            [0.1, 0, 5],
            [0, 3, 3.0],
            [-3, 0, 3.0],
            [3, 0, 3.0],
            [0, -3, 3.0],
        ]
        cfgs["cam"]["cam_targets"] = [
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
        ]
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])

        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)
        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # # render bbox for visualization and debug
        # if not cfgs["HEADLESS"] and True:
        #     gym.gym.clear_lines(gym.viewer)
        # for env_i in range(gym.num_envs):
        #     for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
        #         all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
        #         rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
        #         rotation_matrix = rotation.as_matrix()
        #         rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
        #         all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
        #         if not cfgs["HEADLESS"] and True:
        #             idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
        #             for part_i in range(len(gapart_raw_valid_anno)):
        #                 bbox_now_i = all_bbox_now[part_i]
        #                 for i in range(len(idx_set)):
        #                     gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
        #                         np.concatenate((bbox_now_i[idx_set[i][0]], 
        #                                         bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
        #                         np.array([1, 0 ,0], dtype=np.float32))
        
        print(gym.save_root)
        gym.run_steps(pre_steps = 3, refresh_obs=False, print_step=False)
        
        ## render
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        
        for img_i,rgb in enumerate(rgb_envs[0]):
            img_i_str = str(img_i).zfill(3)
            os.makedirs(save_dir+f"/{rm_name}", exist_ok=True)
            imageio.imwrite(f"{save_dir}/{rm_name}/{save_name}-{rm_name}-img-{img_i_str}.png", rgb)
        # import pdb; pdb.set_trace()
        os.makedirs(f"{save_dir}", exist_ok=True)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_envs[0][:, :3]-np.array(cfgs["asset"]["arti_obj_pose_ps"][0]))
        point_cloud.colors = o3d.utility.Vector3dVector(colors_envs[0][:, :3]/255.0)
        # save_to ply
        fname = os.path.join(save_dir, rm_name, f"{save_name}-{rm_name}-articulated-point_cloud.ply")
        o3d.io.write_point_cloud(fname, point_cloud)
        
        # import pdb; pdb.set_trace()
        rm_link_name = path.split("/")[-1].split(".")[0].split("-")[-1]
        bbox_id = [i for i, link_name in enumerate(gym.gapart_link_names[0]) if link_name == rm_link_name][0]
        
        bbox_info = None
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        if bbox_id != part_i:
                            continue
                        bbox_now_i = all_bbox_now[part_i]
                        bbox_info = {
                            "bbox_id": bbox_id,
                            "bbox": bbox_now_i,
                            "rot": gym.arti_init_obj_rot_list[env_i],
                            "pos": gym.arti_init_obj_pos_list[env_i],
                            "scale": cfgs["asset"]["arti_obj_scale"]
                        }
                        for i in range(len(idx_set)):
                            if not cfgs["HEADLESS"]:
                                gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                    np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                    bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                    np.array([1, 0 ,0], dtype=np.float32))
        # gym.run_steps(pre_steps = 3000, refresh_obs=False, print_step=False)
        np.save(f"{save_dir}/{rm_name}/{save_name}-{rm_name}-bbox_info.npy", bbox_info, allow_pickle=True)
        gym.clean_up()
        del gym
         
elif args.mode == "run_arti_render_eval":
    '''
    function: init gym and run render code, render the articulated object point cloud
    '''
    ROOT = "gapartnet"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/45661/remo*.urdf")
    arti_task_cfgs = json.load(open("assets/tasks_open_drawer_with_handle.json", "r"))
    random.shuffle(arti_task_cfgs)
    for arti_task_cfg in tqdm.tqdm(arti_task_cfgs, total=len(arti_task_cfgs)):
        gapart_id = arti_task_cfg["id"]
        if gapart_id in ["30666", "47711", "47712", "47713", "47714", "47715", "47716", "47717", "47718", "47719", "47720"]:
            continue
        part_anno = arti_task_cfg["part_anno"]
        drawer_link_name = part_anno["link_name"]
        paths = glob.glob(f"assets/{ROOT}/{gapart_id}/remo*{drawer_link_name}*.urdf")
        assert len(paths) == 1
        path = paths[0]
    # paths -= unused_paths
    # for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        if gapart_id in ["102278","103989","103560", "103863","103425", 
                         "103869", "47315", "47613", "48018", "47290", "49062",
                         "41003","46456","45203"]:
            continue
        save_dir = f"gapartnet_obj_rm_new/{gapart_id}"
        save_name = gapart_id
        rm_name = path.split("/")[-1].split(".")[0]
        fname = os.path.join(save_dir, f"{save_name}-{rm_name}-articulated-point_cloud.ply")
        print("processing ", gapart_id)
        if os.path.exists(fname):
            print("skip", fname)
            continue
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        cfgs["HEADLESS"] = args.headless
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_obj_pose_ps"] = [[0,0, 3]]
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 10.0
        cfgs["asset"]["arti_dof_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 1.0
        cfgs["asset"]["arti_rotation"] = 0
        
        cfgs["asset"]["arti_urdf_name"] = path.split("/")[-1].split(".")[0]
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["cam"]["point_cloud_bound"] = [            
            [-1, 1],
            [-1, 1],
            [1, 10.0]
        ]
        cfgs["cam"]["cam_poss"] = [
            [0.1, 0, 1],
            [0.1, 0, 5],
            [0, 3, 3.0],
            [-3, 0, 3.0],
            [3, 0, 3.0],
            [0, -3, 3.0],
        ]
        cfgs["cam"]["cam_targets"] = [
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
        ]
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])

        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)
        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # # render bbox for visualization and debug
        # if not cfgs["HEADLESS"] and True:
        #     gym.gym.clear_lines(gym.viewer)
        # for env_i in range(gym.num_envs):
        #     for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
        #         all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
        #         rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
        #         rotation_matrix = rotation.as_matrix()
        #         rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
        #         all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
        #         if not cfgs["HEADLESS"] and True:
        #             idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
        #             for part_i in range(len(gapart_raw_valid_anno)):
        #                 bbox_now_i = all_bbox_now[part_i]
        #                 for i in range(len(idx_set)):
        #                     gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
        #                         np.concatenate((bbox_now_i[idx_set[i][0]], 
        #                                         bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
        #                         np.array([1, 0 ,0], dtype=np.float32))
        
        print(gym.save_root)
        gym.run_steps(pre_steps = 3, refresh_obs=False, print_step=False)
        
        ## render
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        
        for img_i,rgb in enumerate(rgb_envs[0]):
            img_i_str = str(img_i).zfill(3)
            os.makedirs(save_dir+f"/{rm_name}", exist_ok=True)
            imageio.imwrite(f"{save_dir}/{rm_name}/{save_name}-{rm_name}-img-{img_i_str}.png", rgb)
        # import pdb; pdb.set_trace()
        os.makedirs(f"{save_dir}", exist_ok=True)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_envs[0][:, :3]-np.array(cfgs["asset"]["arti_obj_pose_ps"][0]))
        point_cloud.colors = o3d.utility.Vector3dVector(colors_envs[0][:, :3]/255.0)
        
        # import pdb; pdb.set_trace()
        
        
        
        # save_to ply
        fname = os.path.join(save_dir, rm_name, f"{save_name}-{rm_name}-articulated-point_cloud.ply")
        o3d.io.write_point_cloud(fname, point_cloud)
        
        # import pdb; pdb.set_trace()
        rm_link_name = path.split("/")[-1].split(".")[0].split("-")[-1]
        bbox_id = [i for i, link_name in enumerate(gym.gapart_link_names[0]) if link_name == rm_link_name][0]
        
        bbox_info = None
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        if bbox_id != part_i:
                            continue
                        bbox_now_i = rotated_bbox_now[part_i]
                        bbox_info = {
                            "bbox_id": bbox_id,
                            "bbox": bbox_now_i,
                        }
                        for i in range(len(idx_set)):
                            if not cfgs["HEADLESS"]:
                                gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                    np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                    bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                    np.array([1, 0 ,0], dtype=np.float32))
        # gym.run_steps(pre_steps = 3000, refresh_obs=False, print_step=False)
        np.save(f"{save_dir}/{rm_name}/{save_name}-{rm_name}-bbox_info.npy", bbox_info, allow_pickle=True)
        gym.clean_up()
        del gym

elif args.mode == "run_arti_replay":
    '''
    function: init gym and run render code, render the articulated object point cloud
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    demo_paths = ["/home/haoran/Projects/Part/GAPartNet/manipulation/output/2024-04-06-15-45-21/interaction_infos.npy"]
    # paths -= unused_paths
    for demo_path in tqdm.tqdm(demo_paths, total=len(demo_paths)):
        demo_info = np.load(demo_path, allow_pickle=True).item()
        cfgs = demo_info["cfg"]
        interaction_info = demo_info["interaction"]
        task_cfg = demo_info["task_cfg"]
        cfgs["HEADLESS"] = args.headless
        # import pdb; pdb.set_trace()
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)
        # import pdb; pdb.set_trace()
        for interaction in tqdm.tqdm(interaction_info, total=len(interaction_info)):
            pos_action = torch.zeros_like(gym.dof_pos).squeeze(-1)
            pos_action[:, :9] = torch.tensor(interaction["arm_state"][:,0], device=gym.device, dtype=torch.float32)
            gym.gym.set_dof_position_target_tensor(gym.sim, gymtorch.unwrap_tensor(pos_action))
            gym.run_steps(pre_steps = 1)
        gym.clean_up()
        del gym

else:
    raise NotImplementedError   
