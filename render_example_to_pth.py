#!/usr/bin/env python3
"""
Render the bundled example_assets into GAPartNet .pth input files and place
them in the data layout the inference DataModule expects:

    gapartnet/data/GAPartNet_All/<split>/pth/<name>.pth

Mirrors the demo.ipynb render_one_image() + the dataset/process_tools
convert_rendered_into_input.py conversion, but as a single self-contained
script so we can run the official `train.py test` inference afterwards.
"""
import os
import sys
import json
import pickle
from os.path import join as pjoin

import numpy as np
import torch

# repo root on sys.path so `dataset.render_tools...` imports resolve
REPO = "/home/rai-laptop/actance/GAPartNet"
sys.path.insert(0, REPO)

from dataset.render_tools.utils.config_utils import (
    PARTNET_CAMERA_POSITION_RANGE, TARGET_GAPARTS, BACKGROUND_RGB,
)
from dataset.render_tools.utils.read_utils import (
    get_id_category, read_joints_from_urdf_file,
)
from dataset.render_tools.utils.render_utils import (
    get_cam_pos_fix, set_all_scene, render_rgb_image, render_depth_map,
    render_sem_ins_seg_map, add_background_color_for_image, get_camera_pos_mat,
)
from dataset.render_tools.utils.pose_utils import (
    query_part_pose_from_joint_qpos, get_NPCS_map_from_oriented_bbox,
)
from dataset.process_tools.utils.sample_utils import FPS

PARTNET_ID_PATH = pjoin(REPO, "dataset/render_tools/meta/partnet_all_id_list.txt")
DATA_ROOT = pjoin(REPO, "example_assets")
SAVE_ROOT = pjoin(REPO, "example_rendered")
PTH_ROOT = pjoin(REPO, "gapartnet/data/GAPartNet_All/val/pth")

# Camera poses (elevation_deg, azimuth_deg, distance) — same as demo.ipynb
CAMERA_POSES = [
    (30, 120, 4),
    (30, 240, 4),
    (45, 180, 4),
    (60, 180, 4),
    (80, 120, 4),
    (80, 240, 4),
]
NUM_POINTS = 20000


def find_max_dis(pc):
    mx = pc.max(0); mn = pc.min(0)
    c = (mx + mn) / 2
    r = ((((pc - c) ** 2).sum(1)) ** 0.5).max()
    return r, c


def world_to_ball(pc):
    r, c = find_max_dis(pc)
    return (pc - c) / r, r, c


def get_point_cloud(rgb, depth, sem, ins, npcs, K):
    H, W = depth.shape
    pts, prgb, psem, pins, pnpcs = [], [], [], [], []
    for y in range(H):
        for x in range(W):
            if sem[y, x] == -2 or ins[y, x] == -2:
                continue
            z = float(depth[y, x])
            xx = (x - K[0, 2]) * z / K[0, 0]
            yy = (y - K[1, 2]) * z / K[1, 1]
            pts.append([xx, yy, z])
            prgb.append(rgb[y, x] / 255.0)
            psem.append(sem[y, x])
            pins.append(ins[y, x])
            pnpcs.append(npcs[y, x])
    return (np.array(pts), np.array(prgb), np.array(psem),
            np.array(pins), np.array(pnpcs))


def render_one(model_id, cam_pose, render_idx=0, width=800, height=800):
    category = get_id_category(model_id, PARTNET_ID_PATH)
    data_path = pjoin(DATA_ROOT, str(model_id))
    joints_dict = read_joints_from_urdf_file(data_path, "mobility_annotation_gapartnet.urdf")
    joint_qpos = {}
    for jn, jd in joints_dict.items():
        jt = jd["type"]
        if jt in ("prismatic", "revolute"):
            jl = jd["limit"]
            joint_qpos[jn] = np.random.uniform(jl[0], jl[1])
        elif jt == "fixed":
            joint_qpos[jn] = 0.0
        elif jt == "continuous":
            joint_qpos[jn] = np.random.uniform(-10000.0, 10000.0)

    cam_pos = get_cam_pos_fix(cam_pose[0], cam_pose[1], cam_pose[2])
    scene, camera, engine, robot = set_all_scene(
        data_path=data_path,
        urdf_file="mobility_annotation_gapartnet.urdf",
        cam_pos=cam_pos, width=width, height=height,
        use_raytracing=False, joint_qpos_dict=joint_qpos,
    )
    link_pose_dict = query_part_pose_from_joint_qpos(
        data_path=data_path, anno_file="link_annotation_gapartnet.json",
        joint_qpos=joint_qpos, joints_dict=joints_dict,
        target_parts=TARGET_GAPARTS, base_link_name="base", robot=robot,
    )
    rgb = render_rgb_image(camera=camera)
    depth = render_depth_map(camera=camera)
    sem_seg, ins_seg, valid_linkName_to_instId = render_sem_ins_seg_map(
        scene=scene, camera=camera, link_pose_dict=link_pose_dict, depth_map=depth,
    )
    valid_link_pose_dict = {k: link_pose_dict[k] for k in valid_linkName_to_instId}
    K, w2c_R, c2w_T = get_camera_pos_mat(camera)
    _, npcs_map = get_NPCS_map_from_oriented_bbox(
        depth, ins_seg, valid_linkName_to_instId, valid_link_pose_dict, K, w2c_R, c2w_T,
    )
    rgb = add_background_color_for_image(rgb, depth, BACKGROUND_RGB)

    name = f"{category}_{model_id}_0_{render_idx}_{str(cam_pose).replace(' ','')}"
    return name, rgb, depth, sem_seg, ins_seg, npcs_map, K


def to_pth(name, rgb, depth, sem_seg, ins_seg, npcs_map, K):
    pts, prgb, psem, pins, pnpcs = get_point_cloud(rgb, depth, sem_seg, ins_seg, npcs_map, K)
    if pts.shape[0] < NUM_POINTS:
        print(f"  [skip] {name}: only {pts.shape[0]} pts (<{NUM_POINTS})")
        return False
    # FPS to NUM_POINTS
    sampled, fps_idx = FPS(pts, NUM_POINTS)
    if sampled is None:
        print(f"  [skip] {name}: FPS returned None")
        return False
    pts_s = pts[fps_idx]
    prgb_s = prgb[fps_idx]
    psem_s = psem[fps_idx]
    pins_s = pins[fps_idx]
    pnpcs_s = pnpcs[fps_idx]

    # normalize to ball space
    pts_n, r, c = world_to_ball(pts_s)

    # label conversion: old sem -1 for others / [0,nClass-1] parts
    # new sem: 0 for others, [1,nClass] parts
    psem_c = psem_s + 1
    pins_c = pins_s.copy()
    pins_c[pins_c == -1] = -100
    # relabel continuous
    j = 0
    while j < pins_c.max():
        if (pins_c == j).sum() == 0:
            pins_c[pins_c == pins_c.max()] = j
        j += 1

    os.makedirs(PTH_ROOT, exist_ok=True)
    out = (
        pts_n.astype(np.float32),
        prgb_s.astype(np.float32),
        psem_c.astype(np.int32),
        pins_c.astype(np.int32),
        pnpcs_s.astype(np.float32),
    )
    torch.save(out, pjoin(PTH_ROOT, name + ".pth"))
    print(f"  [ok] {name}: {pts_s.shape[0]} pts -> {PTH_ROOT}/{name}.pth")
    # quick stats
    n_parts = len(np.unique(pins_c[pins_c >= 0]))
    print(f"       sem classes present: {sorted(set(psem_c.tolist()))}, instances: {n_parts}")
    return True


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    model_ids = [45780, 102442]
    np.random.seed(0)
    total = 0
    for mid in model_ids:
        for render_idx in range(2):
            for ci, cam_pose in enumerate(CAMERA_POSES[:3]):  # 3 camera views each
                try:
                    name, rgb, depth, sem, ins, npcs, K = render_one(mid, cam_pose, render_idx)
                except Exception as e:
                    print(f"  [render-fail] {mid} cam {cam_pose}: {repr(e)[:160]}")
                    continue
                if to_pth(name, rgb, depth, sem, ins, npcs, K):
                    total += 1
    print(f"\nTotal .pth files generated: {total}")


if __name__ == "__main__":
    main()
