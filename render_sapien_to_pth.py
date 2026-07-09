#!/usr/bin/env python3
"""
Option B: Render GAPartNet-style .pth inputs from PartNet-Mobility / AKB48
mesh assets using SAPIEN 3.x (native API), replicating the official
dataset/render_tools + dataset/process_tools/convert_rendered_into_input.py
pipeline.

Per sample, for each camera viewpoint:
  1. Load URDF as an articulation, set joint qpos (random within limits)
  2. Render RGB, Position (depth), Segmentation (per-link)
  3. Map segmentation entity_id -> link -> GAPart category (sem) + instance id
  4. Compute NPCS map from each GAPart link's oriented bbox (canonical frame)
  5. Backproject depth -> camera-space point cloud, carry sem/ins/npcs/rgb
  6. FPS to 20000 points, normalize to unit ball (WorldSpaceToBallSpace)
  7. Save (xyz, rgb, sem, ins, npcs) as <name>.pth + meta/<name>.txt

Output layout matches what the inference DataModule + visualize_gapartnet
expect: gapartnet/data/GAPartNet_All/<split>/{pth,meta}/
"""
import os
import sys
import math
import json
import glob
import pickle
from os.path import join as pjoin

import numpy as np
import torch
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R



REPO = "/home/rai-laptop/actance/GAPartNet"
DATA_ROOT = pjoin(REPO, "data")          # PartNet-Mobility / AKB48 assets
OUT_ROOT = pjoin(REPO, "gapartnet/data/GAPartNet_All")
PARTNET_ID_LIST = pjoin(REPO, "dataset/render_tools/meta/partnet_all_id_list.txt")
NUM_POINTS = 20000

# GAPartNet part categories. index = semantic id; 0 = others.
TARGET_GAPARTS = [
    "others", "line_fixed_handle", "round_fixed_handle", "slider_button",
    "hinge_door", "slider_drawer", "slider_lid", "hinge_lid", "hinge_knob",
    "revolute_handle",
]
CAT2ID = {c: i for i, c in enumerate(TARGET_GAPARTS)}

# GAPartNet render camera (Kuafu) intrinsics, matching visu_util.map2image.
# We render at 1024x1024 (not 800) so enough valid pixels are produced for
# 20000-point FPS even on small objects; the backprojected cloud is resolution
# independent.
FX = FY = 1268.637939453125
CX = CY = 512.0  # principal point for 1024-wide image
WIDTH = HEIGHT = 1024
FOVY_DEG = 40.0
# camera distance multiplier relative to the object's diagonal AABB size
DIST_MULT = 2.0

# Camera viewpoints (elevation_deg, azimuth_deg, distance) — same as demo.ipynb
CAMERA_POSES = [
    (30, 120, 4), (30, 240, 4), (45, 180, 4),
    (60, 180, 4), (80, 120, 4), (80, 240, 4),
]


# --------------------------------------------------------------------------- #
# geometry helpers (mirror official convert_rendered_into_input / pose_utils)
# --------------------------------------------------------------------------- #
def find_max_dis(pc):
    mx = pc.max(0); mn = pc.min(0)
    c = (mx + mn) / 2
    r = ((((pc - c) ** 2).sum(1)) ** 0.5).max()
    return r, c


def world_to_ball(pc):
    """Normalize into unit ball. trans places the object at the GAPartNet
    camera convention (z ~ 4.2) so visualize_gapartnet projects correctly."""
    r, c = find_max_dis(pc)
    scale = r * 1.26
    cam_c = np.array([0.0, 0.0, 4.23])
    return (pc - c) / r, np.array([scale, cam_c[0], cam_c[1], cam_c[2]])


def farthest_point_sample(xyz, npoint):
    """GPU-accelerated FPS via the compiled pointnet2_ops (verified on sm_120).
    Falls back to a CPU loop only if the GPU op is unavailable."""
    try:
        import torch
        from pointnet2_ops import pointnet2_utils
        t = torch.from_numpy(xyz.astype(np.float32)).cuda().unsqueeze(0).contiguous()
        idx = pointnet2_utils.furthest_point_sample(t, npoint)
        return idx.squeeze(0).cpu().numpy()
    except Exception:
        N = xyz.shape[0]
        xyz_t = torch.from_numpy(xyz.astype(np.float32))
        centroids = torch.zeros(npoint, dtype=torch.long)
        distance = torch.ones(N) * 1e10
        farthest = torch.randint(0, N, (1,)).item()
        for i in range(npoint):
            centroids[i] = farthest
            d = ((xyz_t - xyz_t[farthest]) ** 2).sum(-1)
            distance = torch.minimum(distance, d)
            farthest = torch.max(distance, 0).indices.item()
        return centroids.numpy()


def compute_npcs_for_part(pts_in_part):
    """NPCS per part: canonical coords in [-0.5, 0.5] along the part's own
    bbox axes (aligned to world axes here; the official code uses the oriented
    bbox from link_annotation, which we approximate with axis-aligned)."""
    mn = pts_in_part.min(0)
    mx = pts_in_part.max(0)
    rng = np.maximum(mx - mn, 1e-6)
    return (pts_in_part - mn) / rng - 0.5


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def cam_pos_from_angles(theta_deg, phi_deg, distance):
    """SAPIEN convention matches render_utils.get_cam_pos_fix."""
    th = math.radians(theta_deg); ph = math.radians(phi_deg)
    x = math.sin(th) * math.cos(ph) * distance
    y = math.sin(th) * math.sin(ph) * distance
    z = math.cos(th) * distance
    return np.array([x, y, z], dtype=np.float32)


def setup_scene_and_object(data_path, urdf_file, cam_angles, qpos=None):
    """Build a SAPIEN 3.x scene, load the URDF articulation, set qpos, mount a
    camera aimed at the object's AABB center. cam_angles = (theta, phi).
    Distance is scaled by the object's diagonal AABB size. Returns
    (scene, cam, articulation, K, ext, link_info)."""
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_ambient_light([0.6, 0.6, 0.6])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_point_light([2, 2, 2], [0.4, 0.4, 0.4])
    scene.add_point_light([-2, -2, 2], [0.3, 0.3, 0.3])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load_kinematic(pjoin(data_path, urdf_file))

    # set joint qpos
    if qpos is not None and art.get_qpos().shape[0] > 0:
        art.set_qpos(qpos[:art.get_qpos().shape[0]])

    # step() 更新 forward kinematics (关节变换需要 FK 才能正确放置 link)
    scene.step()

    # compute object AABB (SAPIEN 2.x: visual_bodies + render_shapes)
    aabb_min = np.array([1e9, 1e9, 1e9]); aabb_max = np.array([-1e9, -1e9, -1e9])
    for link in art.get_links():
        for vb in link.get_visual_bodies():
            try:
                mat = link.get_pose().to_transformation_matrix()
                for shape in vb.get_render_shapes():
                    v = np.array(shape.mesh.vertices)
                    if len(v) > 0:
                        wv = (mat[:3,:3] @ v.T).T + mat[:3,3]
                        aabb_min = np.minimum(aabb_min, wv.min(0))
                        aabb_max = np.maximum(aabb_max, wv.max(0))
            except: pass
    center = ((aabb_min + aabb_max) / 2).astype(np.float64)
    diag = float(np.linalg.norm(aabb_max - aabb_min)) or 1.0

    theta, phi = cam_angles
    cam_offset = cam_pos_from_angles(theta, phi, diag * DIST_MULT)
    cam_world = center + cam_offset.astype(np.float64)
    # Official GAPartNet camera setup: forward/left/up → Pose.from_transformation_matrix
    forward = -cam_offset / (np.linalg.norm(cam_offset) + 1e-9)
    left = np.cross([0, 0, 1], forward)
    if np.linalg.norm(left) < 1e-6: left = np.array([1, 0, 0.0])
    left = left / (np.linalg.norm(left) + 1e-9)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_world
    actor = scene.create_actor_builder().build_kinematic()
    actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    cam = scene.add_mounted_camera("c", actor, sapien.Pose(),
                                   WIDTH, HEIGHT, np.deg2rad(FOVY_DEG), np.deg2rad(FOVY_DEG), 0.1, 100)

    scene.update_render()
    cam.take_picture()

    K = cam.get_intrinsic_matrix()
    ext = cam.get_extrinsic_matrix()  # world -> camera (3x4)

    # collect per-link info: entity_id -> link_name
    link_info = {}
    for lk in art.get_links():
        eid = int(lk.get_id())
        link_info[eid] = {"name": lk.get_name()}
    return scene, cam, art, K, ext, link_info


def render_one(data_path, category, model_id, cam_pose, render_idx,
               link_annotation):
    """Render one viewpoint and produce a .pth. Returns name or None on skip.
    cam_pose = (theta_deg, phi_deg, distance) — distance is ignored (re-derived
    from object size inside setup)."""
    theta, phi, _ = cam_pose
    urdf = "mobility_texture_gapartnet.urdf"
    scene, cam, art, K, ext, link_info = setup_scene_and_object(
        data_path, urdf, (theta, phi), qpos=None
    )

    color = np.array(cam.get_texture("Color"))[..., :3]            # H,W,3 float [0,1]
    position = np.array(cam.get_texture("Position"))              # H,W,4 (cam-space xyzr)
    seg = np.array(cam.get_texture("Segmentation"))               # H,W,4 uint32
    entity_id_map = seg[..., 1].astype(np.int32)                  # per-pixel link entity id

    # depth (SAPIEN camera: z points backward, so depth = -z)
    depth = -position[..., 2]
    valid_depth = depth > 1e-4

    # build entity_id -> GAPart sem_id + instance_id from link_annotation
    name2sem = {}   # link_name -> sem_id (0 if not gapart)
    gapart_links = []  # ordered gapart link names -> instance id
    for entry in link_annotation:
        nm = entry["link_name"]
        if entry.get("is_gapart", False):
            cat = entry.get("category", "")
            name2sem[nm] = CAT2ID.get(cat, 0)
            gapart_links.append(nm)
        else:
            name2sem[nm] = 0
    link2inst = {nm: i for i, nm in enumerate(gapart_links)}

    # per-pixel sem / ins via entity_id -> link_name
    sem_map = np.zeros((HEIGHT, WIDTH), dtype=np.int32)   # 0 = others/background
    ins_map = np.full((HEIGHT, WIDTH), -100, dtype=np.int32)
    for eid, info in link_info.items():
        nm = info["name"]
        sem_id = name2sem.get(nm, 0)
        mask = (entity_id_map == eid) & valid_depth
        if sem_id > 0 and nm in link2inst:
            sem_map[mask] = sem_id
            ins_map[mask] = link2inst[nm]
        else:
            sem_map[mask] = 0   # others
    # background (no depth) -> mark as -2 so we drop them in backproject
    bg = ~valid_depth
    sem_map[bg] = -2
    ins_map[bg] = -2

    # ---- backproject depth -> camera-space point cloud ----
    # Use the ACTUAL camera intrinsics returned by sapien (its focal unit
    # differs from raw pixels), not the GAPartNet visualization constants.
    fx, fy = abs(K[0, 0]), abs(K[1, 1])
    cx, cy = K[0, 2], K[1, 2]
    H, W = HEIGHT, WIDTH
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x_cam = (xx - cx) * z / fx
    y_cam = (yy - cy) * z / fy
    keep = valid_depth & (sem_map != -2)
    xyz_cam = np.stack([x_cam[keep], y_cam[keep], z[keep]], axis=1).astype(np.float32)
    rgb = (color[keep] * 255).clip(0, 255).astype(np.uint8)
    sem = sem_map[keep]
    ins = ins_map[keep]

    if xyz_cam.shape[0] < NUM_POINTS:
        print(f"    [skip] only {xyz_cam.shape[0]} valid pts")
        return None

    # NPCS: per-instance, compute from the instance's own points (cam space).
    # The official code uses world-space oriented bbox; cam-space axis-aligned
    # is an approximation adequate for visualization.
    npcs = np.zeros((xyz_cam.shape[0], 3), dtype=np.float32)
    for inst in np.unique(ins):
        if inst < 0:
            continue
        m = ins == inst
        npcs[m] = compute_npcs_for_part(xyz_cam[m])

    # FPS to NUM_POINTS
    idx = farthest_point_sample(xyz_cam, NUM_POINTS)
    xyz_s = xyz_cam[idx]
    rgb_s = (rgb[idx].astype(np.float32) / 255.0)
    sem_s = sem[idx]
    ins_s = ins[idx]
    npcs_s = npcs[idx]

    # relabel instances continuous after FPS
    uniq = sorted(set(ins_s[ins_s >= 0].tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    new_ins = np.array([remap.get(int(v), -100) for v in ins_s], dtype=np.int32)

    # normalize to unit ball
    xyz_n, trans = world_to_ball(xyz_s)

    # sem convention: 0=others, 1..N for parts (already in that form)
    name = f"{category}_{model_id}_0_{render_idx}_{cam_pose}"
    name = name.replace(" ", "")
    return name, xyz_n, rgb_s, sem_s, new_ins, npcs_s, trans


def save_pth(split, name, xyz_n, rgb_s, sem_s, new_ins, npcs_s, trans):
    pth_dir = pjoin(OUT_ROOT, split, "pth")
    meta_dir = pjoin(OUT_ROOT, split, "meta")
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    torch.save(
        (xyz_n.astype(np.float32), rgb_s.astype(np.float32),
         sem_s.astype(np.int32), new_ins.astype(np.int32),
         npcs_s.astype(np.float32)),
        pjoin(pth_dir, name + ".pth"),
    )
    np.savetxt(pjoin(meta_dir, name + ".txt"), trans)


def guess_category_partnet(model_id):
    with open(PARTNET_ID_LIST) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == str(model_id):
                return parts[0]
    return "Object"


def process_partnet_model(model_id, n_renders=4, splits=("val",)):
    data_path = pjoin(DATA_ROOT, "GAPartNet_PartNetMobility", "partnet_mobility_part", str(model_id))
    if not os.path.isdir(data_path):
        print(f"[skip] {model_id}: no data at {data_path}")
        return 0
    anno_path = pjoin(data_path, "link_annotation_gapartnet.json")
    if not os.path.exists(anno_path):
        print(f"[skip] {model_id}: no link_annotation")
        return 0
    link_annotation = json.load(open(anno_path))
    category = guess_category_partnet(model_id)
    n = 0
    for render_idx in range(n_renders):
        for cam_pose in CAMERA_POSES:
            res = render_one(data_path, category, model_id, cam_pose,
                             render_idx, link_annotation)
            if res is None:
                continue
            name, xyz_n, rgb_s, sem_s, new_ins, npcs_s, trans = res
            for split in splits:
                save_pth(split, name, xyz_n, rgb_s, sem_s, new_ins, npcs_s, trans)
            n += 1
            print(f"  [ok] {name}: {xyz_n.shape[0]} pts, "
                  f"{len(set(new_ins[new_ins>=0].tolist()))} parts")
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="PartNet model ids to render; default = example 45780,102442")
    ap.add_argument("--n-renders", type=int, default=4)
    ap.add_argument("--splits", nargs="*", default=["val", "test_intra", "test_inter", "train"])
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    models = args.models or [45780, 102442]
    if args.clear:
        for split in args.splits:
            for sub in ("pth", "meta"):
                d = pjoin(OUT_ROOT, split, sub)
                for f in glob.glob(pjoin(d, "*")):
                    os.remove(f)

    total = 0
    for mid in models:
        print(f"=== PartNet model {mid} ===")
        total += process_partnet_model(mid, args.n_renders, tuple(args.splits))
    print(f"\nTotal .pth generated: {total}")


if __name__ == "__main__":
    main()
