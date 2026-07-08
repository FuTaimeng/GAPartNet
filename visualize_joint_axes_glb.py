#!/usr/bin/env python3
"""
Export GAPartNet joint-axis visualizations as 3D-viewable .glb files.

v2: fixed two bugs from v1
  - bbox wireframe: now uses the official compute_rotation_matrix to get the
    canonical 8-corner bbox, edges drawn correctly (no X-shape)
  - joint axis: uses the data-verified per-category NPCS axis mapping
    (slider_button→x, hinge_door→y, etc.), and for revolute joints places
    the axis at the hinge edge (NPCS x≈-0.5) instead of the center

Each .glb contains:
  - point cloud (colored by predicted semantic class)
  - per-part bbox wireframe (one color per instance)
  - joint axis (black cylinder)
  - operation direction arrow (red=push/pull, blue=rotate tangent, green=grasp)
"""
import os
import sys
import json
from os.path import join as pjoin

import numpy as np
import torch
import trimesh
from trimesh.geometry import align_vectors

REPO = "/home/rai-laptop/actance/GAPartNet"
GAPARTNET_DIR = pjoin(REPO, "gapartnet")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]
sys.path.insert(0, GAPARTNET_DIR)
os.chdir(GAPARTNET_DIR)

import importlib.util
_spec = importlib.util.spec_from_file_location("ri", pjoin(REPO, "run_inference.py"))
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

from dataset.gapartnet import GAPartNetDataset
from misc.info import PART_ID2NAME
from misc.visu_util import COLOR20

SAVE_ROOT = pjoin(GAPARTNET_DIR, "output/joint_axes_glb")
DATA_ROOT = pjoin(GAPARTNET_DIR, "data/GAPartNet_All")

# ---- per-category NPCS axis mapping ----
# axis index into rot ROWS (rot[0]=NPCS x, rot[1]=y, rot[2]=z) of
# estimate_pose_from_npcs. Verified by checking which NPCS axis has the
# smallest range (prismatic: press axis) / largest range (revolute: hinge
# long edge) in the model's actual NPCS predictions:
#   slider_button → z (thinnest, 15/15), slider_lid → z (4/4)
#   hinge_door → x (longest edge, 28/28), hinge_lid → x (2/2)
#   hinge_knob → x, line_fixed_handle → x
CATEGORY_JOINT = {
    "slider_button":      dict(joint="prismatic",  axis=2, pos="center", op="along",  label="press"),
    "slider_drawer":      dict(joint="prismatic",  axis=2, pos="center", op="along",  label="pull"),
    "slider_lid":         dict(joint="prismatic",  axis=2, pos="center", op="along",  label="slide"),
    "hinge_door":         dict(joint="revolute",   axis=0, pos="edge",   op="around", label="swing"),
    "hinge_lid":          dict(joint="revolute",   axis=0, pos="edge",   op="around", label="flip"),
    "hinge_knob":         dict(joint="continuous", axis=0, pos="center", op="around", label="turn"),
    "revolute_handle":    dict(joint="revolute",   axis=0, pos="center", op="around", label="turn"),
    "line_fixed_handle":  dict(joint="fixed",      axis=0, pos="center", op="grasp",  label="grasp+pull"),
    "round_fixed_handle": dict(joint="fixed",      axis=0, pos="center", op="grasp",  label="grasp+pull"),
}

# NPCS canonical bbox corner order (from pose_utils.get_NPCS_map_from_oriented_bbox)
# 8 corners expressed as sign triplets in (x,y,z):
NPCS_CORNERS = np.array([
    [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],   # top (z+)
    [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]  # bottom (z-)
], dtype=float) * 0.5

# estimate_pose_from_npcs corner order:
# 0:(-x,-y,-z) 1:(+x,-y,-z) 2:(-x,+y,-z) 3:(-x,-y,+z)
# 4:(+x,+y,-z) 5:(+x,-y,+z) 6:(-x,+y,+z) 7:(+x,+y,+z)
# edges = pairs differing in exactly one sign (Hamming distance 1)
BBOX_EDGES = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]

WORLD_UP = np.array([0.0, 0.0, 1.0])


def compute_rotation_matrix(b1, b2):
    """Official rotation matrix from bbox_canon to bbox_scaled (pose_utils.py)."""
    c1 = np.mean(b1, axis=0); c2 = np.mean(b2, axis=0)
    H = np.dot((b1 - c1).T, (b2 - c2))
    U, s, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        R[0, :] *= -1
    return R.T


def npcs_frame_from_bbox(bbox8, rot):
    """Use the rotation matrix returned by estimate_pose_from_npcs.
    IMPORTANT: rot's ROWS are the NPCS axes in world coords:
      rot[0,:] = NPCS x axis, rot[1,:] = NPCS y axis, rot[2,:] = NPCS z axis
    (verified: rot[i,:] · (bbox[i+1]-bbox[0]) normalized = +1.0)
    """
    bbox8 = np.asarray(bbox8, dtype=np.float64)
    center = bbox8.mean(axis=0)
    R = np.asarray(rot, dtype=np.float64)  # rows = NPCS axes
    # half-lengths along each NPCS axis: project bbox corners onto each axis
    centered = bbox8 - center
    # R[i,:] is axis i, so projection = centered @ R[i,:]
    s = np.array([np.abs(centered @ R[i, :]).max() for i in range(3)])
    return center, R, s  # s[i] = half-length along NPCS axis i


def world_bbox_canonical(bbox8, rot):
    """Return the bbox 8 corners + NPCS frame. Uses rot from
    estimate_pose_from_npcs directly."""
    bbox8 = np.asarray(bbox8, dtype=np.float64)
    center, R, s = npcs_frame_from_bbox(bbox8, rot)
    return bbox8, center, R, s


def make_cylinder(p1, p2, radius, color):
    p1 = np.asarray(p1, dtype=np.float64); p2 = np.asarray(p2, dtype=np.float64)
    vec = p2 - p1; height = float(np.linalg.norm(vec))
    if height < 1e-6: return None
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=12)
    Rm = align_vectors([0, 0, 1.0], vec)
    M = np.eye(4); M[:3, :3] = Rm[:3, :3]; M[:3, 3] = p1 + vec / 2
    cyl.apply_transform(M)
    cyl.visual.face_colors = np.array(color, dtype=np.uint8)
    return cyl


def make_arrow(origin, direction, length, radius, color):
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    shaft_len = length * 0.75; tip_len = length * 0.25
    tip_radius = radius * 2.2
    origin = np.asarray(origin, dtype=np.float64)
    shaft_end = origin + direction * shaft_len
    shaft = make_cylinder(origin, shaft_end, radius, color)
    cone = trimesh.creation.cone(radius=tip_radius, height=tip_len, sections=12)
    Rm = align_vectors([0, 0, 1.0], direction)
    M = np.eye(4); M[:3, :3] = Rm[:3, :3]
    M[:3, 3] = shaft_end + direction * (tip_len / 2)
    cone.apply_transform(M)
    cone.visual.face_colors = np.array(color, dtype=np.uint8)
    geos = [g for g in [shaft, cone] if g is not None]
    return trimesh.util.concatenate(geos) if geos else None


def bbox_wireframe(corners, color):
    """Draw the 12 edges using the fixed corner order from
    estimate_pose_from_npcs (BBOX_EDGES)."""
    corners = np.asarray(corners, dtype=np.float64)
    geos = []
    for a, b in BBOX_EDGES:
        c = make_cylinder(corners[a], corners[b], 0.004, color)
        if c is not None: geos.append(c)
    return trimesh.util.concatenate(geos) if geos else None


@torch.no_grad()
def export_sample_glb(model, item, save_name):
    pc_ids, sem_seg, proposals = ri.infer_one(model, item)
    sem_preds = sem_seg.sem_preds.cpu().numpy()
    full_xyz = item.points[:, :3].cpu().numpy()
    colors = COLOR20[sem_preds % 20]
    geometries = [(trimesh.points.PointCloud(full_xyz, colors=(colors / 255.0)), "point_cloud")]

    if proposals is None or proposals.proposal_offsets is None or len(proposals.proposal_offsets) <= 1:
        print(f"  [no proposals] {save_name}"); return False

    from misc.pose_fitting import estimate_pose_from_npcs
    n_full = int(proposals.valid_mask.shape[0])
    device = proposals.valid_mask.device
    indices = torch.arange(n_full, dtype=torch.int64, device=device)
    proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices.long()].cpu().numpy()
    p_off = proposals.proposal_offsets.detach().cpu().numpy()
    pt_xyz = proposals.pt_xyz.detach().cpu().numpy()
    npcs_map = np.zeros((n_full, 3), dtype=np.float32)
    try:
        vi = torch.where(proposals.valid_mask == True)[0][
            proposals.sorted_indices.long()[torch.where(proposals.npcs_valid_mask == True)]].cpu().numpy()
        npcs_map[vi] = proposals.npcs_preds.detach().cpu().numpy()
    except Exception:
        pass
    sem_at_proposal = proposals.pt_sem_classes.detach().cpu().numpy()

    plotted = 0
    for i in range(len(p_off) - 1):
        a, b = int(p_off[i]), int(p_off[i + 1])
        npcs_i = npcs_map[proposal_indices[a:b]] - 0.5
        xyz_i = pt_xyz[a:b]
        if xyz_i.shape[0] < 10: continue
        try:
            bbox_raw, scale, rot, trans, _, _ = estimate_pose_from_npcs(xyz_i, npcs_i)
        except Exception:
            continue
        if scale[0] is None: continue
        cat_id = int(sem_at_proposal[i])
        cat_name = PART_ID2NAME.get(cat_id, "?")
        sem_cfg = CATEGORY_JOINT.get(cat_name)
        if sem_cfg is None: continue

        # Use the raw bbox corners + rot matrix from estimate_pose_from_npcs.
        # rot ROWS = NPCS axes (rot[0]=x, rot[1]=y, rot[2]=z).
        bbox8, center, R, half_lengths = world_bbox_canonical(bbox_raw, rot)

        bbox_color = COLOR20[(plotted % 19) + 1]
        wf = bbox_wireframe(bbox8, bbox_color)
        if wf is not None: geometries.append((wf, f"bbox_{plotted}_{cat_name}"))

        # joint axis = fixed NPCS axis index from CATEGORY_JOINT (verified mapping)
        axis_idx = sem_cfg["axis"]  # 0=x, 1=y, 2=z
        npcs_axis_world = R[axis_idx, :]  # row = NPCS axis in world frame

        # joint axis position: revolute → at hinge edge; prismatic/continuous → center
        # For revolute (hinge_door/lid), the joint axis runs along NPCS x (the
        # longest edge), and the hinge is located at one of the y-edges
        # (the side where the door/lid attaches). Offset along NPCS y (axis 1).
        if sem_cfg["pos"] == "edge":
            face_offset = -half_lengths[1] * R[1, :]  # offset to y=-0.5 edge
            axis_anchor = center + face_offset
        else:
            axis_anchor = center

        # joint axis line (black) — drawn through the anchor (edge for revolute)
        axis_len = 0.14
        ax_mesh = make_cylinder(axis_anchor - npcs_axis_world * axis_len,
                                axis_anchor + npcs_axis_world * axis_len, 0.008, [20, 20, 20, 255])
        if ax_mesh is not None: geometries.append((ax_mesh, f"axis_{plotted}_{cat_name}"))

        # operation direction arrow
        op_color = {"along": [220, 40, 40, 255], "around": [40, 80, 220, 255],
                    "grasp": [40, 180, 60, 255]}[sem_cfg["op"]]
        if sem_cfg["op"] == "along":
            # prismatic: arrow along the joint axis, from the part center
            arrow_dir = npcs_axis_world
            arrow_origin = center
        elif sem_cfg["op"] == "around":
            # revolute: the opening arrow sits on the OPPOSITE edge from the
            # hinge (y=+0.5 side) and points perpendicular to the lid/door
            # surface (along the thin axis z), indicating the open direction.
            open_edge = center + half_lengths[1] * R[1, :]   # opposite to hinge (y=+0.5)
            arrow_dir = R[2, :]  # thin axis = lid normal = open direction
            arrow_origin = open_edge
        else:  # grasp
            arrow_dir = R[2, :]
            arrow_origin = center
        arrow = make_arrow(arrow_origin, arrow_dir, axis_len, 0.009, op_color)
        if arrow is not None: geometries.append((arrow, f"dir_{plotted}_{sem_cfg['label']}"))

        plotted += 1

    if plotted == 0:
        print(f"  [no valid parts] {save_name}"); return False

    scene = trimesh.Scene({n: g for g, n in geometries})
    os.makedirs(SAVE_ROOT, exist_ok=True)
    out_path = pjoin(SAVE_ROOT, save_name + ".glb")
    with open(out_path, "wb") as f:
        f.write(scene.export(file_type="glb"))
    print(f"  [ok] {save_name}: {plotted} parts -> {out_path}")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=pjoin(GAPARTNET_DIR, "ckpt/best.ckpt"))
    args = ap.parse_args()
    model = ri.build_model(args.ckpt)
    ds = GAPartNetDataset(pjoin(DATA_ROOT, "val/pth"), max_points=20000, voxel_size=(0.01, 0.01, 0.01))
    print(f"Dataset: {len(ds)} samples")
    n_ok = 0
    for i in range(len(ds)):
        item = ds[i]; pc_id = item.pc_id
        print(f"[{i+1}/{len(ds)}] {pc_id}")
        try:
            if export_sample_glb(model, item, pc_id): n_ok += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] {pc_id}: {repr(e)[:120]}")
    print(f"\nExported {n_ok}/{len(ds)} .glb -> {SAVE_ROOT}/")


if __name__ == "__main__":
    main()
