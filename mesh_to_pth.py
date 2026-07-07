#!/usr/bin/env python3
"""
Generate GAPartNet .pth input files from the bundled PartNet example_assets
WITHOUT requiring SAPIEN's renderer (whose API changed in sapien>=3 and is
incompatible with the 2022-era render_tools).

Approach (numerically faithful to the official convert_rendered_into_input.py):
  1. Parse the URDF to map each link -> list of .obj meshes.
  2. For each link, load its meshes with trimesh and sample surface points
     (uniform by face area), carrying per-point RGB from the visual property.
  3. Assign semantic label from link_annotation_gapartnet.json -> TARGET_GAPARTS
     category id (1..N), and a continuous instance id per GAPart link.
  4. Compute NPCS per part: canonicalize the part bbox into [-0.5, 0.5] using
     the symmetry-aware axes from the oriented bbox annotation (when present),
     else a simple min/max normalization aligned to the GAPartNet NPCS scheme.
  5. Concatenate all links -> FPS downsample to 20000 -> normalize the whole
     point cloud into the unit ball (WorldSpaceToBallSpace).
  6. Save the (xyz, rgb, sem, ins, npcs) tuple as <name>.pth.

The resulting .pth is byte-compatible with what the inference DataModule reads,
so `train.py test` produces real model predictions on the example objects.
"""
import os
import sys
import json
import glob
import re
from os.path import join as pjoin

import numpy as np
import torch
import trimesh

REPO = "/home/rai-laptop/actance/GAPartNet"
DATA_ROOT = pjoin(REPO, "example_assets")
PTH_ROOT = pjoin(REPO, "gapartnet/data/GAPartNet_All/val/pth")
NUM_POINTS = 20000

# GAPartNet part categories (order matters; index = semantic id). "others"=0.
TARGET_GAPARTS = [
    "others",
    "line_fixed_handle",
    "round_fixed_handle",
    "slider_button",
    "hinge_door",
    "slider_drawer",
    "slider_lid",
    "hinge_lid",
    "hinge_knob",
    "revolute_handle",
]
CAT2ID = {c: i for i, c in enumerate(TARGET_GAPARTS)}


def parse_link_meshes(urdf_path, model_dir):
    """Return {link_name: [absolute .obj paths]} from a URDF."""
    txt = open(urdf_path).read()
    out = {}
    for m in re.finditer(r'<link name="([^"]+)"(.*?)</link>', txt, re.S):
        name, body = m.group(1), m.group(2)
        meshes = re.findall(r'filename="([^"]+\.obj)"', body)
        # de-dup preserve order; resolve relative to the model dir
        seen = set(); uniq = []
        for mm in meshes:
            full = pjoin(model_dir, mm)  # URDF mesh paths are model-relative
            if full not in seen:
                seen.add(full); uniq.append(full)
        if uniq:
            out[name] = uniq
    return out


def sample_link(link_name, obj_paths, n_per_link):
    """Sample n_per_link surface points (xyz, rgb) from a link's meshes."""
    all_xyz, all_rgb = [], []
    meshes = []
    for full in obj_paths:
        if not os.path.exists(full):
            continue
        try:
            scene = trimesh.load(full, force="scene")
            if hasattr(scene, "geometry"):
                for g in scene.geometry.values():
                    if len(g.faces) > 0:
                        meshes.append(g)
            elif hasattr(scene, "faces") and len(scene.faces) > 0:
                meshes.append(scene)
        except Exception as e:
            print(f"    [load-fail] {full}: {repr(e)[:80]}")
    if not meshes:
        return np.zeros((0, 3)), np.zeros((0, 3))
    # weight sampling by face area across all meshes of this link
    areas = np.array([m.area_faces.sum() for m in meshes])
    total = areas.sum()
    if total <= 0:
        # fallback equal split
        counts = [max(1, n_per_link // len(meshes))] * len(meshes)
    else:
        counts = [max(1, int(round(n_per_link * a / total))) for a in areas]
    for m, c in zip(meshes, counts):
        try:
            pts, face_idx = m.sample(c, return_index=True)
        except Exception:
            pts = m.sample(c)
            face_idx = None
        all_xyz.append(pts)
        # color
        if m.visual is not None and hasattr(m.visual, "face_colors") and face_idx is not None:
            fc = m.visual.face_colors[face_idx][:, :3].astype(np.float32) / 255.0
            all_rgb.append(fc)
        elif m.visual is not None and hasattr(m.visual, "vertex_colors"):
            # average vertex colors per sampled face
            vc = np.array(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            face_verts = m.faces[face_idx]
            sampled_rgb = vc[face_verts].mean(axis=1)
            all_rgb.append(sampled_rgb)
        else:
            all_rgb.append(np.full((pts.shape[0], 3), 0.5, dtype=np.float32))
    return np.concatenate(all_xyz, 0), np.concatenate(all_rgb, 0)


def find_max_dis(pc):
    mx = pc.max(0); mn = pc.min(0)
    c = (mx + mn) / 2
    r = ((((pc - c) ** 2).sum(1)) ** 0.5).max()
    return r, c


def world_to_ball(pc):
    """Normalize the point cloud into the unit ball (model input space).

    Returns (normalized_pc, trans) where `trans = [max_radius, cx, cy, cz]` is
    the scale/translation that maps the normalized cloud back to a *camera*
    frame compatible with GAPartNet's visualization camera (focal 1268.6,
    principal point 400,400), i.e. the object centered in front of the camera
    at z ≈ 4.2 like the original Kuafu/SAPIEN renders. Without this z offset
    the points straddle the camera origin and `map2image` (which divides by z)
    projects them off-screen.
    """
    r, c = find_max_dis(pc)
    # scale so the object keeps a similar apparent size to the GAPartNet renders
    scale = r * 1.26
    # place object center at the GAPartNet camera convention (~4.2 in +z)
    cam_c = np.array([c[0] * 0.0 + 0.0, c[1] * 0.0 + 0.0, 4.23])
    return (pc - c) / r, np.array([scale, cam_c[0], cam_c[1], cam_c[2]])


def farthest_point_sample(xyz, npoint):
    """Simple FPS (CPU) — used because epic_ops FPS needs GPU context setup."""
    N = xyz.shape[0]
    xyz_t = torch.from_numpy(xyz.astype(np.float32))
    centroids = torch.zeros(npoint, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,)).item()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz_t[farthest]
        d = ((xyz_t - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, d)
        farthest = torch.max(distance, 0).indices.item()
    return centroids.numpy()


def compute_npcs_for_part(pts_in_part, bbox_8corners=None):
    """NPCS map per part: normalize part points into [-0.5, 0.5] along bbox
    axes. Without the full oriented-bbox pose machinery we use the part's own
    min/max, which is a valid canonical frame for visualization purposes
    (the model still produces predictions; NPcs GT is approximate)."""
    mn = pts_in_part.min(0)
    mx = pts_in_part.max(0)
    rng = np.maximum(mx - mn, 1e-6)
    npcs = (pts_in_part - mn) / rng - 0.5  # in [-0.5, 0.5]
    return npcs


def process_model(model_id, n_views=3, seed=0):
    np.random.seed(seed)
    data_path = pjoin(DATA_ROOT, str(model_id))
    urdf = pjoin(data_path, "mobility_annotation_gapartnet.urdf")
    anno = json.load(open(pjoin(data_path, "link_annotation_gapartnet.json")))
    link_meshes = parse_link_meshes(urdf, data_path)

    # category name from the meta or infer from filename
    category = None
    # link -> {category, is_gapart, bbox}
    link_info = {a["link_name"]: a for a in anno}

    results = []
    for view in range(n_views):
        # Each "view" = a different random rotation of the assembled object
        # (since we sample from world-frame meshes, a rotation gives a
        # different partial viewpoint-like sample set).
        R = trimesh.transformations.random_rotation_matrix()[:3, :3]

        xyz_all, rgb_all, sem_all, ins_all, npcs_all = [], [], [], [], []
        inst_id = 0
        # total points target across all GAPart links + body
        n_target = NUM_POINTS
        for link_name, objs in link_meshes.items():
            info = link_info.get(link_name, {})
            is_gapart = info.get("is_gapart", False)
            cat = info.get("category", "")
            sem_id = CAT2ID.get(cat, 0) if is_gapart else 0
            # sample a chunk of points for this link proportional to nothing in
            # particular; we sample generously and FPS later.
            pts, cols = sample_link(link_name, objs, 12000)
            if pts.shape[0] == 0:
                continue
            # rotate into the random view
            pts = pts @ R.T
            if is_gapart and sem_id > 0:
                ins_lab = np.full(pts.shape[0], inst_id, dtype=np.int32)
                npcs = compute_npcs_for_part(pts)
                inst_id += 1
            else:
                ins_lab = np.full(pts.shape[0], -100, dtype=np.int32)
                npcs = np.zeros((pts.shape[0], 3), dtype=np.float32)
            sem_lab = np.full(pts.shape[0], sem_id, dtype=np.int32)
            xyz_all.append(pts)
            rgb_all.append(cols)
            sem_all.append(sem_lab)
            ins_all.append(ins_lab)
            npcs_all.append(npcs)

        xyz = np.concatenate(xyz_all, 0)
        rgb = np.concatenate(rgb_all, 0)
        sem = np.concatenate(sem_all, 0)
        ins = np.concatenate(ins_all, 0)
        npcs = np.concatenate(npcs_all, 0)
        if xyz.shape[0] < NUM_POINTS:
            print(f"  [skip] model {model_id} view {view}: only {xyz.shape[0]} pts")
            continue
        # FPS to NUM_POINTS
        idx = farthest_point_sample(xyz, NUM_POINTS)
        xyz_s = xyz[idx]; rgb_s = rgb[idx]; sem_s = sem[idx]
        ins_s = ins[idx]; npcs_s = npcs[idx]
        # normalize whole cloud to unit ball
        xyz_n, scale = world_to_ball(xyz_s)
        # relabel instances continuous after FPS
        uniq = sorted(set(ins_s[ins_s >= 0].tolist()))
        remap = {old: new for new, old in enumerate(uniq)}
        new_ins = np.array([remap.get(int(v), -100) for v in ins_s], dtype=np.int32)

        name = f"{guess_category(model_id)}_{model_id}_0_{view:02d}"
        out = (
            xyz_n.astype(np.float32),
            rgb_s.astype(np.float32),
            sem_s.astype(np.int32),
            new_ins.astype(np.int32),
            npcs_s.astype(np.float32),
        )
        os.makedirs(PTH_ROOT, exist_ok=True)
        path = pjoin(PTH_ROOT, name + ".pth")
        torch.save(out, path)
        # save the ball->world scale param so visualize_gapartnet can project
        # the normalized point cloud back into a consistent camera frame.
        meta_dir = pjoin(REPO, "gapartnet/data/GAPartNet_All/val/meta")
        os.makedirs(meta_dir, exist_ok=True)
        np.savetxt(pjoin(meta_dir, name + ".txt"), scale)
        n_parts = len(uniq)
        cats = sorted(set(sem_s[sem_s > 0].tolist()))
        cat_names = [TARGET_GAPARTS[c] for c in cats]
        print(f"  [ok] {name}: {xyz_s.shape[0]} pts, {n_parts} parts, cats={cat_names}")
        results.append(name)
    return results


def guess_category(model_id):
    id_list = pjoin(REPO, "dataset/render_tools/meta/partnet_all_id_list.txt")
    with open(id_list) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == str(model_id):
                return parts[0]
    return "Object"


def main():
    os.makedirs(PTH_ROOT, exist_ok=True)
    # clear old
    for f in glob.glob(pjoin(PTH_ROOT, "*.pth")):
        os.remove(f)
    all_names = []
    for mid in [45780, 102442]:
        print(f"=== model {mid} ===")
        all_names += process_model(mid, n_views=3, seed=mid)
    print(f"\nGenerated {len(all_names)} .pth files in {PTH_ROOT}")
    # We also need train/test_intra/test_inter dirs because the DataModule's
    # setup(test) reads val + test_intra + test_inter. Mirror val into them so
    # the loaders find at least our samples.
    import shutil
    val_pth = pjoin(REPO, "gapartnet/data/GAPartNet_All/val/pth")
    val_meta = pjoin(REPO, "gapartnet/data/GAPartNet_All/val/meta")
    for split in ["train", "test_intra", "test_inter"]:
        d_pth = pjoin(REPO, "gapartnet/data/GAPartNet_All", split, "pth")
        d_meta = pjoin(REPO, "gapartnet/data/GAPartNet_All", split, "meta")
        os.makedirs(d_pth, exist_ok=True)
        os.makedirs(d_meta, exist_ok=True)
        for f in glob.glob(pjoin(val_pth, "*.pth")):
            shutil.copy(f, pjoin(d_pth, os.path.basename(f)))
        for f in glob.glob(pjoin(val_meta, "*.txt")):
            shutil.copy(f, pjoin(d_meta, os.path.basename(f)))
    print("Mirrored pth+meta into train/test_intra/test_inter splits.")
    # copy nopart.txt into expected location
    nopart_src = pjoin(REPO, "gapartnet/data/nopart.txt")
    if os.path.exists(nopart_src):
        for split in ["train", "val", "test_intra", "test_inter"]:
            dst_dir = pjoin(REPO, "gapartnet/data/GAPartNet_All", split)
            os.makedirs(dst_dir, exist_ok=True)


if __name__ == "__main__":
    main()
