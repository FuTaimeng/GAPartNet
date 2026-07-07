#!/usr/bin/env python3
"""
Standalone GAPartNet inference + 3D visualization, bypassing the
lightning.pytorch CLI (whose jsonargparse integration is fragile on modern
PyTorch/Python). Reproduces the data path of `train.py test` and calls the
same `visualize_gapartnet` used by the official pipeline.

For each example .pth sample this:
  - runs the GAPartNet backbone -> semantic seg -> instance proposals -> NPcs
  - estimates 3D bounding boxes from NPcs via RANSAC pose fitting
  - renders the readme-style 2D-projected 3D visualization grid
  - also exports an interactive .ply point cloud colored by predictions
"""
import os
import sys
import glob
from os.path import join as pjoin

import numpy as np
import torch

REPO = "/home/rai-laptop/actance/GAPartNet"
GAPARTNET_DIR = pjoin(REPO, "gapartnet")
# The model imports top-level packages (network, dataset, misc, structure) that
# live under gapartnet/. The repo root also has a *different* `structure/`
# package (the demo code), so we must remove the script's own dir (repo root)
# from sys.path and only keep gapartnet/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]
sys.path.insert(0, GAPARTNET_DIR)
# Run from the gapartnet dir so relative file paths in the yaml/config resolve.
os.chdir(GAPARTNET_DIR)

from network.model import GAPartNet
from network.grouping_utils import filter_invalid_proposals, apply_nms
from structure.point_cloud import PointCloud
from structure.instances import Instances
from dataset.gapartnet import GAPartNetDataset
from misc.visu import visualize_gapartnet
from misc.pose_fitting import estimate_pose_from_npcs


DATA_ROOT = pjoin(GAPARTNET_DIR, "data/GAPartNet_All")
SPLIT = "val"
SAVE_ROOT = pjoin(GAPARTNET_DIR, "output/GAPartNet_result")
RAW_IMG_ROOT = pjoin(GAPARTNET_DIR, "data/image_kuafu")  # may not exist; OK


def build_model(ckpt_path):
    model = GAPartNet(
        in_channels=6,
        num_part_classes=10,
        backbone_type="SparseUNet",
        backbone_cfg={"channels": [16, 32, 48, 64, 80, 96, 112], "block_repeat": 2},
        instance_seg_cfg={
            "ball_query_radius": 0.04,
            "max_num_points_per_query": 50,
            "min_num_points_per_proposal": 5,
            "max_num_points_per_query_shift": 300,
            "score_fullscale": 28,
            "score_scale": 50,
        },
        symmetry_indices=[0, 1, 3, 3, 2, 0, 3, 2, 4, 1],
        training_schedule=[0, 0],
        visualize_cfg={
            "visualize": True,
            "visualize_dir": "visu",
            "sample_num": 100,
            "RAW_IMG_ROOT": RAW_IMG_ROOT,
            "GAPARTNET_DATA_ROOT": DATA_ROOT,
            "SAVE_ROOT": SAVE_ROOT,
            "save_option": ["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt",
                            "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure",
                            "bbox_pred", "bbox_pred_pure"],
        },
        ckpt=ckpt_path,
    )
    model = model.cuda()
    # The model's step gates on `self.current_epoch >= start_*`. current_epoch
    # is a read-only LightningModule property backed by self.trainer. With no
    # trainer attached, replace the property on the class with a constant.
    model.start_clustering = 0
    model.start_scorenet = 0
    model.start_npcs = 0
    type(model).current_epoch = property(lambda self: 999)
    model.eval()
    model.symmetry_indices = model.symmetry_indices.cuda()
    return model


@torch.no_grad()
def infer_one(model, pc_item):
    """Run the model's training/validation step on a single sample."""
    pc_item_gpu = pc_item.to("cuda")
    point_clouds = [pc_item_gpu]
    pc_ids, sem_seg, proposals, _ = model._training_or_validation_step(
        point_clouds, batch_idx=0, running_mode=SPLIT
    )
    if proposals is not None:
        proposals = filter_invalid_proposals(
            proposals,
            score_threshold=model.val_score_threshold,
            min_num_points_per_proposal=model.val_min_num_points_per_proposal,
        )
        proposals = apply_nms(proposals, model.val_nms_iou_threshold)
        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
    return pc_ids, sem_seg, proposals


@torch.no_grad()
def visualize_sample(model, pc_item, pc_id):
    pc_ids, sem_seg, proposals = infer_one(model, pc_item)
    sem_preds = sem_seg.sem_preds.cpu().numpy()

    if proposals is None or proposals.proposal_offsets is None or len(proposals.proposal_offsets) <= 1:
        print(f"  [no-proposals] model produced no instance proposals; "
              f"sem classes pred: {sorted(set(int(x) for x in sem_preds.tolist()))}")
        # still render the semantic-only visualization (no instance/bbox)
        visualize_gapartnet(
            SAVE_ROOT=model.visualize_cfg["SAVE_ROOT"],
            RAW_IMG_ROOT=model.visualize_cfg["RAW_IMG_ROOT"],
            GAPARTNET_DATA_ROOT=model.visualize_cfg["GAPARTNET_DATA_ROOT"],
            save_option=["pc", "sem_pred", "sem_gt", "ins_gt", "npcs_gt",
                         "bbox_gt", "bbox_gt_pure"],
            name=pc_id, split=SPLIT,
            sem_preds=sem_preds,
            ins_preds=np.zeros_like(sem_preds, dtype=np.int32),
            npcs_preds=np.zeros((sem_preds.shape[0], 3), dtype=np.float32),
            bboxes=[],
        )
        from misc.visu_util import COLOR20, save_point_cloud_to_ply
        xyz = pc_item.points[:, :3].cpu().numpy()
        ply_dir = pjoin(SAVE_ROOT, SPLIT, "ply")
        os.makedirs(ply_dir, exist_ok=True)
        save_point_cloud_to_ply(
            xyz, COLOR20[sem_preds % 20],
            save_name=pc_id + "_sem_pred.ply",
            save_root=ply_dir,
        )
        return {
            "pc_id": pc_id, "num_proposals": 0,
            "sem_classes_pred": sorted(set(int(x) for x in sem_preds.tolist())),
            "all_accu": float(sem_seg.all_accu),
            "pixel_accu": 0.0, "num_bboxes": 0,
        }

    # Build per-point instance / npcs maps and 3D bboxes from proposals.
    # Mirror the official on_test_epoch_end visualization logic exactly:
    #   - ins_seg_preds / npcs_maps live in the FULL point space (valid_mask.shape[0])
    #   - proposal_indices maps proposal-internal order -> full point index
    #   - pt_xyz is indexed directly by proposal_offsets (sequential proposal pts)
    pt_xyz = proposals.pt_xyz
    proposal_offsets = proposals.proposal_offsets.detach().cpu().numpy()
    batch_indices = proposals.batch_indices.detach().cpu().numpy()
    npcs_preds = proposals.npcs_preds
    npcs_valid_mask = proposals.npcs_valid_mask
    sorted_indices = proposals.sorted_indices
    valid_mask = proposals.valid_mask

    n_full = int(valid_mask.shape[0])
    device = valid_mask.device
    indices = torch.arange(n_full, dtype=torch.int64, device=device)
    proposal_indices = indices[valid_mask][sorted_indices.long()].detach().cpu().numpy()

    ins_seg_preds = np.zeros(n_full, dtype=np.int32)
    for i in range(len(proposal_offsets) - 1):
        ins_seg_preds[proposal_indices[proposal_offsets[i]:proposal_offsets[i + 1]]] = i + 1

    npcs_maps = np.zeros((n_full, 3), dtype=np.float32)
    if npcs_preds is not None and npcs_valid_mask is not None:
        try:
            valid_index = torch.where(valid_mask == True)[0][
                sorted_indices.long()[torch.where(npcs_valid_mask == True)]
            ].detach().cpu().numpy()
            npcs_maps[valid_index] = npcs_preds.detach().cpu().numpy()
        except Exception as e:
            print(f"  [npcs-map warn] {repr(e)[:100]}")

    # estimate 3D bboxes from npcs for each proposal
    bboxes = []
    bboxes_batch_index = []
    for i in range(len(proposal_offsets) - 1):
        a, b = int(proposal_offsets[i]), int(proposal_offsets[i + 1])
        npcs_i = npcs_maps[proposal_indices[a:b]] - 0.5
        xyz_i = pt_xyz[a:b].detach().cpu().numpy()
        if xyz_i.shape[0] < 10:
            continue
        try:
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i, npcs_i)
        except Exception:
            continue
        if scale[0] is None:
            continue
        bboxes_batch_index.append(int(batch_indices[a]))
        bboxes.append(bbox_xyz.tolist())

    sample_sem_pred = sem_preds
    sample_ins_seg_pred = ins_seg_preds
    sample_npcs_map = npcs_maps
    sample_bboxes = [bboxes[i] for i in range(len(bboxes)) if bboxes_batch_index[i] == 0]

    # readme-style projected 3D grid visualization
    visualize_gapartnet(
        SAVE_ROOT=model.visualize_cfg["SAVE_ROOT"],
        RAW_IMG_ROOT=model.visualize_cfg["RAW_IMG_ROOT"],
        GAPARTNET_DATA_ROOT=model.visualize_cfg["GAPARTNET_DATA_ROOT"],
        save_option=model.visualize_cfg["save_option"],
        name=pc_id,
        split=SPLIT,
        sem_preds=sample_sem_pred,
        ins_preds=sample_ins_seg_pred,
        npcs_preds=sample_npcs_map,
        bboxes=sample_bboxes,
    )

    # also dump an interactive .ply colored by predicted semantic class.
    # Use the full (20000-pt) point cloud in ball space for the colored dump.
    from misc.visu_util import COLOR20, save_point_cloud_to_ply
    ply_dir = pjoin(SAVE_ROOT, SPLIT, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    full_xyz = pc_item.points[:, :3].cpu().numpy()  # ball-space normalized xyz
    save_point_cloud_to_ply(
        full_xyz, COLOR20[sample_sem_pred % 20],
        save_name=pc_id + "_sem_pred.ply", save_root=ply_dir,
    )
    save_point_cloud_to_ply(
        full_xyz, COLOR20[(sample_ins_seg_pred % 20).astype(np.int_)],
        save_name=pc_id + "_ins_pred.ply", save_root=ply_dir,
    )

    return {
        "pc_id": pc_id,
        "num_proposals": len(proposal_offsets) - 1,
        "sem_classes_pred": sorted(set(int(x) for x in sample_sem_pred.tolist())),
        "all_accu": float(sem_seg.all_accu),
        "pixel_accu": float(sem_seg.pixel_accu) if sem_seg.pixel_accu else 0.0,
        "num_bboxes": len(sample_bboxes),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=pjoin(GAPARTNET_DIR, "ckpt/release.ckpt"))
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()
    ckpt = args.ckpt
    split = args.split
    print(f"Loading model from {ckpt} ...")
    model = build_model(ckpt)
    print("Model ready on", next(model.parameters()).device)

    pth_dir = pjoin(DATA_ROOT, SPLIT, "pth")
    files = sorted(glob.glob(pjoin(pth_dir, "*.pth")))
    print(f"Found {len(files)} samples in {pth_dir}")

    ds = GAPartNetDataset(
        pth_dir, max_points=20000, voxel_size=(0.01, 0.01, 0.01),
    )
    print(f"Dataset size: {len(ds)}")

    results = []
    for i in range(len(ds)):
        item = ds[i]
        pc_id = item.pc_id
        print(f"\n[{i+1}/{len(ds)}] {pc_id}  (GT parts: {int(item.num_instances)})")
        try:
            r = visualize_sample(model, item, pc_id)
            results.append(r)
            from misc.info import PART_ID2NAME
            cats = [PART_ID2NAME.get(c, str(c)) for c in r["sem_classes_pred"]]
            print(f"  -> {r['num_proposals']} proposals, "
                  f"pred classes {r['sem_classes_pred']} = {cats}, "
                  f"{r['num_bboxes']} 3D bboxes, "
                  f"sem all_accu={r['all_accu']:.3f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {pc_id}: {repr(e)[:200]}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['pc_id']}: {r['num_proposals']} props, "
              f"{r['num_bboxes']} bboxes, accu={r['all_accu']:.3f}")
    print(f"\nVisualizations saved under: {SAVE_ROOT}/{SPLIT}/")


if __name__ == "__main__":
    main()
