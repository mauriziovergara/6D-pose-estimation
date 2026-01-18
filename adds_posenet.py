import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pose_net_rgb_geometric import PoseNetRGBGeometric
from train_posenet_rgb import set_seed, load_pred_boxes, load_model_points, quat_to_mat_np, make_datasets


def adds_distance_vectorized(pred_q, pred_t, gt_q, gt_t, cls_id, model_points):
    cls = int(cls_id)
    if cls not in model_points:
        return None

    R_pred = quat_to_mat_np(pred_q)
    R_gt = quat_to_mat_np(gt_q)
    pts = model_points[cls]

    pred_pts = (R_pred @ pts.T).T + pred_t
    gt_pts = (R_gt @ pts.T).T + gt_t

    min_dists = np.array(
        [np.linalg.norm(gt_pts - p, axis=1).min() for p in pred_pts])
    return np.mean(min_dists)


@torch.no_grad()
def evaluate_adds(model, val_loader, device, model_points, model_diameters, symmetric_only=False):
    model.eval()
    stats = defaultdict(lambda: {"add_ok": 0, "add_total": 0, "adds_dist": []})
    all_add_dist, all_add_ok, all_add_total = [], 0, 0

    symmetric_objects = {10, 11}  # 10=eggbox, 11=glue

    for batch in tqdm(val_loader, desc="Evaluating ADD-S (Symmetric objects only)" if symmetric_only else "Evaluating ADD-S"):
        rgb, gt_t, gt_q, bbox_c, cam_K, cls_id, _ = batch
        rgb, gt_t, gt_q, bbox_c, cam_K = rgb.to(device), gt_t.to(
            device), gt_q.to(device), bbox_c.to(device), cam_K.to(device)

        pred_q, pred_t = model(rgb, bbox_center=bbox_c, camera_matrix=cam_K)

        pq, pt = pred_q.cpu().numpy(), pred_t.cpu().numpy()
        gq, gt = gt_q.cpu().numpy(), gt_t.cpu().numpy()
        cls = cls_id.numpy()

        for i in range(rgb.size(0)):
            cls_i = int(cls[i])

            if symmetric_only and cls_i not in symmetric_objects:
                continue

            add_d = adds_distance_vectorized(
                pq[i], pt[i], gq[i], gt[i], cls[i], model_points)
            if add_d is None:
                continue

            diam = model_diameters[cls_i]
            thresh = diam * 0.1

            stats[cls_i]["adds_dist"].append(add_d)
            stats[cls_i]["add_total"] += 1
            stats[cls_i]["add_ok"] += int(add_d < thresh)
            all_add_dist.append(add_d)
            all_add_total += 1
            all_add_ok += int(add_d < thresh)

    overall = {"acc": all_add_ok / max(1, all_add_total),
               "mean": np.mean(all_add_dist) if all_add_dist else 0.0}
    per_class = {cid: {"acc": d["add_ok"] / max(1, d["add_total"]), "mean": np.mean(d["adds_dist"]) if d["adds_dist"] else 0.0, "n": d["add_total"]}
                 for cid, d in stats.items()}

    return per_class, overall


def run_adds_eval(root_dir, models_dir, checkpoint, pred_boxes_path, batch_size=32, val_split=0.2,
                  num_workers=4, seed=42, device="cuda", output_csv=None, symmetric_only=False):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    pred_boxes = load_pred_boxes(pred_boxes_path)
    model_points = load_model_points(models_dir)

    model_diameters = {}
    for cls_id, pts in model_points.items():
        diam = np.linalg.norm(pts.max(0) - pts.min(0))
        model_diameters[cls_id] = diam

    _, val_ds = make_datasets(argparse.Namespace(
        root_dir=root_dir, val_split=val_split, seed=seed), pred_boxes)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    model = PoseNetRGBGeometric(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False).get("model_state",
                          torch.load(checkpoint, map_location=device, weights_only=False)))

    per_class, overall = evaluate_adds(
        model, val_loader, device, model_points, model_diameters, symmetric_only=symmetric_only)

    title = "ADD-S RESULTS - SYMMETRIC OBJECTS ONLY" if symmetric_only else "ADD-S RESULTS"
    print(f"\n{'='*60}\n{title} (Threshold: 0.1 × object diameter)\n{'='*60}")
    if symmetric_only:
        print("Objects: 10=eggbox, 11=glue")
        print("="*60)
    print(f"{'Class':<8} {'ACC%':<10} {'Mean(cm)':<12} {'#Samples':<10}")
    print("-"*60)
    for cid in sorted(per_class.keys()):
        m = per_class[cid]
        obj_name = {10: "(eggbox)", 11: "(glue)"}.get(cid, "")
        print(
            f"{cid:02d} {obj_name:<10} {m['acc']*100:>6.2f}%    {m['mean']*100:>8.2f}      {m['n']:>6}")
    print("-"*60)
    print(
        f"{'GLOBAL':<8} {overall['acc']*100:>6.2f}%    {overall['mean']*100:>8.2f}")
    print("="*60 + "\n")

    if output_csv:
        data = [{"Classe": f"{cid:02d}", "ADD-S_ACC_%": f"{m['acc']*100:.2f}", "ADD-S_Mean_cm": f"{m['mean']*100:.2f}",
                 "n_samples": m["n"]} for cid, m in sorted(per_class.items())]
        data.append({"Classe": "GLOBAL", "ADD-S_ACC_%": f"{overall['acc']*100:.2f}",
                    "ADD-S_Mean_cm": f"{overall['mean']*100:.2f}", "n_samples": ""})
        pd.DataFrame(data).to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}\n")

    return per_class, overall


def main():
    parser = argparse.ArgumentParser(
        description="Valutazione ADD-S con PoseNet e inferenze YOLO"
    )
    parser.add_argument(
        "--root_dir",
        default=r"..\..\Linemod_preprocessed\Linemod_preprocessed\data",
        help="Path a Linemod_preprocessed/data"
    )
    parser.add_argument(
        "--models_dir",
        default=r"..\..\Linemod_preprocessed\Linemod_preprocessed\models",
        help="Path a Linemod_preprocessed/models"
    )
    parser.add_argument(
        "--checkpoint",
        default=r"..\Pose_weight\best_pose.pt",
        help="Checkpoint PoseNet da valutare"
    )
    parser.add_argument(
        "--pred_boxes",
        default=r"..\Predictions\yolo_predicted_boxes.yaml",
        help="File YAML con bbox YOLO predette"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--val_split", type=float,
                        default=0.2, help="Frazione validation set")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Numero workers DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda o cpu")
    parser.add_argument(
        "--output_csv",
        default=r"..\Predictions\adds_results.csv",
        help="Path file CSV per salvare risultati"
    )
    parser.add_argument(
        "--symmetric_only",
        action="store_true",
        help="Calcola ADD-S solo per oggetti simmetrici (10=eggbox, 11=glue)"
    )

    args = parser.parse_args()

    run_adds_eval(
        root_dir=args.root_dir,
        models_dir=args.models_dir,
        checkpoint=args.checkpoint,
        pred_boxes_path=args.pred_boxes,
        batch_size=args.batch,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        output_csv=args.output_csv,
        symmetric_only=args.symmetric_only,
    )


if __name__ == "__main__":
    main()
