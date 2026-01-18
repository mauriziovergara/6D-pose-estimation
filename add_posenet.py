import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from pose_net_rgb_geometric import PoseNetRGBGeometric
from train_posenet_rgb import (
    set_seed,
    load_pred_boxes,
    load_model_points,
    quat_to_mat_np,
    make_datasets,
)


def quaternion_angle_deg(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between unit quaternions in degrees."""
    pred_q = pred_q / (pred_q.norm(dim=1, keepdim=True) + 1e-8)
    gt_q = gt_q / (gt_q.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred_q * gt_q, dim=1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot).clamp(min=0.0, max=np.pi) * (180.0 / np.pi)


def add_distance(pred_q, pred_t, gt_q, gt_t, cls_id, model_points):
    cls = int(cls_id)
    if cls not in model_points:
        return None
    R_pred = quat_to_mat_np(pred_q)
    R_gt = quat_to_mat_np(gt_q)
    pts = model_points[cls]
    pred_pts = (R_pred @ pts.T).T + pred_t
    gt_pts = (R_gt @ pts.T).T + gt_t
    return np.linalg.norm(pred_pts - gt_pts, axis=1).mean()


@torch.no_grad()
def evaluate_add(
    model,
    val_loader,
    device,
    model_points,
    save_predictions_path=None,
):
    model_diameters = {}
    for cls_id, pts in model_points.items():
        diam = np.linalg.norm(pts.max(0) - pts.min(0))
        model_diameters[cls_id] = diam

    model.eval()
    stats = defaultdict(lambda: {
        "add_ok": 0,
        "add_total": 0,
        "add_dist": [],
        "trans_err": [],
        "rot_err": [],
    })
    all_trans_err = []
    all_rot_err = []
    all_add_dist = []
    all_add_ok = 0
    all_add_total = 0

    predictions = {}

    for batch in val_loader:
        rgb, gt_t, gt_q, bbox_c, cam_K, cls_id, img_meta = batch
        rgb = rgb.to(device)
        gt_t = gt_t.to(device)
        gt_q = gt_q.to(device)
        bbox_c = bbox_c.to(device)
        cam_K = cam_K.to(device)

        pred_q, pred_t = model(rgb, bbox_center=bbox_c, camera_matrix=cam_K)

        trans_err = torch.norm(pred_t - gt_t, dim=1).cpu()
        rot_err = quaternion_angle_deg(pred_q, gt_q).cpu()

        pq = pred_q.cpu().numpy()
        pt = pred_t.cpu().numpy()
        gq = gt_q.cpu().numpy()
        gt = gt_t.cpu().numpy()
        cls = cls_id.numpy()

        for i in range(rgb.size(0)):
            cls_i = int(cls[i])
            stats[cls_i]["trans_err"].append(float(trans_err[i]))
            stats[cls_i]["rot_err"].append(float(rot_err[i]))

            add_d = add_distance(pq[i], pt[i], gq[i],
                                 gt[i], cls[i], model_points)
            if add_d is None:
                continue

            diam = model_diameters[cls_i]
            thresh = diam * 0.1

            stats[cls_i]["add_total"] += 1
            stats[cls_i]["add_ok"] += int(add_d < thresh)
            stats[cls_i]["add_dist"].append(float(add_d))

            all_add_total += 1
            all_add_ok += int(add_d < thresh)
            all_add_dist.append(float(add_d))

        all_trans_err.extend(trans_err.tolist())
        all_rot_err.extend(rot_err.tolist())

        if save_predictions_path is not None:
            import re
            for i in range(rgb.size(0)):
                cls_i = int(cls[i])
                img_path = img_meta[i] if isinstance(
                    img_meta, list) else img_meta

                R_pred = quat_to_mat_np(pq[i])

                img_path_str = str(img_path)
                basename = os.path.basename(img_path_str)
                match = re.search(r'(\d+)', basename)
                img_num = int(match.group(1)) if match else 0

                key = f"{cls_i:02d}_{img_num:04d}"
                predictions[key] = {
                    'cam_R_m2c': R_pred.tolist(),
                    'cam_t_m2c': pt[i].tolist(),
                }

    overall = {
        "add_acc": all_add_ok / max(1, all_add_total),
        "add_mean": float(np.mean(all_add_dist)) if all_add_dist else 0.0,
        "trans_mean": float(np.mean(all_trans_err)) if all_trans_err else 0.0,
        "rot_mean": float(np.mean(all_rot_err)) if all_rot_err else 0.0,
    }

    per_class = {}
    for cls_id, d in stats.items():
        per_class[cls_id] = {
            "add_acc": d["add_ok"] / max(1, d["add_total"]),
            "add_mean": float(np.mean(d["add_dist"])) if d["add_dist"] else 0.0,
            "trans_mean": float(np.mean(d["trans_err"])) if d["trans_err"] else 0.0,
            "rot_mean": float(np.mean(d["rot_err"])) if d["rot_err"] else 0.0,
            "samples": d["add_total"],
        }

    if save_predictions_path is not None:
        os.makedirs(os.path.dirname(save_predictions_path)
                    or ".", exist_ok=True)
        print(f"DEBUG: Numero di predizioni salvate: {len(predictions)}")
        if predictions:
            print(f"DEBUG: Prime 5 chiavi: {list(predictions.keys())[:5]}")
        with open(save_predictions_path, 'w') as f:
            yaml.dump(predictions, f, default_flow_style=False)
        print(f"Predizioni salvate in: {save_predictions_path}")

    return per_class, overall


def run_add_eval(
    root_dir,
    models_dir,
    checkpoint,
    pred_boxes_path=None,
    batch_size=32,
    val_split=0.2,
    num_workers=4,
    seed=42,
    device="cuda",
    output_csv=None,
    save_predictions=None,
):
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    pred_boxes = load_pred_boxes(pred_boxes_path)
    model_points = load_model_points(models_dir)

    args_ns = argparse.Namespace(
        root_dir=root_dir, val_split=val_split, seed=seed)
    _, val_ds = make_datasets(args_ns, pred_boxes)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint}")

    model = PoseNetRGBGeometric(pretrained=False).to(device_t)
    ckpt = torch.load(checkpoint, map_location=device_t, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)

    per_class, overall = evaluate_add(
        model, val_loader, device_t, model_points, save_predictions_path=save_predictions)

    print("\n=== ADD per classe (threshold = 0.1 × diametro oggetto) ===")
    for cls_id in sorted(per_class.keys()):
        m = per_class[cls_id]
        print(
            f"Classe {cls_id:02d} | ADD@0.1d: {m['add_acc']*100:.2f}% | "
            f"ADD medio: {m['add_mean']*100:.2f}cm | "
            f"Trans err: {m['trans_mean']*100:.2f}cm | Rot err: {m['rot_mean']:.2f}deg | "
            f"n={m['samples']}"
        )

    print("\n=== Media globale ===")
    print(
        f"ADD@0.1d: {overall['add_acc']*100:.2f}% | "
        f"ADD medio: {overall['add_mean']*100:.2f}cm | "
        f"Trans err: {overall['trans_mean']*100:.2f}cm | Rot err: {overall['rot_mean']:.2f}deg"
    )

    if output_csv:
        data = []
        for cls_id in sorted(per_class.keys()):
            m = per_class[cls_id]
            data.append({
                "Classe": f"{cls_id:02d}",
                "ADD@0.1d": f"{m['add_acc']*100:.2f}%",
                "ADD_medio_cm": f"{m['add_mean']*100:.2f}",
                "Trans_err_cm": f"{m['trans_mean']*100:.2f}",
                "Rot_err_deg": f"{m['rot_mean']:.2f}",
                "n_samples": m["samples"],
            })

        data.append({
            "Classe": "GLOBALE",
            "ADD@0.1d": f"{overall['add_acc']*100:.2f}%",
            "ADD_medio_cm": f"{overall['add_mean']*100:.2f}",
            "Trans_err_cm": f"{overall['trans_mean']*100:.2f}",
            "Rot_err_deg": f"{overall['rot_mean']:.2f}",
            "n_samples": "",
        })

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"\nRisultati salvati in: {output_csv}")

    return per_class, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True,
                        help="Path a Linemod_preprocessed/data")
    parser.add_argument("--models_dir", required=True,
                        help="Path a Linemod_preprocessed/models")
    parser.add_argument("--checkpoint", default="best_pose.pt",
                        help="Checkpoint PoseNet da valutare")
    parser.add_argument("--pred_boxes", default=None,
                        help="File YAML con bbox YOLO (opzionale)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", default=None,
                        help="Percorso file CSV per salvare risultati")
    parser.add_argument("--save_predictions", default=None,
                        help="Percorso file YAML per salvare predizioni di posa (per la visualizzazione)")
    args = parser.parse_args()

    run_add_eval(
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
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()
