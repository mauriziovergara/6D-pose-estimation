import argparse
import os
import random
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pose_net_rgb_geometric import PoseNetRGBGeometric
from dataset import LineModDataset, get_train_transforms, get_val_transforms
from types import SimpleNamespace
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    trimesh = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pred_boxes(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_model_points(models_dir):
    if trimesh is None:
        raise ImportError("Install trimesh for ADD: pip install trimesh")
    pts = {}
    for name in os.listdir(models_dir):
        if not name.endswith(".ply"):
            continue
        obj_id = int(name.split("_")[1].split(".")[0])
        mesh = trimesh.load(os.path.join(models_dir, name), process=False)
        pts[obj_id] = np.asarray(
            mesh.vertices, dtype=np.float32) / 1000.0  # to meters
    return pts


def freeze_backbone_ratio(model, freeze_ratio=0.0):
    freeze_ratio = min(max(freeze_ratio, 0.0), 1.0)
    modules = list(model.rgb_backbone.children())
    cutoff = int(len(modules) * freeze_ratio + 1e-9)
    for idx, m in enumerate(modules):
        req = idx >= cutoff
        for p in m.parameters():
            p.requires_grad = req


def unfreeze_backbone(model):
    for p in model.rgb_backbone.parameters():
        p.requires_grad = True


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_add, awl_criterion):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'awl_state': awl_criterion.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_add': best_add,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, awl_criterion):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if 'awl_state' in ckpt:
        awl_criterion.load_state_dict(ckpt['awl_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and ckpt.get('scheduler_state') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    if 'rng_state' in ckpt:
        rng = ckpt['rng_state']
        if rng.get('python'):
            random.setstate(rng['python'])
        if rng.get('numpy'):
            np.random.set_state(rng['numpy'])
        if rng.get('torch') is not None:
            torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng.get('cuda'):
            torch.cuda.set_rng_state_all(rng['cuda'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_add = ckpt.get('best_add', 0.0)
    return start_epoch, best_add


def quat_to_mat_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def rotation_loss(pred_q, gt_q):
    """Chordal distance loss for quaternions."""
    pred_q = pred_q / (pred_q.norm(dim=1, keepdim=True) + 1e-8)
    gt_q = gt_q / (gt_q.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred_q * gt_q, dim=1).abs()
    return (1.0 - dot).mean()


def geodesic_rotation_loss(pred_q, gt_q):
    """Geodesic distance loss on SO(3) manifold (true angle in radians)."""
    pred_q = pred_q / (pred_q.norm(dim=1, keepdim=True) + 1e-8)
    gt_q = gt_q / (gt_q.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred_q * gt_q, dim=1).abs()
    # Clamp to avoid numerical issues with arccos near ±1
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    # Geodesic distance: 2 * arccos(|dot|) gives rotation angle in radians [0, π]
    return (2.0 * torch.acos(dot)).mean()


class AutoWeightedLoss(torch.nn.Module):
    """Automatic Weighting Loss (Kendall et al.) for multi-task learning.
    Loss = (1/2σ_rot²)·L_rot + (1/2σ_trans²)·L_trans + log(σ_rot) + log(σ_trans)
    """

    def __init__(self):
        super().__init__()
        # Initialize log_sigma parameters (learnable uncertainty)
        self.log_sigma_rot = torch.nn.Parameter(torch.zeros(1))
        self.log_sigma_trans = torch.nn.Parameter(torch.zeros(1))

    def forward(self, loss_rot, loss_trans, lambda_rot=1.0, lambda_trans=1.0):
        # Compute weighted loss with uncertainty regularization
        precision_rot = torch.exp(-2*self.log_sigma_rot)
        precision_trans = torch.exp(-2*self.log_sigma_trans)

        loss = (0.5 * precision_rot * lambda_rot * loss_rot + self.log_sigma_rot +
                0.5 * precision_trans * lambda_trans * loss_trans + self.log_sigma_trans)

        return loss

    def get_weights(self):
        return {
            'w_rot': torch.exp(-2 * self.log_sigma_rot).item(),
            'w_trans': torch.exp(-2 * self.log_sigma_trans).item(),
            'sigma_rot': torch.exp(self.log_sigma_rot).item(),
            'sigma_trans': torch.exp(self.log_sigma_trans).item(),
        }


def compute_add(batch_pred_r, batch_pred_t, batch_gt_r, batch_gt_t, batch_cls, model_points, add_thresh):
    ok = 0
    total = 0
    add_values = []
    for i in range(batch_pred_r.shape[0]):
        cls_id = int(batch_cls[i])
        if cls_id not in model_points:
            continue
        pts = model_points[cls_id]
        R_pred = quat_to_mat_np(batch_pred_r[i])
        R_gt = quat_to_mat_np(batch_gt_r[i])
        pred_pts = (R_pred @ pts.T).T + batch_pred_t[i]
        gt_pts = (R_gt @ pts.T).T + batch_gt_t[i]
        add = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
        add_values.append(add)
        ok += 1 if add < add_thresh else 0
        total += 1
    mean_add = np.mean(add_values) if add_values else 0.0
    return ok, total, mean_add


def make_datasets(args, pred_boxes):
    base = LineModDataset(
        root_dir=args.root_dir,
        transform=None,
        use_pred_boxes=bool(pred_boxes),
        pred_boxes=pred_boxes,
    )
    val_len = max(1, int(len(base) * args.val_split))
    train_len = len(base) - val_len
    gen = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(
        range(len(base)), [train_len, val_len], generator=gen)
    train_items = [base.data_items[i] for i in train_idx]
    val_items = [base.data_items[i] for i in val_idx]

    train_ds = LineModDataset(
        root_dir=args.root_dir,
        transform=get_train_transforms(),
        use_pred_boxes=bool(pred_boxes),
        data_items=train_items,
        pred_boxes=pred_boxes,
    )
    val_ds = LineModDataset(
        root_dir=args.root_dir,
        transform=get_val_transforms(),
        use_pred_boxes=bool(pred_boxes),
        data_items=val_items,
        pred_boxes=pred_boxes,
    )
    return train_ds, val_ds


def run_epoch(model, loader, optimizer, device, awl_criterion, rot_loss_fn, lambda_rot=1.0, lambda_trans=1.0):
    model.train()
    awl_criterion.train()
    total = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        rgb, gt_t, gt_q, bbox_c, cam_K, _, _ = batch
        rgb = rgb.to(device)
        gt_t = gt_t.to(device)
        gt_q = gt_q.to(device)
        bbox_c = bbox_c.to(device)
        cam_K = cam_K.to(device)

        pred_q, pred_t = model(rgb, bbox_center=bbox_c, camera_matrix=cam_K)
        loss_r = rot_loss_fn(pred_q, gt_q)
        loss_t = F.l1_loss(pred_t, gt_t)
        loss = awl_criterion(
            loss_r, loss_t, lambda_rot=lambda_rot, lambda_trans=lambda_trans)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * rgb.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, model_points, add_thresh, awl_criterion, rot_loss_fn, lambda_rot=1.0, lambda_trans=1.0):
    model.eval()
    awl_criterion.eval()
    total = 0.0
    ok_add = 0
    tot_add = 0
    all_add_values = []
    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch in pbar:
        rgb, gt_t, gt_q, bbox_c, cam_K, cls_id, _ = batch
        rgb = rgb.to(device)
        gt_t = gt_t.to(device)
        gt_q = gt_q.to(device)
        bbox_c = bbox_c.to(device)
        cam_K = cam_K.to(device)

        pred_q, pred_t = model(rgb, bbox_center=bbox_c, camera_matrix=cam_K)
        loss_r = rot_loss_fn(pred_q, gt_q)
        loss_t = F.l1_loss(pred_t, gt_t)
        loss = awl_criterion(
            loss_r, loss_t, lambda_rot=lambda_rot, lambda_trans=lambda_trans)
        total += loss.item() * rgb.size(0)

        pq = pred_q.cpu().numpy()
        pt = pred_t.cpu().numpy()
        gq = gt_q.cpu().numpy()
        gt = gt_t.cpu().numpy()
        cls = cls_id.numpy()
        ok, tot, mean_add = compute_add(
            pq, pt, gq, gt, cls, model_points, add_thresh)
        ok_add += ok
        tot_add += tot
        all_add_values.append(mean_add)

    val_loss = total / len(loader.dataset)
    mean_add_global = np.mean(all_add_values) if all_add_values else 0.0
    add_acc = ok_add / max(1, tot_add)
    return val_loss, mean_add_global, add_acc


def build_optimizer(model, awl_criterion, optimizer_type, lr, weight_decay):
    # Combine model and AWL parameters
    params = list(model.parameters()) + list(awl_criterion.parameters())
    opt_name = optimizer_type.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_pose_net(
    root_dir,
    models_dir,
    pred_boxes_path=None,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    learning_rate=None,
    weight_decay=1e-5,
    lambda_rot=1.0,
    lambda_trans=1.0,
    val_split=0.2,
    add_thresh=0.02,
    num_workers=4,
    seed=42,
    device="cuda",
    checkpoint_best="best_pose.pt",
    checkpoint_last="last_pose.pt",
    resume_from=None,
    eval_only=False,
    scheduler_patience=0,
    early_stop_patience=0,
    optimizer_type="adam",
    freeze_backbone_ratio_val=0.0,
    unfreeze_epoch=None,
    rotation_loss_type="chordal",
):
    lr_eff = learning_rate if learning_rate is not None else lr
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    pred_boxes = load_pred_boxes(pred_boxes_path)
    model_points = load_model_points(models_dir)

    if rotation_loss_type.lower() == "geodesic":
        rot_loss_fn = geodesic_rotation_loss
        print(f"✓ Rotation loss: GEODESIC (2*arccos(|dot|))")
    else:
        rot_loss_fn = rotation_loss
        print(f"✓ Rotation loss: CHORDAL (1-|dot|)")

    args_ns = SimpleNamespace(
        root_dir=root_dir, val_split=val_split, seed=seed)
    train_ds, val_ds = make_datasets(args_ns, pred_boxes)
    print(f"✓ Dataset splits creati: train={len(train_ds)}, val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"✓ DataLoaders creati (num_workers={num_workers})")

    print(
        f"Inizializzo PoseNetRGBGeometric (pretrained=True) su {device_t}...")
    model = PoseNetRGBGeometric(pretrained=True).to(device_t)
    print(f"✓ Modello caricato su {device_t}")
    freeze_backbone_ratio(model, freeze_backbone_ratio_val)
    print(f"✓ Backbone freeze ratio applicato: {freeze_backbone_ratio_val}")

    awl_criterion = AutoWeightedLoss().to(device_t)
    print(f"✓ AutoWeightedLoss inizializzata")

    optimizer = build_optimizer(
        model, awl_criterion, optimizer_type, lr_eff, weight_decay)
    print(f"✓ Optimizer {optimizer_type} creato (lr={lr_eff:.2e})")
    scheduler = None
    if scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=scheduler_patience,
            min_lr=1e-7)
        print(f"✓ Scheduler attivo (patience={scheduler_patience})")

    start_epoch = 1
    best_add = float('inf')
    if resume_from:
        print(f"Caricamento checkpoint da {resume_from}...")
        start_epoch, best_add = load_checkpoint(
            resume_from, model, optimizer, scheduler, awl_criterion)
        weights = awl_criterion.get_weights()

        print(
            f"✓ Riprendo da {resume_from}: start_epoch={start_epoch}, best_add (mean)={best_add:.6f}")

        print(
            f"✓ AWL weights: w_rot={weights['w_rot']:.4f}, w_trans={weights['w_trans']:.4f}, σ_rot={weights['sigma_rot']:.4f}, σ_trans={weights['sigma_trans']:.4f}")

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_eff
            param_group['weight_decay'] = weight_decay
        print(
            f"✓ Parametri optimizer aggiornati: lr={lr_eff:.2e}, weight_decay={weight_decay:.2e}")

        if scheduler_patience > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=scheduler_patience,
                min_lr=1e-7)
            print(f"✓ Scheduler ricreato con nuovi parametri")

        freeze_backbone_ratio(model, freeze_backbone_ratio_val)
        print(
            f"✓ Backbone freeze ratio riapplicato: {freeze_backbone_ratio_val}")
    else:
        print("✓ Nessun checkpoint da riprendere, parto da epoch 1")

    if eval_only:
        print("Modalità EVAL_ONLY attiva, carico best checkpoint...")
        ckpt = torch.load(
            checkpoint_best, map_location=device_t, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'awl_state' in ckpt:
            awl_criterion.load_state_dict(ckpt['awl_state'])
        print("Eseguo valutazione...")
        val_loss, mean_add_global, add_acc = evaluate(
            model, val_loader, device_t, model_points, add_thresh, awl_criterion, rot_loss_fn, lambda_rot=lambda_rot, lambda_trans=lambda_trans)
        weights = awl_criterion.get_weights()
        print(
            f"Eval only - loss: {val_loss:.4f} | ADD medio: {mean_add_global:.6f} | ADD<{add_thresh*100:.0f}cm: {add_acc*100:.2f}%")
        print(
            f"AWL weights: w_rot={weights['w_rot']:.4f}, w_trans={weights['w_trans']:.4f}")
        return model, []

    history = []
    no_improve = 0
    print("Start training")
    for epoch in range(start_epoch, epochs + 1):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            unfreeze_backbone(model)
            print(f"Unfreeze backbone at epoch {epoch}")

        train_loss = run_epoch(model, train_loader,
                               optimizer, device_t, awl_criterion, rot_loss_fn, lambda_rot=lambda_rot, lambda_trans=lambda_trans)
        val_loss, mean_add_global, add_acc = evaluate(
            model, val_loader, device_t, model_points, add_thresh, awl_criterion, rot_loss_fn, lambda_rot=lambda_rot, lambda_trans=lambda_trans)
        if scheduler:
            scheduler.step(val_loss)

        weights = awl_criterion.get_weights()
        print(f"[{epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | ADD_medio={mean_add_global:.6f} | ADD<{add_thresh*100:.0f}cm={add_acc*100:.2f}% | lr={optimizer.param_groups[0]['lr']:.2e} | w_rot={weights['w_rot']:.3f} | w_trans={weights['w_trans']:.3f}")

        if mean_add_global < best_add:
            best_add = mean_add_global
            save_checkpoint(checkpoint_best, epoch, model,
                            optimizer, scheduler, best_add, awl_criterion)
            print(
                f"  -> saved best to {checkpoint_best} (ADD medio={best_add:.6f})")
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(checkpoint_last, epoch, model,
                        optimizer, scheduler, best_add, awl_criterion)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mean_add': mean_add_global,
            'add_acc': add_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'w_rot': weights['w_rot'],
            'w_trans': weights['w_trans'],
        })

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(
                f"Early stop: nessun miglioramento ADD per {early_stop_patience} epoche consecutive")
            break

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True,
                        help="Path a Linemod_preprocessed/data")
    parser.add_argument("--models_dir", required=True,
                        help="Path a Linemod_preprocessed/models")
    parser.add_argument("--pred_boxes", default=None,
                        help="File YAML con bbox YOLO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_rot", type=float, default=1.0)
    parser.add_argument("--lambda_trans", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"], help="Optimizer: adam o adamw")
    parser.add_argument("--freeze_backbone_ratio_val", type=float, default=0.0,
                        help="Quota di backbone ResNet da congelare (0=none, 0.5=prima metà, 1=tutto)")
    parser.add_argument("--unfreeze_epoch", type=int, default=None,
                        help="Epoca a cui sbloccare tutta la backbone; None per non sbloccare")
    parser.add_argument("--scheduler_patience", type=int, default=0,
                        help="Patience (epochs) for ReduceLROnPlateau; 0 to disable")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Patience (epochs without ADD improvement) before early stop; 0 to disable")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--add_thresh", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint_best", default="best_pose.pt")
    parser.add_argument("--checkpoint_last", default="last_pose.pt")
    parser.add_argument("--resume_from", default=None,
                        help="Path checkpoint da cui riprendere (ripristina optimizer/scheduler/epoca)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--rotation_loss_type", type=str, default="chordal",
                        choices=["chordal", "geodesic"],
                        help="Rotation loss: chordal (1-|dot|, faster) or geodesic (2*arccos(|dot|), geometrically accurate)")
    args = parser.parse_args()

    train_pose_net(
        root_dir=args.root_dir,
        models_dir=args.models_dir,
        pred_boxes_path=args.pred_boxes,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
        val_split=args.val_split,
        add_thresh=args.add_thresh,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        checkpoint_best=args.checkpoint_best,
        checkpoint_last=args.checkpoint_last,
        resume_from=args.resume_from,
        eval_only=args.eval_only,
        scheduler_patience=args.scheduler_patience,
        early_stop_patience=args.early_stop_patience,
        optimizer_type=args.optimizer,
        freeze_backbone_ratio_val=args.freeze_backbone_ratio_val,
        unfreeze_epoch=args.unfreeze_epoch,
        rotation_loss_type=args.rotation_loss_type,
    )
    return


if __name__ == "__main__":
    main()
