import os
import torch
from ultralytics import YOLO


def fine_tune_yolo(pretrained_weights, data_yaml, device='cpu', epochs=30, batch=16, imgsz=640, freeze=10, workers=0, project='yolo_runs', name='finetune'):
    """
    Fine-tune YOLO on LineMOD dataset with checkpoint saving.
    Monitors train/val loss, precision, recall, mAP50 at each epoch.
    Freezes backbone and trains detection head only.
    Saves checkpoints every epoch (best.pt and last.pt).
    """
    model = YOLO(pretrained_weights)
    print(f"\n{'='*70}")
    print(f"YOLO Fine-tuning on LineMOD (All Objects)")
    print(f"{'='*70}")
    print(f"Model: {pretrained_weights}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs} | Batch: {batch} | Image size: {imgsz}")
    print(f"Frozen layers: {freeze} (training head only)")
    print(f"Device: {device}")
    print(f"Checkpoints: saved every epoch to {project}/{name}/weights/")
    print(f"{'='*70}\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        freeze=freeze,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=1,
    )
    print("\n[Ultralytics Results Summary]")
    print(results)
    print(f"\n{'='*70}")
    print(f"Training Complete - Results saved in {project}/{name}")
    print(f"Checkpoints available:")
    weights_dir = os.path.join(project, name, 'weights')
    if os.path.exists(weights_dir):
        for ckpt in os.listdir(weights_dir):
            ckpt_path = os.path.join(weights_dir, ckpt)
            ckpt_size = os.path.getsize(ckpt_path) / (1024**2)
            print(f"  - {ckpt} ({ckpt_size:.1f} MB)")
    print(f"{'='*70}\n")
    return model, results


if __name__ == "__main__":
    output_dir = 'yolo_linemod_all'
    yaml_path = os.path.join(output_dir, 'linemod.yaml')
    pretrained_weights = 'yolov8n.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers = 0 if os.name == 'nt' else 2
    model = fine_tune_yolo(
        pretrained_weights=pretrained_weights,
        data_yaml=yaml_path,
        device=device,
        epochs=30,
        batch=16,
        imgsz=640,
        freeze=10,
        workers=workers,
        project='yolo_linemod_runs',
        name='finetune_all_objs'
    )
