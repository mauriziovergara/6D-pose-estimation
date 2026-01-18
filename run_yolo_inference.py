import os
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch


def run_yolo_inference(
    yolo_weights_path,
    dataset_root,
    output_path='yolo_predictions.yaml',
    conf_threshold=0.25,
    device='cuda'
):

    # Check inputs
    if not os.path.exists(yolo_weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")

    data_root = os.path.join(dataset_root, 'data')
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    print(f"\n{'='*70}")
    print(f"YOLO Inference on LineMOD Dataset")
    print(f"{'='*70}")
    print(f"Weights: {yolo_weights_path}")
    print(f"Dataset: {dataset_root}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*70}\n")

    # Load YOLO model
    model = YOLO(yolo_weights_path)
    model.to(device)

    # Get class names from model
    class_names = model.names
    print(f"Classes loaded: {class_names}\n")

    # Discover all object folders
    folders = sorted([f for f in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, f))])

    predictions = {}
    total_images = 0
    detected_images = 0

    print("Running YOLO inference on all images...\n")

    for folder_id in folders:
        folder_path = os.path.join(data_root, folder_id)
        rgb_dir = os.path.join(folder_path, 'rgb')

        if not os.path.exists(rgb_dir):
            continue

        # Get all PNG images in folder
        img_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.endswith('.png')])

        for img_file in tqdm(img_files, desc=f"Folder {folder_id}"):
            sample_id = int(os.path.splitext(img_file)[0])
            img_path = os.path.join(rgb_dir, img_file)

            # Run YOLO inference
            results = model(img_path, conf=conf_threshold, verbose=False)[0]

            total_images += 1

            # Get best detection (highest confidence)
            best_box = None
            max_conf = -1
            best_class = None

            for box in results.boxes:
                conf = box.conf.item()
                if conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy(
                    ).tolist()
                    best_class = int(box.cls.item())

            # Save prediction if found
            if best_box is not None:
                detected_images += 1
                key = f"{folder_id}/{sample_id:04d}"
                predictions[key] = {
                    'bbox': best_box,
                    'confidence': float(max_conf),
                    'class_id': best_class,
                    'class_name': class_names[best_class],
                    'folder_id': int(folder_id),
                    'sample_id': sample_id
                }

    # Save predictions
    print(f"\n{'='*70}")
    print(f"Inference Complete")
    print(f"{'='*70}")
    print(f"Total images: {total_images}")
    print(
        f"Images with detections: {detected_images} ({100*detected_images/total_images:.1f}%)")
    print(f"Images without detections: {total_images - detected_images}")
    print(f"{'='*70}\n")

    # Determine output format from extension
    output_ext = Path(output_path).suffix.lower()

    if output_ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {output_path} (JSON format)")
    else:  # Default to YAML
        with open(output_path, 'w') as f:
            yaml.dump(predictions, f, default_flow_style=False)
        print(f"Predictions saved to {output_path} (YAML format)")

    return predictions


if __name__ == "__main__":
    # Configuration
    YOLO_WEIGHTS = r"C:\Users\DELL\OneDrive\Desktop\Università\00_MAGISTRALE\DAAI\Project\project_6\yolo_linemod_runs\finetune_all_objs\weights\best.pt"
    DATASET_ROOT = r"C:\Users\DELL\OneDrive\Desktop\Università\00_MAGISTRALE\DAAI\Project\project_6\Linemod_preprocessed\Linemod_preprocessed"
    OUTPUT_PATH = r"C:\Users\DELL\OneDrive\Desktop\Università\00_MAGISTRALE\DAAI\Project\project_6\yolo_predictions.yaml"

    # Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions = run_yolo_inference(
        yolo_weights_path=YOLO_WEIGHTS,
        dataset_root=DATASET_ROOT,
        output_path=OUTPUT_PATH,
        conf_threshold=0.25,
        device=device
    )

    # Print sample predictions
    print("\nSample predictions:")
    for i, (key, pred) in enumerate(list(predictions.items())[:5]):
        print(f"\n{key}:")
        print(f"  bbox: {pred['bbox']}")
        print(f"  confidence: {pred['confidence']:.3f}")
        print(f"  class: {pred['class_name']}")
