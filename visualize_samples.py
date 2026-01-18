import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_camera_matrix(info_path, img_id=0):
    with open(info_path, 'r') as f:
        info = yaml.safe_load(f)

    cam_K = info[img_id]['cam_K']
    K = np.array(cam_K).reshape(3, 3)
    return K


def project_3d_to_2d(points_3d, R, t, K):
    # Convert rotation matrix and translation to proper format
    R = np.array(R).reshape(3, 3)
    t = np.array(t).reshape(3, 1)

    # Transform 3D points to camera coordinates
    points_3d = np.array(points_3d).T  # (3, N)
    points_cam = R @ points_3d + t  # (3, N)

    # Project to image plane
    points_2d_hom = K @ points_cam  # (3, N)
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]  # (2, N)

    return points_2d.T  # (N, 2)


def create_coordinate_axes(axis_length=60):
    axes = np.array([
        [0, 0, 0],  # Origin
        [axis_length, 0, 0],  # X-axis (red)
        [0, axis_length, 0],  # Y-axis (green)
        [0, 0, axis_length],  # Z-axis (blue)
    ], dtype=np.float32)
    return axes


def draw_axes_on_image(img, axes_2d, line_thickness=3):
    origin = tuple(axes_2d[0].astype(int))
    x_end = tuple(axes_2d[1].astype(int))
    y_end = tuple(axes_2d[2].astype(int))
    z_end = tuple(axes_2d[3].astype(int))

    # Draw X-axis in red
    cv2.line(img, origin, x_end, (0, 0, 255), line_thickness)
    cv2.putText(img, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw Y-axis in green
    cv2.line(img, origin, y_end, (0, 255, 0), line_thickness)
    cv2.putText(img, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw Z-axis in blue
    cv2.line(img, origin, z_end, (255, 0, 0), line_thickness)
    cv2.putText(img, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return img


def load_yolo_predictions(yolo_pred_path):
    if not os.path.exists(yolo_pred_path):
        print(f"Warning: YOLO predictions file not found: {yolo_pred_path}")
        return {}

    with open(yolo_pred_path, 'r') as f:
        yolo_preds = yaml.safe_load(f)

    preds = yolo_preds if yolo_preds else {}
    print(f"Loaded {len(preds)} YOLO predictions")
    if preds:
        print(f"Sample YOLO keys: {list(preds.keys())[:3]}")
    return preds


def load_pose_predictions(best_pose_path):
    if not os.path.exists(best_pose_path):
        print(
            f"Warning: Best pose predictions file not found: {best_pose_path}")
        return {}

    with open(best_pose_path, 'r') as f:
        pose_preds = yaml.safe_load(f)

    preds = pose_preds if pose_preds else {}
    print(f"Loaded {len(preds)} pose predictions")
    if preds:
        print(f"Sample pose keys: {list(preds.keys())[:3]}")
    return preds


def visualize_random_samples(root_dir, output_dir=None, axis_length=30,
                             yolo_pred_path=None, pose_pred_path=None):
    # Find all valid object folders
    valid_classes = sorted([d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))
                           and os.path.exists(os.path.join(root_dir, d, 'gt.yml'))])

    print(f"Found {len(valid_classes)} object classes: {valid_classes}")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if yolo_pred_path is None or pose_pred_path is None:
        raise ValueError("yolo_pred_path and pose_pred_path are required")

    yolo_preds = load_yolo_predictions(yolo_pred_path)
    pose_preds = load_pose_predictions(pose_pred_path)
    print(f"Loaded YOLO predictions from: {yolo_pred_path}")
    print(f"Loaded Pose predictions from: {pose_pred_path}")

    # Process each object class
    all_visualizations = []

    # Available pose predictions per class
    available_samples = {}
    for pose_key in pose_preds.keys():
        class_id_p, img_id_str = pose_key.split('_')
        img_id = int(img_id_str)
        if class_id_p not in available_samples:
            available_samples[class_id_p] = []
        available_samples[class_id_p].append(img_id)

    print(
        f"\nClasses with pose predictions: {list(available_samples.keys())}")

    for class_id in valid_classes:
        print(f"\nProcessing object {class_id}...")

        base_path = os.path.join(root_dir, class_id)
        rgb_dir = os.path.join(base_path, 'rgb')
        info_path = os.path.join(base_path, 'info.yml')

        if class_id not in available_samples:
            print(
                f"  No pose predictions available for class {class_id}, skipping...")
            continue

        available_ids = sorted(set(available_samples[class_id]))
        selected_id = None
        selected_yolo_key = None

        for img_id in available_ids:
            possible_yolo_keys = [
                f"{class_id}/rgb/{img_id:04d}.png",
                f"{class_id}/{img_id:04d}",
                f"{class_id}/rgb/{img_id:04d}",
            ]

            key_yolo = next(
                (k for k in possible_yolo_keys if k in yolo_preds), None)
            if key_yolo is None:
                continue

            img_name = f"{img_id:04d}.png"
            img_path = os.path.join(rgb_dir, img_name)
            if not os.path.exists(img_path):
                continue

            selected_id = img_id
            selected_yolo_key = key_yolo
            break

        if selected_id is None:
            print(
                f"  No overlapping YOLO+Pose predictions (or image missing) for class {class_id}, skipping...")
            continue

        print(f"  Using first overlapping sample: {selected_id:04d}")

        # Load image
        img_name = f"{selected_id:04d}.png"
        img_path = os.path.join(rgb_dir, img_name)

        if not os.path.exists(img_path):
            print(f"  Warning: Image {img_path} not found, skipping...")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read image {img_path}, skipping...")
            continue

        img_display = img.copy()

        # Load camera matrix (cam_K not included in pose predictions)
        K = load_camera_matrix(info_path, selected_id)

        # Pose key
        key_pose = f"{class_id}_{selected_id:04d}"
        pose_pred = pose_preds[key_pose]

        yolo_bbox_data = yolo_preds[selected_yolo_key]
        if isinstance(yolo_bbox_data, list):
            x1, y1, x2, y2 = yolo_bbox_data
        elif isinstance(yolo_bbox_data, dict) and 'bbox' in yolo_bbox_data:
            x1, y1, x2, y2 = yolo_bbox_data['bbox']
        else:
            print(
                f"  Warning: Unexpected YOLO bbox format for {selected_yolo_key}, skipping...")
            continue

        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox[2] <= 0 or bbox[3] <= 0:
            print(
                f"  Warning: Invalid YOLO bbox for {selected_yolo_key}, skipping...")
            continue

        R_mat = np.array(pose_pred['cam_R_m2c']).reshape(3, 3)
        t_vec = np.array(pose_pred['cam_t_m2c'])

        label = "PRED"

        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(img_display, (x, y),
                      (x + w, y + h), (255, 255, 0), 2)

        # Create and project 3D coordinate axes
        axes_3d = create_coordinate_axes(axis_length)
        axes_2d = project_3d_to_2d(axes_3d, R_mat, t_vec, K)

        # Draw coordinate axes
        img_display = draw_axes_on_image(img_display, axes_2d)

        # Add text information
        text = f"Object {class_id} - Image {img_id} [{label}]"
        cv2.putText(img_display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Store for visualization
        all_visualizations.append({
            'image': cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB),
            'class_id': class_id,
            'img_id': img_id,
            'label': label
        })

        # Save individual image if output_dir is specified
        if output_dir:
            output_path = os.path.join(
                output_dir, f"object_{class_id}_img_{img_id:04d}_{label}.png")
            cv2.imwrite(output_path, img_display)
            print(f"  Saved: {output_path}")

    # Create a grid visualization
    if all_visualizations:
        num_images = len(all_visualizations)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, vis_data in enumerate(all_visualizations):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            ax.imshow(vis_data['image'])
            ax.set_title(
                f"Object {vis_data['class_id']} - Image {vis_data['img_id']}")
            ax.axis('off')

        # Hide empty subplots
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if output_dir:
            grid_path = os.path.join(output_dir, "all_objects_grid.png")
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved grid visualization: {grid_path}")

        plt.show()

        print(
            f"\nVisualized {len(all_visualizations)} images from {len(valid_classes)} object classes")
    else:
        print("No images to visualize!")


if __name__ == "__main__":
    ROOT_DIR = r"../Linemod_preprocessed/Linemod_preprocessed/data"
    OUTPUT_DIR = r"./visualizations"

    YOLO_PRED_PATH = r"./yolo_predictions.yaml"
    POSE_PRED_PATH = r"./best_pose_predictions.yaml"

    visualize_random_samples(
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR,
        axis_length=30,
        yolo_pred_path=YOLO_PRED_PATH,
        pose_pred_path=POSE_PRED_PATH,
    )
