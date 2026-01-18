import torch
from torch.utils.data import Dataset
import cv2
import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageOps
from torchvision import transforms as T


def get_train_transforms():
    """Training transforms with safe augmentations (no rotation/flip)."""
    return T.Compose([
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """Validation transforms without augmentation."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class LineModDataset(Dataset):
    def __init__(self, root_dir, valid_classes=None, transform=None, use_pred_boxes=False, data_items=None, pred_boxes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.use_pred_boxes = use_pred_boxes
        self.default_cam_K = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        if data_items is not None:
            self.data_items = data_items
            self.pred_boxes = pred_boxes if pred_boxes is not None else {}
            self.use_pred_boxes = self.use_pred_boxes and bool(self.pred_boxes)
            print(
                f"Dataset PoseNet: riuso di {len(self.data_items)} items pre-caricati.")
            return

        if pred_boxes is not None:
            self.pred_boxes = pred_boxes
        else:
            self.pred_boxes = {}
            if self.use_pred_boxes:
                box_file = os.path.join(root_dir, 'yolo_predicted_boxes.yaml')
                if os.path.exists(box_file):
                    print(f"Caricamento BBox predette da {box_file}...")
                    with open(box_file, 'r') as f:
                        self.pred_boxes = yaml.safe_load(f)
                else:
                    print(
                        "ATTENZIONE: File box predette non trovato. Userò le Ground Truth.")
                    self.use_pred_boxes = False

        self.data_items = []

        if valid_classes is None:
            valid_classes = sorted([d for d in os.listdir(root_dir)
                                    if os.path.isdir(os.path.join(root_dir, d))
                                    and os.path.exists(os.path.join(root_dir, d, 'gt.yml'))])

        print(f"Dataset PoseNet: Caricamento classi {valid_classes}...")

        for class_id in valid_classes:
            base_path = os.path.join(root_dir, class_id)
            img_dir = os.path.join(base_path, 'rgb')
            gt_path = os.path.join(base_path, 'gt.yml')

            if not os.path.exists(gt_path):
                continue

            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)

            for img_id, annos in gt_data.items():
                img_name = f"{img_id:04d}.png"
                img_path = os.path.join(img_dir, img_name)

                if not os.path.exists(img_path):
                    img_name = f"{img_id:04d}.jpg"
                    img_path = os.path.join(img_dir, img_name)
                    if not os.path.exists(img_path):
                        continue

                anno = None
                if class_id == "02":
                    for a in annos:
                        if a.get('obj_id') == 2:
                            anno = a
                            break
                    if anno is None and annos:
                        anno = annos[0]
                else:
                    anno = annos[0] if annos else None

                if anno is None or 'cam_t_m2c' not in anno or 'cam_R_m2c' not in anno or 'obj_bb' not in anno:
                    continue

                gt_trans = np.array(
                    anno['cam_t_m2c'], dtype=np.float32) / 1000.0
                r_mat = np.array(anno['cam_R_m2c'],
                                 dtype=np.float32).reshape(3, 3)
                q_scipy = R.from_matrix(r_mat).as_quat().astype(np.float32)
                gt_quat = np.array(
                    [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float32)

                gt_bbox = anno['obj_bb']
                cam_K = self.default_cam_K

                rel_key = f"{class_id}/rgb/{img_name}"

                self.data_items.append({
                    'img_path': img_path,
                    'gt_trans': gt_trans,
                    'gt_quat': gt_quat,
                    'gt_bbox': gt_bbox,
                    'rel_key': rel_key,
                    'class_id': int(class_id),
                    'cam_K': cam_K,
                    'img_id': int(img_id)
                })

        print(f"Dataset caricato: {len(self.data_items)} items.")

    def __len__(self):
        return len(self.data_items)

    def pad_to_square(self, img):
        w, h = img.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return ImageOps.expand(img, (p_left, p_top, p_right, p_bottom), fill=0)

    def _find_pred_box(self, class_id, img_id, rel_key):
        candidates = [
            rel_key,
            f"{class_id}/{img_id:04d}",
            f"{class_id}/{img_id:04d}.png",
            f"{class_id}/{img_id:04d}.jpg",
            f"{class_id}/rgb/{img_id:04d}",
            f"{class_id}/rgb/{img_id:04d}.png",
            f"{class_id}/rgb/{img_id:04d}.jpg",
            f"{class_id}_{img_id:04d}",
            f"{class_id}_{img_id:04d}.png",
            f"{class_id}_{img_id:04d}.jpg",
        ]
        for key in candidates:
            if key in self.pred_boxes:
                return self.pred_boxes[key]
        return None

    def __getitem__(self, idx):
        item = self.data_items[idx]
        image_cv = cv2.imread(item['img_path'])
        if image_cv is None:
            return (
                torch.zeros((3, 224, 224)),
                torch.zeros(3),
                torch.tensor([1., 0., 0., 0.]),
                torch.zeros(2),
                torch.eye(3),
                torch.tensor(-1),
                item['img_path']
            )

        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        h_img, w_img = image_cv.shape[:2]

        bbox = item['gt_bbox']  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        x1, y1, x2, y2 = x, y, x + w, y + h

        if self.use_pred_boxes:
            key = item['rel_key']
            pred = self._find_pred_box(item['class_id'], item['img_id'], key)
            if pred is not None:
                px1, py1, px2, py2 = map(int, pred)
                x1, y1, x2, y2 = px1, py1, px2, py2

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        bbox_center = np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        if x2 <= x1 or y2 <= y1:
            cropped = image_cv
        else:
            cropped = image_cv[y1:y2, x1:x2]

        image_pil = Image.fromarray(cropped)
        image_pil = self.pad_to_square(image_pil)

        if self.transform is not None:
            final_img = self.transform(image_pil)
        else:
            final_img = T.ToTensor()(image_pil)

        cam_K = torch.tensor(
            item.get('cam_K', self.default_cam_K), dtype=torch.float32)

        return (
            final_img,
            torch.tensor(item['gt_trans'], dtype=torch.float32),
            torch.tensor(item['gt_quat'], dtype=torch.float32),
            torch.tensor(bbox_center, dtype=torch.float32),
            cam_K,
            torch.tensor(item['class_id']),
            item['img_path']
        )
