import os
import yaml
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def convert_to_yolo_format(bbox, img_width, img_height, class_idx):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [class_idx, x_center, y_center, w_norm, h_norm]


def prepare_dataset():
    ROOT_DIR = '../data/linemod'
    OUTPUT_DIR = '../data/linemod_yolo'

    # 1. Trova cartelle valide
    valid_folders = sorted([d for d in os.listdir(ROOT_DIR)
                            if os.path.isdir(os.path.join(ROOT_DIR, d))
                            and os.path.exists(os.path.join(ROOT_DIR, d, 'gt.yml'))])

    class_map = {name: i for i, name in enumerate(valid_folders)}
    print(f"Classi rilevate: {class_map}")

    with open('../data/class_map.yaml', 'w') as f:
        yaml.dump(class_map, f)

    # 2. Scansione con RINOMINA (Fix sovrascrittura)
    images_data = {}

    print("Scansione dataset (Modalità Nomi Unici)...")
    for folder in valid_folders:
        cls_id = class_map[folder]
        rgb_dir = os.path.join(ROOT_DIR, folder, 'rgb')
        gt_path = os.path.join(ROOT_DIR, folder, 'gt.yml')

        with open(gt_path, 'r') as f:
            gt = yaml.safe_load(f)

        # Elenca tutti i file immagine
        all_files = os.listdir(rgb_dir)
        img_files = [f for f in all_files if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]

        print(f" -> Cartella {folder}: Trovati {len(img_files)} file.")

        for img_file in img_files:
            try:
                base_name = os.path.splitext(img_file)[0]
                img_id = int(base_name)
            except ValueError:
                continue

            if img_id in gt:
                full_path = os.path.join(rgb_dir, img_file)

                unique_id = f"{folder}_{img_id:04d}"

                if unique_id not in images_data:
                    img = cv2.imread(full_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    images_data[unique_id] = {
                        'path': full_path,
                        'h': h, 'w': w,
                        'labels': [],
                        'folder': folder,
                        'orig_name': img_file
                    }

                annos = gt[img_id]
                for obj in annos:
                    # Per la classe 02, considera solo annotazioni con obj_id == 2
                    if folder == "02" and obj.get('obj_id') != 2:
                        continue

                    if 'obj_bb' in obj:
                        yolo_bbox = convert_to_yolo_format(obj['obj_bb'], images_data[unique_id]['w'],
                                                           images_data[unique_id]['h'], cls_id)
                        images_data[unique_id]['labels'].append(yolo_bbox)

    total_imgs = len(images_data)
    print(f"TOTALE REALE IMMAGINI UNICHE: {total_imgs}")

    if total_imgs < 2000:
        print("ATTENZIONE: Ancora pochi dati? Controlla la logica.")

    # 3. Scrittura Dataset
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    all_keys = list(images_data.keys())
    train_keys, val_keys = train_test_split(
        all_keys, test_size=0.2, random_state=42)

    yolo_yaml = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(valid_folders),
        'names': valid_folders
    }
    with open('../data/linemod_multi.yaml', 'w') as f:
        yaml.dump(yolo_yaml, f)

    def write_split(keys, split_name):
        for key in tqdm(keys, desc=f"Scrivendo {split_name}"):
            data = images_data[key]

            ext = os.path.splitext(data['orig_name'])[1]
            new_filename = f"{key}{ext}"

            dst_img = os.path.join(OUTPUT_DIR, 'images',
                                   split_name, new_filename)
            shutil.copy(data['path'], dst_img)

            dst_txt = os.path.join(OUTPUT_DIR, 'labels',
                                   split_name, f"{key}.txt")
            with open(dst_txt, 'w') as f:
                for lbl in data['labels']:
                    f.write(
                        f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    write_split(train_keys, 'train')
    write_split(val_keys, 'val')
    print("Dataset generato correttamente (Nomi Unici)!")


if __name__ == "__main__":
    prepare_dataset()
