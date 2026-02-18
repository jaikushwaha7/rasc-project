import json
import os
from PIL import Image
from tqdm import tqdm

VG_IMAGES = "data/raw/visual_genome/VG_100K"
INPUT_FILE = "data/processed/vg_5k_subset.json"
LABEL_MAP = "data/processed/label_map.json"
OUTPUT_FILE = "data/processed/relationships/relationship_pairs2.json"

RELATIONS = {
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
}


def normalize_bbox(obj, img_w, img_h):
    xc = (obj["x"] + obj["w"] / 2) / img_w
    yc = (obj["y"] + obj["h"] / 2) / img_h
    w = obj["w"] / img_w
    h = obj["h"] / img_h
    return [xc, yc, w, h]


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    data = json.load(open(INPUT_FILE))
    label_map = json.load(open(LABEL_MAP))

    pairs = []

    for item in tqdm(data):
        image_id = item["image_id"]
        img_path = os.path.join(VG_IMAGES, f"{image_id}.jpg")

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        W, H = img.size

        # map object_id → object
        obj_map = {o["object_id"]: o for o in item["objects"]}

        for rel in item["relationships"]:
            pred = rel["predicate"].lower().strip()
            if pred not in RELATIONS:
                continue

            subj = obj_map.get(rel["subject"]["object_id"])
            obj = obj_map.get(rel["object"]["object_id"])
            if subj is None or obj is None:
                continue

            subj_name = subj["names"][0].lower()
            obj_name = obj["names"][0].lower()

            if subj_name not in label_map or obj_name not in label_map:
                continue

            pairs.append({
                "image_id": image_id,
                "subj_class": label_map[subj_name],
                "subj_bbox": normalize_bbox(subj, W, H),
                "obj_class": label_map[obj_name],
                "obj_bbox": normalize_bbox(obj, W, H),
                "predicate": pred
            })

    json.dump(pairs, open(OUTPUT_FILE, "w"))
    print(f"Saved {len(pairs)} relationship pairs")


if __name__ == "__main__":
    main()
