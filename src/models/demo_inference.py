import torch
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from models.relationship_predictor import create_model

# ------------------------------
# Relationship mapping
# ------------------------------
RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]
rel2idx = {r: i for i, r in enumerate(RELATIONS)}
idx2rel = {i: r for r, i in rel2idx.items()}

# ------------------------------
# Sample object detection output
# ------------------------------
# For real usage, integrate your object detector (e.g., YOLOv8)
# Here is a mock example
sample_objects = [
    {"class": 0, "bbox": [50, 50, 150, 200], "name": "person"},
    {"class": 1, "bbox": [160, 80, 250, 220], "name": "dog"},
    {"class": 2, "bbox": [200, 150, 300, 250], "name": "table"}
]

# ------------------------------
# Utility: Convert object info to tensors
# ------------------------------
def prepare_tensors(objects, device):
    obj_classes = torch.tensor([o["class"] for o in objects], dtype=torch.long, device=device)
    obj_boxes = torch.tensor([o["bbox"] for o in objects], dtype=torch.float, device=device)
    return obj_classes, obj_boxes

# ------------------------------
# Inference function
# ------------------------------
def predict_relationships(model, objects, device):
    model.eval()
    obj_classes, obj_boxes = prepare_tensors(objects, device)

    pairs = []
    preds = []

    # Generate all possible subject-object pairs (excluding self)
    for i, subj in enumerate(objects):
        for j, obj in enumerate(objects):
            if i == j:
                continue
            # Forward pass
            with torch.no_grad():
                logits = model(
                    obj_classes[i].unsqueeze(0),
                    obj_boxes[i].unsqueeze(0),
                    obj_classes[j].unsqueeze(0),
                    obj_boxes[j].unsqueeze(0)
                )
                pred_idx = logits.argmax(dim=1).item()
                relation = idx2rel[pred_idx]
                pairs.append((subj["name"], obj["name"]))
                preds.append(relation)

    return pairs, preds

# ------------------------------
# Visualization
# ------------------------------
def draw_relationships(image_path, objects, pairs, preds, output_path="output.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Draw bounding boxes
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1-10), obj["name"], fill="red", font=font)

    # Draw relationships
    for (subj, obj), rel in zip(pairs, preds):
        draw.text((10, 10 + 15 * list(preds).index(rel)), f"{subj} {rel} {obj}", fill="blue", font=font)

    img.save(output_path)
    print(f"[INFO] Saved visualization to {output_path}")

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained relationship model")
    parser.add_argument("--model-type", type=str, default="neural_motifs", help="mlp or neural_motifs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(args.model_type, num_classes=150, num_relations=10)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Predict relationships
    pairs, preds = predict_relationships(model, sample_objects, device)

    # Print results
    for (subj, obj), rel in zip(pairs, preds):
        print(f"{subj} → {rel} → {obj}")

    # Visualize
    draw_relationships(args.image, sample_objects, pairs, preds)

if __name__ == "__main__":
    main()
