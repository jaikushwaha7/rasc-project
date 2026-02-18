"""
RASC Streamlit Demo
Detection → Relationship Prediction → Caption Generation
"""

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.models.model_mlp import RelationshipMLP

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
CONFIG_PATH = "configs/config.yaml"
YOLO_WEIGHTS = "models/yolo_vg2/weights/best.pt"
# YOLO_WEIGHTS = "runs/detect/experiments/runs/yolo_experiment_1/weights/best.pt"
RELATIONSHIP_WEIGHTS = "models/relationship_predictor/rel_mlp.pt"
CAPTION_MODEL_PATH = "models/caption_generator/t5_scene"

RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Streamlit UI Setup
# -------------------------------------------------------
st.set_page_config(page_title="RASC Demo", layout="wide")
st.title("RASC: Relationship-Aware Scene Captioning")

st.markdown("Upload an image → Detect objects → Predict relationships → Generate caption")

# -------------------------------------------------------
# Load Models (cached)
# -------------------------------------------------------
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_WEIGHTS)

    rel_model = RelationshipMLP(num_classes=150, num_relations=10)
    rel_model.load_state_dict(torch.load(RELATIONSHIP_WEIGHTS, map_location=DEVICE))
    rel_model.to(DEVICE)
    rel_model.eval()

    tokenizer = T5Tokenizer.from_pretrained(CAPTION_MODEL_PATH)
    t5 = T5ForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
    t5.to(DEVICE)
    t5.eval()

    return yolo, rel_model, tokenizer, t5


yolo, rel_model, tokenizer, t5 = load_models()

# -------------------------------------------------------
# Upload Image
# -------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # =====================================================
    # STEP 1 — OBJECT DETECTION
    # =====================================================
    st.subheader("Step 1: Object Detection")

    results = yolo(image)[0]

    objects = []
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    font = ImageFont.load_default()

    for idx, (cls, box) in enumerate(zip(results.boxes.cls, results.boxes.xywh)):
        cls_id = int(cls.item())
        box = box / 640.0
        box = box.to(torch.float32)

        objects.append((cls_id, box))

        # Draw bounding box
        x, y, w, h = box.tolist()
        x0 = x * image.width
        y0 = y * image.height
        x1 = x0 + w * image.width
        y1 = y0 + h * image.height

        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"ID:{idx}", fill="yellow", font=font)

    st.image(draw_img, caption="Detected Objects", use_column_width=True)
    st.write(f"Detected {len(objects)} objects")

    if len(objects) < 2:
        st.warning("Not enough objects for relationship prediction")
        st.stop()

    # =====================================================
    # STEP 2 — RELATIONSHIP PREDICTION
    # =====================================================
    st.subheader("Step 2: Relationship Prediction")

    relations = []
    scene_graph_parts = []

    with torch.no_grad():
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j:
                    continue

                s_cls, s_box = objects[i]
                o_cls, o_box = objects[j]

                logits = rel_model(
                    torch.tensor([s_cls]).to(DEVICE),
                    s_box.unsqueeze(0).to(DEVICE),
                    torch.tensor([o_cls]).to(DEVICE),
                    o_box.unsqueeze(0).to(DEVICE),
                )

                rel_idx = logits.argmax(dim=1).item()
                rel_label = RELATIONS[rel_idx]

                relation_text = f"Object {i} {rel_label} Object {j}"
                relations.append(relation_text)
                scene_graph_parts.append(relation_text)

    for r in relations:
        st.write(r)

    # =====================================================
    # STEP 3 — CAPTION GENERATION
    # =====================================================
    st.subheader("Step 3: Caption Generation")

    scene_graph_text = "; ".join(scene_graph_parts)

    inputs = tokenizer(
        scene_graph_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    output = t5.generate(**inputs, max_length=64)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)

    st.success(f"Generated Caption: {caption}")
