# rasc_streamlit_demo_fixed.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

from src.models.inference import RASCInference  # Your inference pipeline class

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="RASC Demo", layout="wide")
st.title("RASC: Relationship-Aware Scene Captioning")
st.write("End-to-end demo: detect objects → predict relationships → generate caption")

# -----------------------------
# Configuration
# -----------------------------
CONFIG_PATH = "configs/config.yaml"
IMAGE_PATH = "data/test/61.jpg"  # Default image
YOLO_WEIGHTS = "runs/detect/experiments/runs/yolo_experiment_1/weights/best.pt"  # or yolov8n.pt
RELATIONSHIP_WEIGHTS = "models/relationship_predictor/neural_motifs_best.pt"
CAPTION_MODEL = "models/caption_generator/t5_scene"

# -----------------------------
# Load pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return RASCInference(
        config_path=CONFIG_PATH,
        yolo_weights=YOLO_WEIGHTS,
        relationship_weights=RELATIONSHIP_WEIGHTS,
        caption_model=CAPTION_MODEL
    )

pipeline = load_pipeline()

# -----------------------------
# Upload Image or Use Default
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
else:
    image = Image.open(IMAGE_PATH).convert("RGB")
st.image(image, caption="Input Image", use_column_width=True)

# -----------------------------
# Step 1: Object Detection
# -----------------------------
st.subheader("Step 1: Object Detection")
objects = pipeline.detect_objects(image)

if not objects:
    st.warning("No objects detected!")
else:
    st.write(f"Detected {len(objects)} objects:")
    for i, (cls_id, bbox) in enumerate(objects):
        st.write(f"Object {i}: Class={cls_id}, BBox={bbox.tolist()}")

    # Draw bounding boxes
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    font = ImageFont.load_default()
    for i, (cls_id, bbox) in enumerate(objects):
        x, y, w, h = bbox.tolist()
        x0, y0 = x * draw_img.width, y * draw_img.height
        x1, y1 = x0 + w * draw_img.width, y0 + h * draw_img.height
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"ID:{i}", fill="yellow", font=font)
    st.image(draw_img, caption="Detected Objects", use_column_width=True)

# -----------------------------
# Step 2: Relationship Prediction
# -----------------------------
st.subheader("Step 2: Relationship Prediction")
if len(objects) < 2:
    st.warning("Not enough objects for relationship prediction!")
    relationships = []
else:
    # Fix for Neural Motifs shape error: ensure boxes are 2D
    processed_objects = []
    for cls_id, bbox in objects:
        bbox = bbox.squeeze() if len(bbox.shape) > 1 else bbox  # remove extra dims
        processed_objects.append((cls_id, bbox))
    relationships = pipeline.predict_relationships(processed_objects)

if len(relationships) == 0:
    st.warning("No relationships predicted!")
else:
    st.write(f"Predicted Relationships ({len(relationships)}):")
    for rel in relationships:
        st.write(rel)    

# if not relationships:
#     st.warning("No relationships predicted!")
# else:
#     st.write(f"Predicted Relationships ({len(relationships)}):")
#     for rel in relationships:
#         st.write(f"🔗 {rel}")

# -----------------------------
# Step 3: Caption Generation
# -----------------------------
st.subheader("Step 3: Caption Generation")
caption = pipeline.generate_caption(relationships)
st.success(f"📝 Generated Caption: {caption}")
