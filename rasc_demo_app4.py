"""
RASC Streamlit Demo - Enhanced Version
Detection → Relationship Prediction → Caption Generation
"""

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
from pathlib import Path

from src.models.model_mlp import RelationshipMLP

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="RASC Demo - Relationship-Aware Scene Captioning",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #028090;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background: linear-gradient(90deg, #028090 0%, #00A896 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #028090;
        margin: 1rem 0;
    }
    .caption-display {
        background: linear-gradient(135deg, #02C39A 0%, #028090 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        font-size: 1.8rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .relationship-card {
        background-color: #e8f4f5;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #00A896;
        font-size: 0.95rem;
    }
    .info-box {
        background-color: #e8f4f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #028090;
    }
    .stButton>button {
        background: linear-gradient(90deg, #028090 0%, #00A896 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00A896 0%, #02C39A 100%);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
# YOLO_WEIGHTS = "runs/detect/experiments/runs/yolo_experiment_1/weights/best.pt"
# YOLO_WEIGHTS = "models/yolo_vg2/weights/best.pt"
YOLO_WEIGHTS = "yolov8n.pt"
RELATIONSHIP_WEIGHTS = "models/relationship_predictor/rel_mlp.pt"
CAPTION_MODEL_PATH = "models/caption_generator/t5_scene"

RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('<div class="main-header">👁️ RASC Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Relationship-Aware Scene Captioning for Accessibility</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>How it works:</b><br>
1️⃣ <b>Object Detection</b> - YOLOv8 identifies objects in your image<br>
2️⃣ <b>Relationship Prediction</b> - MLP model understands spatial relationships<br>
3️⃣ <b>Caption Generation</b> - T5 creates natural language descriptions
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Visualization settings
    st.subheader("Visualization")
    bbox_color = st.color_picker("Bounding Box Color", value="#FF0000")
    bbox_width = st.slider("Box Width", 1, 5, 3)
    show_object_ids = st.checkbox("Show Object IDs", value=True)
    font_size = st.slider("Label Font Size", 10, 30, 16)
    
    # Inference settings
    st.subheader("Inference Settings")
    max_relationships_display = st.slider(
        "Max Relationships to Display", 
        10, 100, 50,
        help="Limit displayed relationships to avoid clutter"
    )
    
    confidence_threshold = st.slider(
        "Detection Confidence",
        0.1, 0.9, 0.25, 0.05,
        help="Minimum confidence for object detection"
    )
    
    # Info
    st.subheader("ℹ️ Model Info")
    st.info(f"""
    **Device:** {DEVICE}
    
    **Models:**
    - YOLO: Object Detection
    - MLP: Relationships
    - T5-small: Captions
    
    **Authors:**
    Jai Kushwaha & Caner Gel
    """)
    
    # Sample images
    st.subheader("📸 Sample Images")
    sample_dir = Path("sample_images")
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if samples:
            selected_sample = st.selectbox(
                "Choose a sample",
                ["None"] + [s.name for s in samples]
            )
        else:
            selected_sample = "None"
    else:
        selected_sample = "None"

# -------------------------------------------------------
# Load Models (cached)
# -------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all models with progress indication"""
    with st.spinner("🔄 Loading models... This may take a minute."):
        try:
            # YOLO
            yolo = YOLO(YOLO_WEIGHTS)
            
            # Relationship MLP
            rel_model = RelationshipMLP(num_classes=150, num_relations=10)
            rel_model.load_state_dict(
                torch.load(RELATIONSHIP_WEIGHTS, map_location=DEVICE)
            )
            rel_model.to(DEVICE)
            rel_model.eval()
            
            # T5 Caption Generator
            tokenizer = T5Tokenizer.from_pretrained(CAPTION_MODEL_PATH)
            t5 = T5ForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
            t5.to(DEVICE)
            t5.eval()
            
            return yolo, rel_model, tokenizer, t5, None
            
        except Exception as e:
            return None, None, None, None, str(e)

# Load models
yolo, rel_model, tokenizer, t5, error = load_models()

if error:
    st.error(f"❌ Error loading models: {error}")
    st.info("💡 Please check that all model files exist at the specified paths.")
    st.stop()
else:
    st.success("✅ Models loaded successfully!")

# -------------------------------------------------------
# Image Upload/Selection
# -------------------------------------------------------
st.header("📤 Upload Image")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to analyze"
    )

with col2:
    if selected_sample != "None":
        st.info(f"📁 Sample: {selected_sample}")

# Determine image source
image = None
image_name = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name
elif selected_sample != "None":
    sample_path = sample_dir / selected_sample
    image = Image.open(sample_path).convert("RGB")
    image_name = selected_sample
else:
    st.warning("⬆️ Please upload an image or select a sample to begin.")
    st.stop()

# Display input image
st.subheader("📷 Input Image")
st.image(image, caption=f"Source: {image_name}", use_container_width=True)

# -------------------------------------------------------
# Run Analysis Button
# -------------------------------------------------------
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_total = time.time()
    
    try:
        # =====================================================
        # STEP 1 — OBJECT DETECTION
        # =====================================================
        status_text.text("🔍 Step 1/3: Detecting objects...")
        progress_bar.progress(10)
        
        st.markdown('<div class="step-container"><h2>🎯 Step 1: Object Detection</h2></div>', unsafe_allow_html=True)
        
        start_time = time.time()
        results = yolo(image, conf=confidence_threshold)[0]
        detection_time = time.time() - start_time
        
        progress_bar.progress(33)
        
        # Process detections
        objects = []
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # Try to load better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for idx, (cls, box) in enumerate(zip(results.boxes.cls, results.boxes.xywh)):
            cls_id = int(cls.item())
            box = box / 640.0
            box = box.to(torch.float32)
            
            objects.append((cls_id, box))
            
            # Draw bounding box
            x, y, w, h = box.tolist()
            x0 = int(x * image.width)
            y0 = int(y * image.height)
            x1 = int(x0 + w * image.width)
            y1 = int(y0 + h * image.height)
            
            draw.rectangle([x0, y0, x1, y1], outline=bbox_color, width=bbox_width)
            
            if show_object_ids:
                label = f"obj{idx}"
                # Background for text
                text_bbox = draw.textbbox((x0, y0 - 25), label, font=font)
                draw.rectangle(text_bbox, fill=bbox_color)
                draw.text((x0, y0 - 25), label, fill="white", font=font)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><b>Objects Detected:</b><br><h2>{len(objects)}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><b>Detection Time:</b><br><h2>{detection_time:.2f}s</h2></div>', unsafe_allow_html=True)
        with col3:
            avg_time = (detection_time / len(objects)) * 1000 if objects else 0
            st.markdown(f'<div class="metric-box"><b>Avg per Object:</b><br><h2>{avg_time:.1f}ms</h2></div>', unsafe_allow_html=True)
        
        st.image(draw_img, caption="Detected Objects with Bounding Boxes", use_container_width=True)
        
        with st.expander("📋 Object Details"):
            for idx, (cls_id, box) in enumerate(objects):
                st.write(f"**Object {idx}:** Class ID = {cls_id}, BBox = {[f'{x:.3f}' for x in box.tolist()]}")
        
        if len(objects) < 2:
            st.warning("⚠️ Not enough objects for relationship prediction (need at least 2)")
            st.stop()
        
        # =====================================================
        # STEP 2 — RELATIONSHIP PREDICTION
        # =====================================================
        status_text.text("🔗 Step 2/3: Predicting relationships...")
        progress_bar.progress(66)
        
        st.markdown('<div class="step-container"><h2>🔗 Step 2: Relationship Prediction</h2></div>', unsafe_allow_html=True)
        
        start_time = time.time()
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
                    
                    relation_text = f"obj{i} {rel_label} obj{j}"
                    relations.append(relation_text)
                    scene_graph_parts.append(relation_text)
        
        relationship_time = time.time() - start_time
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-box"><b>Relationships Found:</b><br><h2>{len(relations)}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><b>Prediction Time:</b><br><h2>{relationship_time:.2f}s</h2></div>', unsafe_allow_html=True)
        
        # Display relationships
        st.write(f"**Showing top {min(len(relations), max_relationships_display)} relationships:**")
        
        # Display in columns
        num_cols = 2
        cols = st.columns(num_cols)
        
        for idx, rel in enumerate(relations[:max_relationships_display]):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f'<div class="relationship-card">🔗 {rel}</div>', unsafe_allow_html=True)
        
        # Full list in expander
        if len(relations) > max_relationships_display:
            with st.expander(f"📋 View All {len(relations)} Relationships"):
                for rel in relations:
                    st.write(f"• {rel}")
        
        # =====================================================
        # STEP 3 — CAPTION GENERATION
        # =====================================================
        status_text.text("✍️ Step 3/3: Generating caption...")
        progress_bar.progress(90)
        
        st.markdown('<div class="step-container"><h2>✍️ Step 3: Caption Generation</h2></div>', unsafe_allow_html=True)
        
        start_time = time.time()
        scene_graph_text = "; ".join(scene_graph_parts[:40])  # Limit to avoid too long input
        
        inputs = tokenizer(
            scene_graph_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        output = t5.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        
        caption_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display caption prominently
        st.markdown(f'<div class="caption-display">📝 "{caption}"</div>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><b>Caption Length:</b><br><h2>{len(caption.split())} words</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><b>Generation Time:</b><br><h2>{caption_time:.2f}s</h2></div>', unsafe_allow_html=True)
        with col3:
            total_time = time.time() - start_total
            st.markdown(f'<div class="metric-box"><b>Total Time:</b><br><h2>{total_time:.2f}s</h2></div>', unsafe_allow_html=True)
        
        # =====================================================
        # SUMMARY & EXPORT
        # =====================================================
        st.header("📊 Summary & Export")
        
        results_dict = {
            "image": image_name,
            "objects_detected": len(objects),
            "relationships_found": len(relations),
            "caption": caption,
            "timing": {
                "detection": f"{detection_time:.2f}s",
                "relationships": f"{relationship_time:.2f}s",
                "caption": f"{caption_time:.2f}s",
                "total": f"{total_time:.2f}s"
            },
            "relationships_list": relations[:20]  # Include top 20
        }
        
        with st.expander("📄 Export Results (JSON)"):
            import json
            st.json(results_dict)
            
            st.download_button(
                label="💾 Download Results",
                data=json.dumps(results_dict, indent=2),
                file_name=f"rasc_results_{image_name.replace('.', '_')}.json",
                mime="application/json"
            )
        
        st.success("✅ Analysis completed successfully!")
        
        # Time breakdown chart
        with st.expander("⏱️ Time Breakdown"):
            timing_data = {
                "Detection": detection_time,
                "Relationships": relationship_time,
                "Caption": caption_time
            }
            st.bar_chart(timing_data)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>RASC: Relationship-Aware Scene Captioning for Accessibility</b></p>
    <p>Jai Kushwaha & Caner Gel | Maschinelles Sehen SoSe 2024</p>
    <p>Pipeline: YOLOv8 → MLP Relationships → T5 Captions</p>
</div>
""", unsafe_allow_html=True)