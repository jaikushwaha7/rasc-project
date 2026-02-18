"""
RASC Streamlit Demo Application
Enhanced version with better UI, error handling, and visualization
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from models.inference import RASCInference
except ImportError:
    st.error("Could not import RASCInference. Make sure you're running from the project root.")
    st.stop()

# -----------------------------
# Page Configuration
# -----------------------------
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
    .step-header {
        background: linear-gradient(90deg, #028090 0%, #00A896 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #028090;
    }
    .caption-box {
        background: linear-gradient(135deg, #02C39A 0%, #028090 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        font-size: 1.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .relationship-item {
        background-color: #e8f4f5;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #00A896;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-header">👁️ RASC Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Relationship-Aware Scene Captioning for Accessibility</div>', unsafe_allow_html=True)

st.markdown("""
Upload an image to see our three-stage pipeline in action:
1. **Object Detection** - Identify objects in the scene
2. **Relationship Prediction** - Understand spatial relationships
3. **Caption Generation** - Create natural language descriptions
""")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")

# Model paths
with st.sidebar.expander("Model Paths", expanded=False):
    config_path = st.text_input(
        "Config Path",
        value="configs/config.yaml",
        help="Path to configuration file"
    )
    
    yolo_weights = st.text_input(
        "YOLO Weights",
        value="yolov8n.pt",
        help="Path to YOLO model weights"
    )
    
    relationship_weights = st.text_input(
        "Relationship Model",
        value="models/relationship_predictor/neural_motifs_best.pt",
        help="Path to relationship model weights"
    )
    
    caption_model = st.text_input(
        "Caption Model",
        value="models/caption_generator/t5_scene",
        help="Path to caption model directory"
    )

# Inference settings
with st.sidebar.expander("Inference Settings", expanded=True):
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence for object detection"
    )
    
    max_objects = st.slider(
        "Max Objects",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        help="Maximum number of objects to detect"
    )
    
    max_relationships = st.slider(
        "Max Relationships",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Maximum number of relationships to display"
    )

# Visualization settings
with st.sidebar.expander("Visualization", expanded=True):
    bbox_color = st.color_picker("Bounding Box Color", value="#FF0000")
    bbox_width = st.slider("Box Width", 1, 5, 2)
    show_labels = st.checkbox("Show Object Labels", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=False)

# Sample images
st.sidebar.header("📸 Sample Images")
sample_images_dir = Path("sample_images")
if sample_images_dir.exists():
    sample_images = list(sample_images_dir.glob("*.jpg")) + list(sample_images_dir.glob("*.png"))
    if sample_images:
        selected_sample = st.sidebar.selectbox(
            "Choose a sample image",
            ["None"] + [img.name for img in sample_images]
        )
    else:
        selected_sample = "None"
else:
    selected_sample = "None"
    st.sidebar.info("No sample images found. Create a 'sample_images' directory.")

# Info section
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**RASC** generates relationship-aware captions for accessibility.

**Pipeline:**
- YOLOv8 for object detection
- Neural Motifs for relationships
- T5 for caption generation

**Authors:** Jai Kushwaha & Caner Gel
""")

# -----------------------------
# Load Pipeline
# -----------------------------
@st.cache_resource
def load_pipeline(config_path, yolo_weights, relationship_weights, caption_model):
    """Load the RASC inference pipeline"""
    try:
        with st.spinner("Loading models... This may take a minute."):
            pipeline = RASCInference(
                config_path=config_path if Path(config_path).exists() else None,
                yolo_weights=yolo_weights,
                relationship_weights=relationship_weights if Path(relationship_weights).exists() else None,
                caption_model=caption_model if Path(caption_model).exists() else None
            )
        return pipeline, None
    except Exception as e:
        return None, str(e)

# Load with error handling
pipeline, error = load_pipeline(config_path, yolo_weights, relationship_weights, caption_model)

if error:
    st.error(f"❌ Error loading pipeline: {error}")
    st.info("💡 Make sure all model paths are correct and models are trained.")
    st.stop()
else:
    st.success("✅ Pipeline loaded successfully!")

# -----------------------------
# Image Upload/Selection
# -----------------------------
st.header("📤 Upload Image")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to analyze"
    )

with col2:
    st.write("**Or use a sample image:**")
    if selected_sample != "None":
        st.info(f"Selected: {selected_sample}")

# Determine which image to use
image = None
image_source = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_source = uploaded_file.name
elif selected_sample != "None":
    sample_path = sample_images_dir / selected_sample
    image = Image.open(sample_path).convert("RGB")
    image_source = selected_sample
else:
    st.warning("⬆️ Please upload an image or select a sample image to begin.")
    st.stop()

# Display input image
st.subheader("Input Image")
st.image(image, caption=f"Source: {image_source}", use_container_width=True)

# -----------------------------
# Run Inference Button
# -----------------------------
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save uploaded image temporarily
    temp_image_path = Path("temp_input.jpg")
    image.save(temp_image_path)
    
    try:
        # -----------------------------
        # Step 1: Object Detection
        # -----------------------------
        status_text.text("🔍 Step 1/3: Detecting objects...")
        progress_bar.progress(10)
        
        start_time = time.time()
        objects = pipeline.detect_objects(str(temp_image_path))
        detection_time = time.time() - start_time
        
        progress_bar.progress(33)
        
        st.markdown('<div class="step-header"><h2>🎯 Step 1: Object Detection</h2></div>', unsafe_allow_html=True)
        
        if not objects:
            st.warning("⚠️ No objects detected!")
            st.stop()
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><b>Objects Detected:</b> {len(objects)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><b>Detection Time:</b> {detection_time:.2f}s</div>', unsafe_allow_html=True)
        with col3:
            avg_time = (detection_time / len(objects)) * 1000
            st.markdown(f'<div class="metric-card"><b>Avg per Object:</b> {avg_time:.1f}ms</div>', unsafe_allow_html=True)
        
        # Object list
        with st.expander("📋 Detected Objects Details", expanded=False):
            for i, (cls_id, bbox) in enumerate(objects):
                st.write(f"**Object {i}:** Class ID = {cls_id}, BBox = {[f'{x:.3f}' for x in bbox.tolist()]}")
        
        # Visualize detections
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # Try to load a better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        img_width, img_height = draw_img.size
        
        for i, (cls_id, bbox) in enumerate(objects):
            x, y, w, h = bbox.tolist()
            x0 = int(x * img_width)
            y0 = int(y * img_height)
            x1 = int(x0 + w * img_width)
            y1 = int(y0 + h * img_height)
            
            # Draw rectangle
            draw.rectangle([x0, y0, x1, y1], outline=bbox_color, width=bbox_width)
            
            # Draw label
            if show_labels:
                label = f"obj{i}"
                if show_confidence:
                    label += f" ({cls_id})"
                
                # Background for text
                text_bbox = draw.textbbox((x0, y0 - 20), label, font=font)
                draw.rectangle(text_bbox, fill=bbox_color)
                draw.text((x0, y0 - 20), label, fill="white", font=font)
        
        st.image(draw_img, caption="Detected Objects with Bounding Boxes", use_container_width=True)
        
        # -----------------------------
        # Step 2: Relationship Prediction
        # -----------------------------
        status_text.text("🔗 Step 2/3: Predicting relationships...")
        progress_bar.progress(66)
        
        start_time = time.time()
        relationships = pipeline.predict_relationships(objects)
        relationship_time = time.time() - start_time
        
        st.markdown('<div class="step-header"><h2>🔗 Step 2: Relationship Prediction</h2></div>', unsafe_allow_html=True)
        
        if not relationships:
            st.warning("⚠️ No relationships predicted!")
        else:
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><b>Relationships Found:</b> {len(relationships)}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><b>Prediction Time:</b> {relationship_time:.2f}s</div>', unsafe_allow_html=True)
            
            # Display relationships
            st.write(f"**Top {min(len(relationships), max_relationships)} Relationships:**")
            
            # Show in columns for better layout
            num_cols = 2
            cols = st.columns(num_cols)
            
            for idx, rel in enumerate(relationships[:max_relationships]):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.markdown(f'<div class="relationship-item">🔗 {rel}</div>', unsafe_allow_html=True)
            
            # Full list in expander
            if len(relationships) > max_relationships:
                with st.expander(f"📋 View All {len(relationships)} Relationships"):
                    for rel in relationships:
                        st.write(f"• {rel}")
        
        # -----------------------------
        # Step 3: Caption Generation
        # -----------------------------
        status_text.text("✍️ Step 3/3: Generating caption...")
        progress_bar.progress(90)
        
        start_time = time.time()
        caption = pipeline.generate_caption(relationships)
        caption_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.markdown('<div class="step-header"><h2>✍️ Step 3: Caption Generation</h2></div>', unsafe_allow_html=True)
        
        # Display caption prominently
        st.markdown(f'<div class="caption-box">📝 {caption}</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><b>Caption Length:</b> {len(caption.split())} words</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><b>Generation Time:</b> {caption_time:.2f}s</div>', unsafe_allow_html=True)
        with col3:
            total_time = detection_time + relationship_time + caption_time
            st.markdown(f'<div class="metric-card"><b>Total Time:</b> {total_time:.2f}s</div>', unsafe_allow_html=True)
        
        # -----------------------------
        # Summary & Export
        # -----------------------------
        st.header("📊 Summary")
        
        # Create summary dictionary
        results = {
            "image": image_source,
            "objects_detected": len(objects),
            "relationships_found": len(relationships),
            "caption": caption,
            "timing": {
                "detection": f"{detection_time:.2f}s",
                "relationships": f"{relationship_time:.2f}s",
                "caption": f"{caption_time:.2f}s",
                "total": f"{total_time:.2f}s"
            }
        }
        
        # Display as JSON
        with st.expander("📄 Export Results (JSON)", expanded=False):
            st.json(results)
            
            # Download button
            st.download_button(
                label="💾 Download Results",
                data=json.dumps(results, indent=2),
                file_name=f"rasc_results_{image_source.replace('.', '_')}.json",
                mime="application/json"
            )
        
        # Success message
        st.success("✅ Analysis completed successfully!")
        
        # Cleanup
        if temp_image_path.exists():
            temp_image_path.unlink()
    
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        st.exception(e)
        
        # Cleanup
        if temp_image_path.exists():
            temp_image_path.unlink()

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    <p><strong>RASC: Relationship-Aware Scene Captioning for Accessibility</strong></p>
    <p>Jai Kushwaha & Caner Gel | Maschinelles Sehen SoSe 2024</p>
    <p>Three-stage pipeline: YOLOv8 → Neural Motifs → T5</p>
</div>
""", unsafe_allow_html=True)