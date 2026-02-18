# t5_caption_demo.py
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="RASC T5 Caption Demo", layout="wide")
st.title("🖼️ RASC T5 Caption Generator")
st.write("Enter object relationships (scene graph) to generate a descriptive caption.")

# -----------------------------
# Load T5 model
# -----------------------------
@st.cache_resource
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

# Path to trained T5 model (update this!)
MODEL_PATH = "models/caption_generator/t5_scene"
tokenizer, model, device = load_model(MODEL_PATH)

# -----------------------------
# Input relationships
# -----------------------------
st.subheader("Input Scene Graph")
st.write("Enter relationships between objects, one per line, e.g.:")
st.code("obj0 left of obj1\nobj1 on top of obj2")

relationships_input = st.text_area("Relationships:", height=200)

# -----------------------------
# Generate caption
# -----------------------------
if st.button("Generate Caption"):
    if relationships_input.strip() == "":
        st.warning("Please enter at least one relationship.")
    else:
        # Process input
        relationships = [line.strip() for line in relationships_input.strip().split("\n") if line.strip()]
        graph_text = "; ".join(relationships)
        
        inputs = tokenizer(
            graph_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("Generated Caption:")
        st.write(f"📝 {caption}")

# -----------------------------
# Example Relationships
# -----------------------------
st.subheader("Example Relationships")
st.write(
    "```\nobj0 left of obj1\nobj1 on top of obj2\nobj2 next to obj3\n```"
)
