"""Streamlit demo app for building image classification."""
import json
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Building Classifier", page_icon="🏛️", layout="centered")

st.title("🏛️ Building Image Classifier")
st.markdown("Upload a building image to classify its type using a pretrained deep learning model.")

# Sidebar config
st.sidebar.header("Configuration")
config_path = st.sidebar.text_input("Config path", value="configs/default.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint path", value="outputs/checkpoints/best_model.pth")

@st.cache_resource
def load_predictor(ckpt_path, cfg_path):
    try:
        from src.inference.predict import BuildingPredictor
        from src.utils.config import load_config
        config = load_config(cfg_path)
        return BuildingPredictor(ckpt_path, config), None
    except Exception as e:
        return None, str(e)

uploaded_file = st.file_uploader("Upload a building image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    from PIL import Image
    import tempfile
    import os

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if not Path(checkpoint_path).exists():
        st.warning("No trained model checkpoint found. Train a model first:\n```\npython -m src.training.train --config configs/default.yaml\n```")
    else:
        with st.spinner("Classifying..."):
            predictor, error = load_predictor(checkpoint_path, config_path)
            if error:
                st.error(f"Failed to load model: {error}")
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    result = predictor.predict_image(tmp.name)
                    os.unlink(tmp.name)

                st.success(f"**Predicted Class:** `{result['predicted_class']}`")
                st.metric("Confidence", f"{result['confidence']:.1%}")

                st.subheader("All Class Probabilities")
                probs = result["all_probabilities"]
                sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
                for cls, prob in sorted_probs:
                    st.progress(prob, text=f"{cls}: {prob:.1%}")

st.markdown("---")
st.markdown("**[GitHub](https://github.com/vineel31/Classifying-Buildings-from-Images)** | Built with PyTorch + Streamlit")
