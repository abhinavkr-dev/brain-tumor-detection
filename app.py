"""NeuroScan AI - brain MRI image analysis (educational research prototype).

Class mapping (from the training notebook): 0 = Brain Tumor, 1 = Healthy.
Preprocessing (unchanged): grayscale -> 150x150 -> 3 channels -> MobileNetV2 preprocess_input.
"""
from __future__ import annotations

import hashlib
import io
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logger = logging.getLogger("neuroscan")

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_DIR / "outputs" / "mobilenetv2_best.keras", APP_DIR / "mobilenetv2_best.keras"]
IMG_SIZE = 150
CLASS_NAMES = {0: "Brain Tumor", 1: "Healthy / No Tumor"}
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_MB = 10
MAX_PIXELS = 40_000_000
DISCLAIMER = "Educational research prototype. Not a medical device and not a diagnosis."

# Minimal CSS. Colours are set explicitly so text stays readable in Streamlit's light or dark theme.
st.markdown(
    """
    <style>
    .stApp{background:#fff}
    .block-container{max-width:1100px;padding-top:2rem;padding-bottom:2rem}
    #MainMenu,footer{visibility:hidden}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp li,.stApp label,
    .stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricLabel"],.stApp [data-testid="stMetricValue"]{color:#1f2937}
    .stApp [data-testid="stCaptionContainer"]{color:#4b5563}
    [data-testid="stFileUploaderDropzone"]{background:#f6f8fa;border:1px dashed #9ca3af}
    [data-testid="stFileUploaderDropzone"] *{color:#1f2937}
    [data-testid="stFileUploaderDropzone"] button{background:#fff;border:1px solid #9ca3af}
    .stButton>button[kind="secondary"]{background:#fff;border:1px solid #9ca3af}
    .stButton>button[kind="secondary"] p{color:#1f2937}
    .stButton>button[kind="primary"],.stDownloadButton>button{background:#0f766e;border:1px solid #0f766e}
    .stButton>button[kind="primary"] p,.stDownloadButton>button p{color:#fff}
    .stButton>button:disabled{opacity:.5}
    div[data-testid="stTabs"] button p{color:#1f2937}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ Model & preprocessing ------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, str | None]:
    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                return tf.keras.models.load_model(path), str(path)
            except Exception:
                logger.exception("Failed to load model at %s", path)
                return None, str(path)
    return None, None


def validate_uploaded_image(uploaded) -> tuple[Image.Image | None, dict[str, Any], str | None]:
    """Return (rgb_image, info, error_message). Never raises."""
    try:
        raw = uploaded.getvalue()
        if not raw:
            return None, {}, "The file is empty."
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            return None, {}, f"The file is larger than {MAX_FILE_MB} MB."
        Image.open(io.BytesIO(raw)).verify()
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_PIXELS:
            return None, {}, "The image is too large to process safely."
        info = {"filename": uploaded.name, "digest": hashlib.sha256(raw).hexdigest()}
        return ImageOps.exif_transpose(img).convert("RGB"), info, None
    except Exception:
        logger.exception("Image validation failed")
        return None, {}, "This file could not be read as an image. Please upload a valid JPG or PNG."


def preprocess_mri(image: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(image).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    rgb = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.expand_dims(preprocess_input(rgb), axis=0)


def predict_mri(model, image: Image.Image) -> dict[str, Any]:
    out = model(preprocess_mri(image), training=False)
    probs = np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=float)
    if probs.ndim != 2 or probs.shape != (1, 2):
        raise ValueError(f"Expected output shape (1, 2), got {probs.shape}.")
    probs = probs[0]
    cls = int(np.argmax(probs))
    return {
        "predicted_class": cls,
        "label": CLASS_NAMES[cls],
        "confidence": float(probs[cls] * 100),
        "tumor_score": float(probs[0] * 100),
        "healthy_score": float(probs[1] * 100),
    }


# ------------------------------ Grad-CAM ------------------------------
def _find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any("conv" in s.name.lower() for s in layer.layers):
            return layer
    return None


def generate_gradcam(model, image: Image.Image, class_index: int) -> dict[str, Image.Image] | None:
    """Grad-CAM on the last conv feature map of the MobileNetV2 backbone; None if it cannot be computed."""
    try:
        base = _find_backbone(model)
        if base is None:
            return None
        conv_layer = None
        for name in ("out_relu", "Conv_1_relu"):
            try:
                cand = base.get_layer(name)
                if len(cand.output.shape) == 4:
                    conv_layer = cand
                    break
            except (ValueError, AttributeError):
                continue
        if conv_layer is None:
            for layer in reversed(base.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) and len(layer.output.shape) == 4:
                    conv_layer = layer
                    break
        if conv_layer is None:
            return None

        feature_model = tf.keras.Model(base.inputs, [conv_layer.output, base.output])
        head_layers = model.layers[model.layers.index(base) + 1:]
        x_in = tf.convert_to_tensor(preprocess_mri(image))
        with tf.GradientTape() as tape:
            conv_feats, x = feature_model(x_in, training=False)
            tape.watch(conv_feats)
            for layer in head_layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x, training=False)
            score = x[:, int(class_index)]
        grads = tape.gradient(score, conv_feats)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.maximum(tf.reduce_sum(conv_feats[0] * weights, axis=-1), 0)
        cam = (cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())).numpy()
        if not np.isfinite(cam).all() or float(cam.max()) <= 0:
            return None

        h_img = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(image.size, Image.Resampling.BILINEAR)
        h = np.asarray(h_img, dtype=np.float32) / 255.0
        col = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        col[..., 0] = np.uint8(np.clip(255 * np.minimum(1.0, h * 2.2), 0, 255))
        col[..., 1] = np.uint8(np.clip(255 * np.maximum(0.0, (h - 0.22) * 1.6), 0, 255))
        col[..., 2] = np.uint8(np.clip(90 * np.maximum(0.0, h - 0.72), 0, 255))
        heat = Image.fromarray(col, mode="RGB")
        return {"heatmap": heat, "overlay": Image.blend(image.convert("RGB"), heat, alpha=0.42)}
    except Exception:
        logger.exception("Grad-CAM failed")
        return None


def run_case(model, image: Image.Image, info: dict[str, Any]) -> dict[str, Any]:
    result = predict_mri(model, image)
    return {
        "case_id": "NS-" + uuid.uuid4().hex[:8].upper(),
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        "image": image.copy(),
        "info": info,
        "result": result,
        "gradcam": generate_gradcam(model, image, result["predicted_class"]),
    }


# ------------------------------ PDF report ------------------------------
def _png_buffer(img: Image.Image, max_side: int = 700) -> io.BytesIO:
    copy = img.convert("RGB")
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    copy.save(buf, format="PNG")
    buf.seek(0)
    return buf


def create_pdf_report(case: dict[str, Any], model_name: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    ink = colors.HexColor("#1f2937")
    title = ParagraphStyle("t", parent=ss["Title"], textColor=ink, fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=ink, fontSize=12, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, textColor=ink)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=colors.HexColor("#4b5563"))

    r, info = case["result"], case["info"]
    story: list[Any] = [Paragraph("NeuroScan AI", title), Paragraph("Brain MRI image analysis report", small), Spacer(1, 8)]

    rows = [
        ("Case ID", case["case_id"]), ("Date / time", case["timestamp"]), ("File", info.get("filename", "")),
        ("Model", model_name), ("Predicted class", r["label"]), ("Model confidence", f"{r['confidence']:.2f}%"),
        ("Brain Tumor score", f"{r['tumor_score']:.2f}%"), ("Healthy score", f"{r['healthy_score']:.2f}%"),
    ]
    t = Table([[Paragraph(escape(a), body), Paragraph(escape(str(b)), body)] for a, b in rows], colWidths=[50 * mm, 120 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Paragraph("Result", h2), t]

    story += [Paragraph("About this output", h2), Paragraph(
        "The predicted class is the class with the higher model output. The confidence is the model's own score for that class; "
        "it is not necessarily a calibrated medical probability.", body)]

    def fit(img: Image.Image) -> RLImage:
        w, h = img.size
        s = min(80 * mm / w, 80 * mm / h)
        return RLImage(_png_buffer(img), width=w * s, height=h * s)

    story.append(Paragraph("Images", h2))
    g = case["gradcam"]
    cells = [[fit(case["image"]), fit(g["overlay"]) if g else Paragraph("Grad-CAM not available.", body)],
             [Paragraph("Original MRI", small), Paragraph("Grad-CAM overlay", small)]]
    story.append(Table(cells, colWidths=[85 * mm, 85 * mm]))
    story.append(Paragraph("Grad-CAM shows regions that influenced the model output. It does not segment a tumor or prove that one is present.", small))

    story += [Spacer(1, 10), Paragraph(escape(DISCLAIMER), small)]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title=f"NeuroScan report {case['case_id']}").build(story)
    return buf.getvalue()


# ------------------------------ UI helpers ------------------------------
def show_scores(r: dict[str, Any]) -> None:
    st.progress(min(max(r["tumor_score"] / 100, 0.0), 1.0), text=f"Brain Tumor: {r['tumor_score']:.2f}%")
    st.progress(min(max(r["healthy_score"] / 100, 0.0), 1.0), text=f"Healthy / No Tumor: {r['healthy_score']:.2f}%")


def show_results(case: dict[str, Any], model_name: str) -> None:
    r = case["result"]
    st.divider()
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", r["label"])
    c2.metric("Model confidence", f"{r['confidence']:.2f}%")
    show_scores(r)
    st.caption("Confidence is the model's own output score and is not necessarily a calibrated medical probability.")

    st.subheader("Grad-CAM explanation")
    g = case["gradcam"]
    if g is None:
        st.info("Grad-CAM could not be generated for this image. The prediction above is unaffected.")
    else:
        a, b, c = st.columns(3)
        a.image(case["image"], caption="Original MRI", use_container_width=True)
        b.image(g["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
        c.image(g["overlay"], caption="Overlay", use_container_width=True)
    st.caption("Grad-CAM highlights regions that influenced the model's output. It does not segment a tumor or prove that one is present.")

    try:
        st.download_button("Download report (PDF)", data=create_pdf_report(case, model_name),
                           file_name=f"NeuroScan_{case['case_id']}.pdf", mime="application/pdf")
    except ImportError:
        st.error("PDF export needs `reportlab`. Run: pip install reportlab")
    except Exception:
        logger.exception("PDF generation failed")
        st.error("The PDF report could not be generated.")


# ------------------------------ Page ------------------------------
model, model_path = load_model()
model_name = Path(model_path).name if model_path else "not found"

st.markdown("## NeuroScan AI")
st.caption("Brain MRI image analysis" + (f" · Model loaded: {model_name}" if model else ""))
st.caption(DISCLAIMER)

if model is None:
    if model_path:
        st.error(f"`{model_name}` was found but could not be loaded. It may be corrupted or built with an incompatible TensorFlow/Keras version.")
    else:
        st.error("Model not found. Place `mobilenetv2_best.keras` in the `outputs` folder, or next to `app.py`.")
    st.stop()

st.session_state.setdefault("single_nonce", 0)
st.session_state.setdefault("cmp_nonce", 0)

tab_single, tab_compare = st.tabs(["Analyze MRI", "Compare MRIs"])

with tab_single:
    st.subheader("Upload MRI")
    left, right = st.columns(2, gap="large")
    image, info = None, {}
    with left:
        uploaded = st.file_uploader("MRI image (JPG, JPEG or PNG)", type=ALLOWED_TYPES, key=f"single_{st.session_state['single_nonce']}")
        if uploaded is not None:
            image, info, err = validate_uploaded_image(uploaded)
            if err:
                st.error(err)
        b1, b2 = st.columns(2)
        analyze = b1.button("Analyze MRI", type="primary", disabled=image is None, use_container_width=True)
        if b2.button("Clear / New Case", use_container_width=True):
            st.session_state["single_nonce"] += 1
            st.session_state.pop("single_case", None)
            st.rerun()
    with right:
        if image is not None:
            st.image(image, use_container_width=True)
            st.caption(f"{info['filename']} · {image.width} × {image.height} px")
        else:
            st.caption("Select an MRI image to preview it here, then click Analyze MRI.")

    if analyze and image is not None:
        try:
            with st.spinner("Analyzing…"):
                st.session_state["single_case"] = run_case(model, image, info)
        except Exception:
            logger.exception("Analysis failed")
            st.session_state.pop("single_case", None)
            st.error("The model could not analyze this image. Please try a different file.")

    case = st.session_state.get("single_case")
    if case is not None and image is not None:
        if case["info"]["digest"] == info["digest"]:
            show_results(case, model_name)
        else:
            st.info("A different image is selected. Click Analyze MRI to analyze it.")

with tab_compare:
    st.subheader("Compare MRIs")
    st.caption("Each image is analyzed independently. A difference between predictions does not indicate disease progression or treatment response.")
    n = st.session_state["cmp_nonce"]
    loaded: dict[str, tuple[Image.Image, dict[str, Any]]] = {}
    cols = st.columns(2, gap="large")
    for col, tag, label in ((cols[0], "a", "First MRI"), (cols[1], "b", "Second MRI")):
        with col:
            f = st.file_uploader(label, type=ALLOWED_TYPES, key=f"cmp_{tag}_{n}")
            if f is not None:
                im, inf, e = validate_uploaded_image(f)
                if e:
                    st.error(f"{label}: {e}")
                else:
                    loaded[tag] = (im, inf)
                    st.image(im, use_container_width=True)
                    st.caption(f"{inf['filename']} · {im.width} × {im.height} px")
    ready = len(loaded) == 2
    b1, b2 = st.columns(2)
    compare = b1.button("Compare", type="primary", disabled=not ready, use_container_width=True)
    if b2.button("Clear", use_container_width=True):
        st.session_state["cmp_nonce"] += 1
        st.session_state.pop("cmp_case", None)
        st.rerun()

    if compare and ready:
        try:
            with st.spinner("Analyzing both images…"):
                st.session_state["cmp_case"] = {t: run_case(model, *loaded[t]) for t in ("a", "b")}
        except Exception:
            logger.exception("Comparison failed")
            st.session_state.pop("cmp_case", None)
            st.error("The images could not be compared.")

    cc = st.session_state.get("cmp_case")
    if cc is not None and ready and all(cc[t]["info"]["digest"] == loaded[t][1]["digest"] for t in ("a", "b")):
        st.divider()
        res_cols = st.columns(2, gap="large")
        for col, tag, title in zip(res_cols, ("a", "b"), ("First MRI", "Second MRI")):
            with col:
                c = cc[tag]
                st.markdown(f"**{title}**")
                st.metric("Predicted class", c["result"]["label"])
                show_scores(c["result"])
                if c["gradcam"] is not None:
                    st.image(c["gradcam"]["overlay"], caption="Grad-CAM overlay", use_container_width=True)
                else:
                    st.caption("Grad-CAM not available for this image.")