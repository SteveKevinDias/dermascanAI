import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.model import (
    DEVICE, CLASS_INFO, DEFAULT_CLASS_NAMES,
    load_unet, predict_mask, crop_with_margin,
    load_classifiers, run_ensemble, load_class_names, load_ensemble_config,
    compute_gradcam, sync_artifacts_from_hub,
)

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="DermaScan AI · Lesion Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_SEG = Path("artifacts/segmentation")
ARTIFACTS_CLS = Path("artifacts/classification")

# ============================================================================
# Styling — dark, clinical, "classy" theme
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(90,140,140,0.10), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(140,110,200,0.08), transparent 40%),
        #0b0e11;
    color: #e8e6e1;
}

section[data-testid="stSidebar"] {
    background: #0f1316;
    border-right: 1px solid rgba(255,255,255,0.06);
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.01em;
}

.hero {
    padding: 2.2rem 2.6rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(31,58,58,0.55), rgba(20,24,30,0.4));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-size: 2.4rem;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(90deg, #f4efe6, #b9d3cf 60%, #9fb8c9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p {
    color: #a9b3ae;
    font-size: 1.02rem;
    margin: 0;
    max-width: 620px;
}
.badge-row { margin-top: 1rem; display:flex; gap:0.5rem; flex-wrap: wrap;}
.pill {
    display:inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.12);
    color: #cfd6d2;
    background: rgba(255,255,255,0.03);
}

.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
}
.card h3 { margin-top: 0; font-size: 1.15rem; }

.metric-box {
    text-align:center;
    padding: 1rem 0.5rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; font-family:'Fraunces', serif;}
.metric-box .lbl { font-size: 0.78rem; color:#9aa39d; text-transform:uppercase; letter-spacing:0.05em;}

.risk-high { color: #f28b82 !important; }
.risk-medium { color: #f4c069 !important; }
.risk-low { color: #8fd3b6 !important; }

.verdict-banner {
    border-radius: 16px;
    padding: 1.4rem 1.7rem;
    margin: 0.6rem 0 1.2rem 0;
    border: 1px solid rgba(255,255,255,0.1);
}
.verdict-high { background: linear-gradient(135deg, rgba(160,50,50,0.28), rgba(60,20,20,0.15)); }
.verdict-medium { background: linear-gradient(135deg, rgba(160,120,30,0.25), rgba(60,45,10,0.15)); }
.verdict-low { background: linear-gradient(135deg, rgba(30,110,90,0.28), rgba(15,45,38,0.15)); }

.footnote { color:#7c847e; font-size:0.82rem; line-height:1.5; }

div[data-testid="stFileUploader"] {
    border-radius: 14px;
}

.stButton>button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
    background: linear-gradient(135deg, #21403c, #1a2b30);
    color: #eef3f0;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}
.stButton>button:hover {
    border-color: #8fd3b6;
    color: #8fd3b6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Hero
# ============================================================================
st.markdown("""
<div class="hero">
  <h1>DermaScan AI</h1>
  <p>U‑Net lesion segmentation paired with a calibrated three‑model ensemble
  (EfficientNet‑B3 · ConvNeXt‑Small · EfficientNetV2‑M) for research‑grade
  dermoscopic image triage — or pick a single backbone yourself.
  Includes Grad‑CAM explainability.</p>
  <div class="badge-row">
    <span class="pill">🧠 Segmentation → Classification pipeline</span>
    <span class="pill">🔬 8-class ISIC taxonomy</span>
    <span class="pill">🎛️ Ensemble or single-model mode</span>
    <span class="pill">🔎 Grad-CAM explainability</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="footnote">⚠️ Research / educational demo only. This tool does not '
    'provide medical diagnoses. Always consult a qualified dermatologist for any '
    'skin concern.</p>',
    unsafe_allow_html=True,
)

# ============================================================================
# Sidebar — settings & artifact status
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    # --- Hugging Face Hub sync ---
    with st.expander("📥 Load models from Hugging Face", expanded=False):
        try:
            default_repo = st.secrets.get("HF_REPO_ID", "")
        except Exception:
            default_repo = ""
        hf_repo_id = st.text_input(
            "Repo ID", value=default_repo,
            placeholder="your-username/skin-cancer-models",
            help="A Hugging Face model repo containing best_unet.pt, "
                 "best_efficientnet_b3.pt, best_convnext_small.pt, "
                 "best_efficientnetv2_m.pt, class_names.json, ensemble_config.json",
        )
        hf_token_input = st.text_input(
            "HF token (only needed for private repos)", value="", type="password",
        )
        sync_clicked = st.button("⬇️ Download / sync checkpoints", use_container_width=True)
        if sync_clicked:
            if not hf_repo_id.strip():
                st.warning("Enter a repo ID first, e.g. `your-username/skin-cancer-models`.")
            else:
                token = hf_token_input.strip() or None
                if not token:
                    try:
                        token = st.secrets.get("HF_TOKEN", None)
                    except Exception:
                        token = None
                with st.spinner(f"Fetching checkpoints from `{hf_repo_id}`..."):
                    try:
                        found = sync_artifacts_from_hub(
                            hf_repo_id.strip(), ARTIFACTS_SEG, ARTIFACTS_CLS, hf_token=token
                        )
                        if found:
                            st.success(f"Synced {len(found)} file(s): {', '.join(found)}")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error(
                                "No matching files found in that repo. Check the repo ID "
                                "and that filenames match exactly (best_unet.pt, "
                                "best_efficientnet_b3.pt, etc.)."
                            )
                    except Exception as e:
                        st.error(f"Could not sync from Hugging Face: {e}")

    cls_names_present = []
    for n in ["efficientnet_b3", "convnext_small", "efficientnetv2_m"]:
        p = ARTIFACTS_CLS / f"best_{n}.pt"
        if p.exists():
            cls_names_present.append(n)

    model_choice_options = ["Ensemble (all models)"] + cls_names_present
    model_labels = {
        "Ensemble (all models)": "Ensemble (all models)",
        "efficientnet_b3": "efficientnet_b3",
        "convnext_small": "convnext_small",
        "efficientnetv2_m": "efficientnetv2_m ⭐ best overall",
    }
    model_choice_display = st.selectbox(
        "Classifier",
        options=[model_labels.get(o, o) for o in model_choice_options],
        index=0,
        help="efficientnetv2_m outperformed both the other individual backbones "
             "and the full ensemble on validation — recommended default.",
    )
    reverse_labels = {v: k for k, v in model_labels.items()}
    model_choice = reverse_labels.get(model_choice_display, model_choice_display)
    use_ensemble = model_choice == "Ensemble (all models)"
    selected_single_model = None if use_ensemble else model_choice

    tta_enabled = st.toggle("Test-Time Augmentation (TTA)", value=True,
                             help="Averages predictions over flips/rotations for more robust results.")
    show_gradcam = st.toggle("Grad-CAM explanation", value=True)
    crop_margin = st.slider("Segmentation crop margin", 0.0, 0.4, 0.15, 0.05)
    st.markdown("---")
    st.markdown("### 📦 Model artifacts")

    seg_ckpt = ARTIFACTS_SEG / "best_unet.pt"
    st.write(f"{'✅' if seg_ckpt.exists() else '❌'} U-Net segmentation")

    for n in ["efficientnet_b3", "convnext_small", "efficientnetv2_m"]:
        p = ARTIFACTS_CLS / f"best_{n}.pt"
        suffix = "  ⭐ *best overall (beats ensemble)*" if n == "efficientnetv2_m" else ""
        st.write(f"{'✅' if p.exists() else '❌'} {n}{suffix}")

    if not seg_ckpt.exists() or not cls_names_present:
        st.info(
            "Place your trained checkpoints here:\n\n"
            "`artifacts/segmentation/best_unet.pt`\n\n"
            "`artifacts/classification/best_efficientnet_b3.pt`\n\n"
            "`artifacts/classification/best_convnext_small.pt`\n\n"
            "`artifacts/classification/best_efficientnetv2_m.pt`\n\n"
            "plus `class_names.json` and `ensemble_config.json` "
            "(exported by the notebooks) in `artifacts/classification/`.",
            icon="📁",
        )
    st.caption(f"Device: `{DEVICE}`")


# ============================================================================
# Cached model loading
# ============================================================================
@st.cache_resource(show_spinner=False)
def _load_unet_cached():
    return load_unet(str(seg_ckpt), device=DEVICE)


@st.cache_resource(show_spinner=False)
def _load_classifiers_cached(num_classes):
    return load_classifiers(str(ARTIFACTS_CLS), num_classes, device=DEVICE)


@st.cache_data(show_spinner=False)
def _load_class_meta():
    return load_class_names(str(ARTIFACTS_CLS))


models_ready = seg_ckpt.exists() and len(cls_names_present) > 0

# ============================================================================
# Upload
# ============================================================================
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="card"><h3>1 · Upload a lesion image</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Dermoscopic or close-up skin photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    demo = st.checkbox("Use a synthetic demo image instead", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    image_rgb = None
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        image_rgb = np.array(pil_img)
    elif demo:
        rng = np.random.default_rng(0)
        canvas = (rng.normal(190, 12, (400, 400, 3))).clip(0, 255).astype(np.uint8)
        cv2.circle(canvas, (200, 200), 70, (90, 55, 45), -1)
        cv2.circle(canvas, (200, 200), 70, (60, 35, 30), 6)
        image_rgb = canvas

    if image_rgb is not None:
        st.image(image_rgb, caption="Input image")

with right:
    st.markdown('<div class="card"><h3>2 · Run analysis</h3>', unsafe_allow_html=True)
    run = st.button("🔬 Analyze lesion", type="primary", use_container_width=True,
                     disabled=(image_rgb is None or not models_ready))
    if image_rgb is None:
        st.caption("Upload an image (or check the demo box) to enable analysis.")
    elif not models_ready:
        st.caption("Model checkpoints are missing — see sidebar for setup instructions.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# Analysis
# ============================================================================
if run and image_rgb is not None and models_ready:
    class_names, mel_idx = _load_class_meta()

    with st.spinner("Loading models..."):
        unet_model = _load_unet_cached()
        classifiers = _load_classifiers_cached(len(class_names))
        ensemble_weights, cfg_tta = load_ensemble_config(str(ARTIFACTS_CLS), list(classifiers.keys()))

    t0 = time.time()
    with st.spinner("Segmenting lesion with U-Net..."):
        mask, prob_map = predict_mask(unet_model, image_rgb, device=DEVICE)
        cropped, bbox = crop_with_margin(image_rgb, mask, margin_frac=crop_margin)

    with st.spinner("Running classification..."):
        if use_ensemble:
            combined_probs, per_model_probs = run_ensemble(
                classifiers, ensemble_weights, cropped, tta_enabled=tta_enabled, device=DEVICE
            )
            active_weights = ensemble_weights
        else:
            from utils.model import predict_probs_single
            info_m = classifiers[selected_single_model]
            single_probs = predict_probs_single(
                info_m["model"], cropped, info_m["img_size"], tta_enabled=tta_enabled, device=DEVICE
            )
            combined_probs = single_probs
            per_model_probs = {selected_single_model: single_probs}
            active_weights = {selected_single_model: 1.0}
    elapsed = time.time() - t0

    pred_idx = int(np.argmax(combined_probs))
    pred_code = class_names[pred_idx]
    pred_conf = float(combined_probs[pred_idx])
    info = CLASS_INFO.get(pred_code, {"name": pred_code, "risk": "medium"})

    st.markdown("---")
    st.markdown("## 🧾 Results")

    # --- Verdict banner ---
    risk = info["risk"]
    risk_class = {"high": "verdict-high", "medium": "verdict-medium", "low": "verdict-low"}[risk]
    risk_word = {"high": "Elevated priority", "medium": "Moderate priority", "low": "Lower priority"}[risk]
    st.markdown(f"""
    <div class="verdict-banner {risk_class}">
      <div style="font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; color:#c8cec9;">{'Ensemble prediction' if use_ensemble else f'{selected_single_model} prediction'}</div>
      <div style="font-size:1.7rem; font-weight:700; font-family:'Fraunces',serif; margin-top:0.2rem;">
        {info['name']} <span style="font-size:1rem; color:#c8cec9; font-weight:400;">({pred_code})</span>
      </div>
      <div style="margin-top:0.3rem;" class="risk-{risk}">● {risk_word} — confidence {pred_conf*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Metrics row ---
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in zip(
        [m1, m2, m3, m4],
        [f"{pred_conf*100:.1f}%", f"{len(per_model_probs)}" if use_ensemble else "1 (single)",
         "✓" if tta_enabled else "–", f"{elapsed:.1f}s"],
        ["Confidence", "Models used", "TTA applied", "Inference time"],
    ):
        col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                      unsafe_allow_html=True)

    st.write("")

    # --- Segmentation visuals ---
    st.markdown('<div class="card"><h3>🩹 Segmentation</h3>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    overlay = image_rgb.copy()
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[mask > 0] = (30, 200, 160)
    blended = cv2.addWeighted(overlay, 0.75, overlay_mask, 0.45, 0)
    s1.image(image_rgb, caption="Original")
    s2.image(blended, caption="U-Net lesion mask overlay")
    s3.image(cropped, caption="Cropped for classification")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Probability chart ---
    st.markdown(f'<div class="card"><h3>📊 Class probabilities ({"ensemble" if use_ensemble else selected_single_model})</h3>', unsafe_allow_html=True)
    order = np.argsort(combined_probs)[::-1]
    sorted_codes = [class_names[i] for i in order]
    sorted_names = [CLASS_INFO.get(c, {"name": c})["name"] for c in sorted_codes]
    sorted_probs = [combined_probs[i] * 100 for i in order]
    colors = ["#f28b82" if CLASS_INFO.get(c, {}).get("risk") == "high"
              else "#f4c069" if CLASS_INFO.get(c, {}).get("risk") == "medium"
              else "#8fd3b6" for c in sorted_codes]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=[f"{n} ({c})" for n, c in zip(sorted_names, sorted_codes)],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in sorted_probs],
        textposition="outside",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e6e1", family="Inter"),
        xaxis=dict(title="Probability (%)", range=[0, max(sorted_probs) * 1.25], gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Per-model breakdown ---
    if use_ensemble and len(per_model_probs) > 1:
        with st.expander("🔍 Per-model breakdown & ensemble weights"):
            cols = st.columns(len(per_model_probs))
            for col, (name, probs) in zip(cols, per_model_probs.items()):
                top_i = int(np.argmax(probs))
                col.markdown(f"**{name}**")
                col.metric(class_names[top_i], f"{probs[top_i]*100:.1f}%")
                col.caption(f"ensemble weight: {active_weights.get(name, 0):.3f}")

    # --- Grad-CAM ---
    if show_gradcam:
        st.markdown('<div class="card"><h3>🔎 Grad-CAM — model attention</h3>', unsafe_allow_html=True)
        try:
            if use_ensemble:
                best_name = max(classifiers, key=lambda n: per_model_probs[n].max())
            else:
                best_name = selected_single_model
            best_model = classifiers[best_name]["model"]
            best_size = classifiers[best_name]["img_size"]
            cam_overlay = compute_gradcam(best_model, best_name, cropped, best_size, device=DEVICE)
            gc1, gc2 = st.columns(2)
            gc1.image(cropped, caption="Cropped lesion input")
            gc2.image(cam_overlay, caption=f"Grad-CAM ({best_name})")
            st.caption("Warmer regions indicate areas the model weighted most heavily for its prediction.")
        except Exception as e:
            import traceback
            st.warning(f"Grad-CAM unavailable: {e}")
            with st.expander("Show full error details"):
                st.code(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="footnote">This output is generated by an automated model ensemble and is '
        'intended for research and educational demonstration purposes only. It is not a '
        'medical device and must not be used as a substitute for professional dermatological '
        'evaluation.</p>',
        unsafe_allow_html=True,
    )