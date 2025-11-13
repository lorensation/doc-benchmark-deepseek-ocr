import streamlit as st
from PIL import Image
from io import BytesIO
import tempfile
import torch
from transformers import AutoModel, AutoTokenizer
import pytesseract

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="DeepSeek-OCR Benchmark",
    layout="wide",
)

st.title("📄 DeepSeek-OCR vs Tesseract — CPU Benchmark Demo")
st.write(
    "Upload an image and compare **DeepSeek-OCR** (open-weight model) "
    "against **Tesseract** (classic OCR baseline). "
    "This Space runs entirely on **CPU** — no external API required."
)

# ---------------------------------------------------------
# MODEL LOAD (cached for faster cold starts)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_deepseek():
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True
    )
    model.eval()
    return tokenizer, model


st.sidebar.subheader("Model Loading")
with st.sidebar:
    with st.spinner("Loading DeepSeek-OCR (CPU)…"):
        tokenizer, deepseek_model = load_deepseek()
    st.success("DeepSeek-OCR loaded!")

# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an image (PNG/JPG/JPEG). For PDFs, upload a single-page image.",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📎 Uploaded Image")
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

    # -----------------------------------------------------
    # RUN BENCHMARK
    # -----------------------------------------------------
    if st.button("Run OCR Benchmark"):
        st.subheader("🔍 Extracting text...")

        # ---------------------------
        # DeepSeek-OCR extraction
        # ---------------------------
        with st.spinner("Running DeepSeek-OCR…"):
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

            with tempfile.TemporaryDirectory() as tmpdir:
                result = deepseek_model.infer(
                    tokenizer,
                    prompt=prompt,
                    image=image,
                    image_file=None,
                    output_path=tmpdir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                    test_compress=False
                )

            if isinstance(result, dict) and "text" in result:
                deepseek_text = result["text"]
            else:
                deepseek_text = str(result)

        # ---------------------------
        # Tesseract extraction
        # ---------------------------
        with st.spinner("Running Tesseract…"):
            tess_text = pytesseract.image_to_string(image)

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------
        col_d, col_t = st.columns(2)

        with col_d:
            st.subheader("🤖 DeepSeek-OCR Output")
            st.code(deepseek_text, language="markdown")
            st.metric("Text Length", len(deepseek_text))

        with col_t:
            st.subheader("📘 Tesseract Output")
            st.code(tess_text, language="text")
            st.metric("Text Length", len(tess_text))

        # -------------------------------------------------
        # QUALITATIVE COMPARISON
        # -------------------------------------------------
        st.subheader("📊 Comparison Summary")
        st.write(
            f"**DeepSeek-OCR length:** {len(deepseek_text)} characters\n\n"
            f"**Tesseract length:** {len(tess_text)} characters\n\n"
        )

        if len(deepseek_text) > len(tess_text):
            st.success("DeepSeek-OCR extracted more content.")
        elif len(deepseek_text) < len(tess_text):
            st.info("Tesseract extracted more content.")
        else:
            st.warning("Both models extracted text of equal length (rare!).")

else:
    st.info("Upload an image to begin.")
