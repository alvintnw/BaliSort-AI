
from pathlib import Path

import streamlit as st

from database.db import (
    export_feedback_csv,
    get_prediction_summary,
    get_recent_predictions,
    save_feedback,
    save_prediction,
)
from src.predict import get_device, load_model, open_uploaded_image, predict_image
from src.model import CLASS_NAMES
from src.recommendation import RECOMMENDATIONS, get_recommendation


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DATABASE_PATH = PROJECT_ROOT / "database" / "balisort.db"

# Page configuration
st.set_page_config(
    page_title="BaliSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0f766e, #166534);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero h1 { margin-bottom: 0.25rem; }
    .hero p { margin-bottom: 0; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang BaliSort AI")
    st.write(
        "Bantu kenali sampah dan pilih penanganan yang lebih bertanggung jawab "
        "untuk lingkungan Bali."
    )
    st.divider()
    st.caption("Model: MobileNetV3-Large")
    st.caption("Input: RGB 224 × 224 piksel")
    st.caption("Kategori: 6 jenis sampah")

st.markdown(
    """
    <div class="hero">
        <h1>♻️ BaliSort AI</h1>
        <p>Kenali sampah. Pahami dampaknya. Pilah dengan lebih baik.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Klasifikasi gambar sampah")
st.write("Unggah foto dengan objek sampah terlihat jelas dan tidak terlalu kecil.")

@st.cache_resource
def get_app_model():
    """Load the model once per Streamlit process."""
    device = get_device()
    return load_model(MODEL_PATH, device), device


uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, dan PNG.",
)

if uploaded_file is not None:
    try:
        image = open_uploaded_image(uploaded_file.getvalue())
        image_column, result_column = st.columns([1, 1], gap="large")
        with image_column:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        model, device = get_app_model()
        with st.spinner("Menganalisis gambar..."):
            result = predict_image(model, image, device)

        recommendation = get_recommendation(result["class_name"])
        with result_column:
            st.caption("Hasil analisis")
            st.metric("Kategori", recommendation["label"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            if result["is_uncertain"]:
                st.warning("Hasil belum yakin. Coba unggah foto yang lebih jelas.")
            else:
                st.success("Prediksi cukup yakin untuk referensi pemilahan.")
            st.info(f"**Wadah:** {recommendation['bin']}\n\n{recommendation['advice']}")

            if st.button("Simpan ke riwayat", type="primary"):
                prediction_id = save_prediction(
                    DATABASE_PATH,
                    uploaded_file.name,
                    result["class_name"],
                    result["confidence"],
                )
                st.session_state["last_prediction_id"] = prediction_id
                st.toast("Prediksi tersimpan.")

            with st.expander("Lihat probabilitas semua kategori"):
                for class_name, probability in result["probabilities"].items():
                    st.progress(probability, text=f"{class_name}: {probability:.1%}")

            prediction_id = st.session_state.get("last_prediction_id")
            if prediction_id is not None:
                st.subheader("Bantu tingkatkan model")
                feedback_choice = st.radio(
                    "Apakah prediksi ini benar?",
                    options=("Ya, benar", "Tidak, pilih kategori yang benar"),
                    key=f"feedback_choice_{prediction_id}",
                )
                actual_class = None
                if feedback_choice != "Ya, benar":
                    actual_class = st.selectbox(
                        "Kategori yang benar",
                        options=list(CLASS_NAMES),
                        key=f"actual_class_{prediction_id}",
                    )
                if st.button("Kirim feedback", key=f"submit_feedback_{prediction_id}"):
                    save_feedback(
                        DATABASE_PATH,
                        prediction_id,
                        feedback_choice == "Ya, benar",
                        actual_class,
                    )
                    st.success("Terima kasih, feedback tersimpan.")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        st.error(f"Prediksi tidak dapat dilakukan: {error}")

st.divider()
st.header("Riwayat prediksi tersimpan")
try:
    history = get_recent_predictions(DATABASE_PATH)
    if history:
        st.caption(f"{len(history)} prediksi terakhir")
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada prediksi yang disimpan.")
except (OSError, ValueError) as error:
    st.error(f"Riwayat tidak dapat dibaca: {error}")

st.header("Ringkasan penggunaan")
try:
    summary = get_prediction_summary(DATABASE_PATH)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Total prediksi", summary["prediction_count"])
    metric_columns[1].metric(
        "Confidence rata-rata",
        f"{summary['average_confidence']:.1%}",
    )
    feedback_accuracy = summary["feedback_accuracy"]
    metric_columns[2].metric(
        "Feedback benar",
        "Belum ada" if feedback_accuracy is None else f"{feedback_accuracy:.1%}",
    )
    if summary["category_counts"]:
        st.caption("Distribusi kategori hasil prediksi")
        st.bar_chart(summary["category_counts"])
    if summary["feedback_count"] == 0:
        st.caption("Belum ada feedback untuk dihitung.")
    else:
        st.download_button(
            "Unduh feedback untuk kurasi dataset",
            data=export_feedback_csv(DATABASE_PATH),
            file_name="balisort_feedback.csv",
            mime="text/csv",
            help="Unduh metadata feedback tanpa gambar unggahan.",
        )
except (OSError, ValueError) as error:
    st.error(f"Ringkasan tidak dapat dibaca: {error}")

st.divider()
st.header("Panduan pemilahan sampah")
st.write(
    "Gunakan panduan ini sebagai referensi. Bersihkan dan keringkan material "
    "yang masih dapat didaur ulang sebelum disalurkan."
)

education_tabs = st.tabs(
    [RECOMMENDATIONS[class_name]["label"] for class_name in CLASS_NAMES]
)
for tab, class_name in zip(education_tabs, CLASS_NAMES):
    recommendation = get_recommendation(class_name)
    with tab:
        st.subheader(recommendation["label"])
        st.markdown(f"**Wadah yang disarankan:** {recommendation['bin']}")
        st.write(recommendation["advice"])
        if class_name == "trash":
            st.warning(
                "Residu tidak boleh dicampur dengan bahan yang masih bisa "
                "digunakan atau didaur ulang."
            )
        elif class_name == "glass":
            st.warning(
                "Tangani pecahan kaca dengan hati-hati dan bungkus sebelum "
                "dimasukkan ke wadah."
            )
        else:
            st.success("Pisahkan dari residu dan pastikan material tetap bersih.")

st.info(
    "BaliSort AI membantu mengenali pola visual pada gambar. Untuk benda "
    "berbahaya, tercampur, atau hasil yang tidak yakin, lakukan pemeriksaan "
    "manual dan ikuti aturan pengelolaan sampah setempat."
)

st.caption("BaliSort AI · AI Web Innovation Challenge 2026")
