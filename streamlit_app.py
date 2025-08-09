# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import time
import os
from utils.tflite_predictor import WeedDetector

# --- Page config ---
st.set_page_config(page_title="🌱 Favors CNN Weed Detector", 
                   layout="wide",
                   page_icon="assets/icon.ico")

# --- Cache model load ---
@st.cache_resource
def load_model():
    model_path = os.path.join("assets", "weed_detector.tflite")
    return WeedDetector(model_path)

detector = load_model()
interpreter = detector.interpreter
input_details = detector.input_details
output_details = detector.output_details
labels = detector.classes

# --- Session state defaults ---
if "latest_probabilities" not in st.session_state:
    st.session_state.latest_probabilities = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = False

# --- Utilities ---
def run_detection_on_frame(frame):
    """
    Takes a BGR cv2 frame, runs the TFLite model, updates session_state.latest_probabilities,
    and returns an annotated frame (BGR).
    """
    img_resized = cv2.resize(frame, (128, 128))
    img_normalized = img_resized.astype("float32") / 255.0
    input_data = np.expand_dims(img_normalized, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output_data[0]))
    pred_label = labels[pred_index]
    st.session_state.latest_probabilities = {label: float(output_data[0][i]) for i, label in enumerate(labels)}

    annotated = frame.copy()
    h, w, _ = annotated.shape
    # Put predicted label and top confidence
    conf = float(output_data[0][pred_index])
    disp_text = f"{pred_label} ({conf*100:.1f}%)"
    cv2.putText(annotated, disp_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated

def plot_metrics_placeholder(placeholder):
    """Plot current probabilities to the given placeholder (st.empty()), overwriting each time."""
    if not st.session_state.latest_probabilities:
        placeholder.info("No metrics yet.")
        return

    probs = st.session_state.latest_probabilities
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct fixed colors

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(probs.keys(), probs.values(), color=colors[:len(probs)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    placeholder.pyplot(fig, clear_figure=True)  # Ensures no stacking
    plt.close(fig)  # Free memory


# --- Layout ---
st.title("🌱 Favors CNN Weed Detector — Unified Streamlit App")
st.markdown("Live webcam, image upload, and video upload with **real-time** metrics.")

col1, col2 = st.columns([2, 1])
feed_placeholder = col1.empty()
controls = col2

# Controls column
with controls:
    st.subheader("Controls")
    # Camera controls
    start_btn = st.button("▶ Start Live Camera Detection") if not st.session_state.camera_active else None
    stop_btn = st.button("⏹ Stop Camera") if st.session_state.camera_active else None

    # Upload widgets
    uploaded_image = st.file_uploader("📷 Detect from Image", type=["jpg", "jpeg", "png"])
    uploaded_video = st.file_uploader("🎞 Detect from Video", type=["mp4", "mov", "avi"])

    # Metrics display area
    st.markdown("---")
    st.subheader("Live Metrics")
    metrics_placeholder = st.empty()


# React to Start / Stop
if start_btn:
    st.session_state.camera_active = True
    st.session_state.stop_signal = False

if stop_btn:
    st.session_state.stop_signal = True
    st.session_state.camera_active = False

# --- Webcam live loop ---
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera. Make sure it is connected and not used by another app.")
        st.session_state.camera_active = False
    else:
        # Small UI: show a stop button inline (in addition to sidebar)
        stop_inline = col2.button("⏹ Stop Camera (inline)")
        if stop_inline:
            st.session_state.stop_signal = True
            st.session_state.camera_active = False

        try:
            # Run until stop signal set
            while cap.isOpened() and not st.session_state.stop_signal:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = run_detection_on_frame(frame)
                # Show feed
                feed_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                # Update metrics
                plot_metrics_placeholder(metrics_placeholder)

                # small sleep to limit CPU and keep UI responsive (~15-20 FPS)
                time.sleep(0.05)

            cap.release()
        except Exception as e:
            cap.release()
            st.error(f"Camera loop error: {e}")
        finally:
            # Ensure camera state is consistent
            st.session_state.camera_active = False
            st.session_state.stop_signal = False

# --- Image upload detection ---
if uploaded_image is not None:
    # Read image bytes into OpenCV
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read the uploaded image.")
    else:
        annotated = run_detection_on_frame(img)
        feed_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        plot_metrics_placeholder(metrics_placeholder)

# --- Video upload detection (process frames with live metrics) ---
if uploaded_video is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output file
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        progress_bar = st.progress(0)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = run_detection_on_frame(frame)
                out.write(annotated)

                # show annotated frame and metrics as we go
                feed_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                plot_metrics_placeholder(metrics_placeholder)

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # allow UI to update and limit CPU
                time.sleep(0.01)

        finally:
            cap.release()
            out.release()
            progress_bar.empty()

        # Offer download of annotated video
        with open(out_path, "rb") as f:
            st.download_button("⬇ Download Annotated Video", f, file_name="annotated_output.mp4")

        # cleanup
        os.remove(tfile.name)
        os.remove(out_path)
    except Exception as e:
        st.error(f"Error processing uploaded video: {e}")
        try:
            os.remove(tfile.name)
        except Exception:
            pass

# --- Final metrics display (if nothing else is running) ---
if not st.session_state.camera_active and uploaded_video is None and uploaded_image is None:
    # Show last-known metrics or a helpful message
    if st.session_state.latest_probabilities:
        plot_metrics_placeholder(metrics_placeholder)
    else:
        metrics_placeholder.info("No predictions yet — run a detection (webcam, image or video).")

# --- Footer / quick tips ---
st.markdown("---")
st.markdown(
    """
    **Notes & Tips**
    - If the camera can't be accessed, ensure no other application is using it and that your browser (if running remotely) allows camera access.
    - Model file should be at `assets/weed_detector.tflite`. Keep `utils/tflite_predictor.py` in the `utils` package (same layout as your Kivy app).
    """
)
