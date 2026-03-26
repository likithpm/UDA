"""Streamlit app for multimodal underwater detection and classification."""

from __future__ import annotations

from collections import Counter, deque
import sys
import tempfile
import time
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import streamlit as st

try:
    from src.inference.pipeline import (
        annotate_and_collect_objects,
        load_models,
        process_audio_chunks,
    )
except ModuleNotFoundError:
    # Allow streamlit execution from different working directories.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.inference.pipeline import (
        annotate_and_collect_objects,
        load_models,
        process_audio_chunks,
    )

AudioPrediction = Tuple[float, float, str, float]


def _prediction_for_timestamp(
    timestamp_seconds: float,
    audio_predictions: List[AudioPrediction],
) -> AudioPrediction:
    """Return the audio chunk prediction corresponding to frame timestamp."""
    for prediction in audio_predictions:
        start_time, end_time, _, _ = prediction
        if start_time <= timestamp_seconds <= end_time:
            return prediction
    return audio_predictions[-1]


def _render_uploaded_video(temp_video_path: Path) -> None:
    """Run full multimodal pipeline for an uploaded video file."""
    with st.container():
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔊 Audio")
            audio_placeholder = st.empty()
        with col2:
            st.markdown("### 🐠 Object Insights")
            object_placeholder = st.empty()

    st.markdown("---")
    with st.container():
        st.subheader("🎥 Live Detection")
        frame_placeholder = st.empty()
        progress_bar = st.progress(0.0)

    with st.spinner("Processing video..."):
        yolo_model, audio_model, class_to_idx, device, yolo_device = load_models()
        audio_predictions = process_audio_chunks(
            video_path=temp_video_path,
            audio_model=audio_model,
            class_to_idx=class_to_idx,
            device=device,
        )

    capture = cv2.VideoCapture(str(temp_video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open uploaded video file.")

    fps_placeholder = st.empty()
    audio_history: Deque[str] = deque(maxlen=3)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = Path(tempfile.gettempdir()) / "streamlit_processed_output.mp4"
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (frame_width, frame_height),
    )

    try:
        while True:
            start_time = time.time()
            ok, frame = capture.read()
            if not ok:
                break

            current_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            current_audio = "Unknown"
            current_conf = 0.0
            for start, end, label, conf in audio_predictions:
                if start <= current_time <= end:
                    current_audio = label
                    current_conf = conf
                    break
            else:
                _, _, current_audio, current_conf = audio_predictions[-1]

            filtered_audio = current_audio if current_conf >= 0.6 else "Low confidence"
            audio_history.append(filtered_audio)
            smoothed_audio = Counter(audio_history).most_common(1)[0][0]
            audio_placeholder.success(f"🔊 Audio: {smoothed_audio} ({current_conf:.2f})")

            annotated, current_objects = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
            )
            objects_str = ", ".join(sorted(current_objects)) if current_objects else "None"
            object_placeholder.info(f"🐠 Objects: {objects_str}")

            writer.write(annotated)
            frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            frame_elapsed = max(time.time() - start_time, 1e-9)
            fps_placeholder.caption(f"⚡ FPS: {1.0 / frame_elapsed:.2f}")

            current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = current_frame / total_frames if total_frames > 0 else 0.0
            progress_bar.progress(min(max(progress, 0.0), 1.0))
    finally:
        writer.release()
        capture.release()

    st.success("✅ Video processing completed")
    if output_video_path.exists():
        with output_video_path.open("rb") as output_file:
            st.download_button(
                label="📥 Download Processed Video",
                data=output_file.read(),
                file_name="output.mp4",
                mime="video/mp4",
            )


def _run_webcam_mode() -> None:
    """Run webcam detection stream using YOLO only."""
    with st.container():
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔊 Audio")
            audio_placeholder = st.empty()
        with col2:
            st.markdown("### 🐠 Object Insights")
            object_placeholder = st.empty()

    st.markdown("---")
    with st.container():
        st.subheader("🎥 Live Detection")
        frame_placeholder = st.empty()

    with st.spinner("Processing video..."):
        yolo_model, _, _, _, yolo_device = load_models()

    st.info("Webcam mode active. Turn off 🔴 Live Mode to end stream.")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to access webcam.")

    try:
        max_frames = 600
        for _ in range(max_frames):
            if not st.session_state.get("webcam_active", False):
                break

            ok, frame = capture.read()
            if not ok:
                st.warning("Failed to read frame from webcam.")
                break

            annotated, current_objects = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
            )
            audio_placeholder.success("🔊 Audio: Webcam mode (N/A)")
            objects_str = ", ".join(sorted(current_objects)) if current_objects else "None"
            object_placeholder.info(f"🐠 Objects: {objects_str}")

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
    finally:
        capture.release()

    if st.session_state.get("webcam_active", False):
        st.session_state.webcam_active = False
        st.info("Webcam session ended. Start again if needed.")


def main() -> None:
    """Render Streamlit UI and route to uploaded-video or webcam mode."""
    st.set_page_config(page_title="Underwater AI Detection System", layout="wide")
    st.markdown(
        """
<style>
.stApp {
    background: linear-gradient(135deg, #0e1a2b, #1e3c72);
    color: white;
}

h1, h2, h3 {
    color: #4fc3f7;
}

div[data-testid="stSidebar"] {
    background-color: #162447;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stText"] {
    color: #e8f1ff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("")
    st.markdown(
        "<h1 style='text-align: center;'>🌊 Underwater AI Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Real-time Marine Object & Audio Recognition</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("---")

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 0.5

    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("### 🎛 Input")
    uploaded_video = st.sidebar.file_uploader("Upload video (mp4)", type=["mp4"])
    webcam_mode = st.sidebar.checkbox("Use webcam mode")
    live_mode = st.sidebar.toggle("🔴 Live Mode", value=st.session_state.webcam_active)
    st.sidebar.markdown("### 🎯 Detection")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    st.session_state.confidence_threshold = confidence_threshold
    st.session_state.webcam_active = live_mode

    try:
        if uploaded_video is not None:
            if webcam_mode:
                st.warning("Webcam mode is ignored while uploaded video is selected.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_video_path = Path(tmp_file.name)

            try:
                _render_uploaded_video(temp_video_path)
            finally:
                temp_video_path.unlink(missing_ok=True)
        elif webcam_mode or live_mode:
            _run_webcam_mode()
        else:
            st.info("Upload an MP4 video or enable webcam mode from the sidebar.")
    except FileNotFoundError as err:
        st.error(f"Missing model or invalid file: {err}")
    except RuntimeError as err:
        st.error(f"Runtime error: {err}")
    except Exception as err:
        st.error(f"Unexpected error: {err}")


if __name__ == "__main__":
    main()
