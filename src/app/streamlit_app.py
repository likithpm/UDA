"""Streamlit app for multimodal underwater detection and classification."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import sys
import tempfile
import time
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import pandas as pd
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
ThreatHistory = List[Tuple[float, List[Tuple[str, float]]]]

THREAT_LEVELS = {
    "shark": 9,
    "submarine": 8,
    "whale": 7,
    "seal": 5,
    "dolphin": 4,
    "octopus": 5,
    "crab": 2,
    "starfish": 1,
    "seahorse": 1,
    "human": 6,
}


def _render_signal_card(
    placeholder,
    title: str,
    value: str,
    tone: str = "neutral",
) -> None:
    """Render one glassmorphism dashboard card."""
    placeholder.markdown(
        (
            f"<div class='glass-card card signal-card tone-{tone}'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_video_label(placeholder, high_threat_active: bool) -> None:
    """Render video section label with optional high-threat glow."""
    tone_class = "video-label-danger" if high_threat_active else "video-label-safe"
    placeholder.markdown(
        (
            f"<div class='glass-card card video-label {tone_class}'>"
            "🎥 Live Underwater Feed"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_threat(level: int, label: str) -> str:
    """Build threat-panel HTML content based on severity level."""
    if level >= 7:
        return (
            "<div class='glass-card card threat-card threat-high pulse-high'>"
            "<div class='card-title'>Threat Panel</div>"
            f"<div class='card-value'><b>🚨 HIGH THREAT: {label.upper()} (Level {level})</b></div>"
            "</div>"
        )
    if level >= 4:
        return (
            "<div class='glass-card card threat-card threat-medium'>"
            "<div class='card-title'>Threat Panel</div>"
            f"<div class='card-value'>⚠️ MEDIUM THREAT: {label} (Level {level})</div>"
            "</div>"
        )
    return (
        "<div class='glass-card card threat-card threat-low'>"
        "<div class='card-title'>Threat Panel</div>"
        f"<div class='card-value'>🟢 LOW THREAT: {label} (Level {level})</div>"
        "</div>"
    )


def _update_object_history(
    object_history: ThreatHistory,
    current_time: float,
    detected_objects: List[Tuple[str, float]],
) -> None:
    """Append current detections and keep only the last 5 seconds."""
    object_history.append((current_time, detected_objects))
    object_history[:] = [
        (timestamp, objs)
        for (timestamp, objs) in object_history
        if current_time - timestamp <= 5.0
    ]


def _dominant_object_by_confidence(object_history: ThreatHistory) -> str | None:
    """Return dominant object using confidence-weighted scores over history."""
    # Confidence-weighted aggregation reduces noise from weak, one-off detections.
    scores: defaultdict[str, float] = defaultdict(float)
    for _, objects in object_history:
        for label, conf in objects:
            scores[label] += conf

    if not scores:
        return None
    return max(scores, key=scores.get)


def _render_threat_alert(
    alert_placeholder,
    blink_placeholder,
    dominant_object: str | None,
    stable_duration: float,
    stability_threshold: float = 3.0,
) -> None:
    """Render threat UI once a dominant object remains stable long enough."""
    if not dominant_object:
        _render_signal_card(
            alert_placeholder,
            "Threat Panel",
            "🟢 Threat monitor: No recent objects in 5-second window",
            tone="low",
        )
        blink_placeholder.empty()
        return

    # Stability gate avoids rapid alert switching when detections fluctuate frame to frame.
    if stable_duration < stability_threshold:
        _render_signal_card(
            alert_placeholder,
            "Threat Panel",
            "Monitoring environment...",
            tone="neutral",
        )
        blink_placeholder.empty()
        return

    level = THREAT_LEVELS.get(dominant_object, 3)
    if level >= 7:
        alert_placeholder.markdown(_render_threat(level, dominant_object), unsafe_allow_html=True)
        blink_placeholder.markdown(
            (
                "<div class='alert-blink'>"
                f"🚨 HIGH THREAT DETECTED: {dominant_object.upper()}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    elif level >= 4:
        alert_placeholder.markdown(_render_threat(level, dominant_object), unsafe_allow_html=True)
        blink_placeholder.empty()
    else:
        alert_placeholder.markdown(_render_threat(level, dominant_object), unsafe_allow_html=True)
        blink_placeholder.empty()


def _play_sound_alert(sound_placeholder) -> None:
    """Trigger a browser-safe autoplay sound using HTML audio."""
    sound_placeholder.markdown(
        """
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/beep-07.mp3" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _append_threat_log(event_time: float, object_name: str, level: int) -> None:
    """Add a threat event while avoiding duplicate consecutive entries."""
    event = {
        "time": event_time,
        "object": object_name,
        "level": level,
    }

    # Deduplicate repeated stable alerts for the same object/level.
    if st.session_state.threat_log:
        last_event = st.session_state.threat_log[-1]
        if (
            last_event.get("object") == event["object"]
            and last_event.get("level") == event["level"]
        ):
            return

    st.session_state.threat_log.append(event)
    if len(st.session_state.threat_log) > 20:
        st.session_state.threat_log = st.session_state.threat_log[-20:]


def _render_threat_history(log_placeholder) -> None:
    """Render recent threat events below video (latest first)."""
    with log_placeholder.container():
        st.markdown("### 📜 Threat History")
        recent_events = st.session_state.threat_log[-10:]
        if not recent_events:
            st.markdown(
                "<div class='glass-card card history-card'>No threat events logged yet.</div>",
                unsafe_allow_html=True,
            )
        else:
            for event in reversed(recent_events):
                st.markdown(
                    (
                        "<div class='glass-card card history-card'>"
                        f"[{event['time']:.1f}s] {event['object']} -> Level {event['level']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("### 📊 Threat Analytics")
        if st.session_state.threat_log:
            # Convert log to DataFrame so charting stays explicit and extensible.
            threat_df = pd.DataFrame(st.session_state.threat_log)
            object_counts_df = (
                threat_df["object"]
                .value_counts()
                .rename_axis("object")
                .reset_index(name="count")
                .set_index("object")
            )
            st.bar_chart(object_counts_df)
        else:
            st.info("No analytics data available yet.")


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


def _render_uploaded_video(
    temp_video_path: Path,
    enable_threat: bool,
    enable_sound: bool,
    stability_threshold: float,
) -> None:
    """Run full multimodal pipeline for an uploaded video file."""
    st.markdown("<div class='section-title'>System Signals</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔊 Audio Panel")
            audio_placeholder = st.empty()
        with col2:
            st.markdown("### 🐠 Object Panel")
            object_placeholder = st.empty()

    st.markdown("---")
    st.markdown("<div class='section-title'>Threat Intelligence</div>", unsafe_allow_html=True)
    with st.container():
        alert_placeholder = st.empty()
        blink_placeholder = st.empty()
        sound_placeholder = st.empty()
    st.markdown("<div class='section-title'>Video Stream</div>", unsafe_allow_html=True)
    with st.container():
        video_label_placeholder = st.empty()
        frame_placeholder = st.empty()
        progress_bar = st.progress(0.0)
    st.markdown("<div class='section-title'>Logs & Analytics</div>", unsafe_allow_html=True)
    threat_log_placeholder = st.empty()

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
    object_history: ThreatHistory = []
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None

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
            _render_signal_card(
                audio_placeholder,
                "Audio Intelligence",
                f"🔊 {smoothed_audio} ({current_conf:.2f})",
                tone="neutral",
            )

            annotated, current_objects, current_detections = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
            )

            if enable_threat:
                _update_object_history(
                    object_history=object_history,
                    current_time=current_time,
                    detected_objects=current_detections,
                )
                dominant_object = _dominant_object_by_confidence(object_history)

                if dominant_object == last_dominant_object:
                    if stable_start_time is None:
                        stable_start_time = current_time
                else:
                    last_dominant_object = dominant_object
                    stable_start_time = current_time

                stable_duration = (
                    current_time - stable_start_time
                    if stable_start_time is not None
                    else 0.0
                )

                _render_threat_alert(
                    alert_placeholder,
                    blink_placeholder,
                    dominant_object,
                    stable_duration,
                    stability_threshold,
                )

                is_stable = stable_duration >= stability_threshold
                high_threat_active = False
                if dominant_object and is_stable:
                    level = THREAT_LEVELS.get(dominant_object, 3)
                    high_threat_active = level >= 7

                    # Play sound once when a new stable high-threat object is detected.
                    if level >= 7 and enable_sound:
                        if dominant_object != last_alert_object:
                            _play_sound_alert(sound_placeholder)
                            last_alert_object = dominant_object

                    if level < 7:
                        sound_placeholder.empty()
                        last_alert_object = None

                    _append_threat_log(current_time, dominant_object, level)
                else:
                    sound_placeholder.empty()
                    last_alert_object = None
                    high_threat_active = False
            else:
                alert_placeholder.empty()
                blink_placeholder.empty()
                sound_placeholder.empty()
                last_alert_object = None
                high_threat_active = False

            objects_str = ", ".join(sorted(current_objects)) if current_objects else "None"
            _render_signal_card(
                object_placeholder,
                "Object Intelligence",
                f"🐠 {objects_str}",
                tone="neutral",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            writer.write(annotated)
            frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            frame_elapsed = max(time.time() - start_time, 1e-9)
            fps_placeholder.caption(f"⚡ FPS: {1.0 / frame_elapsed:.2f}")

            current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = current_frame / total_frames if total_frames > 0 else 0.0
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            _render_threat_history(threat_log_placeholder)
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


def _run_webcam_mode(
    enable_threat: bool,
    enable_sound: bool,
    stability_threshold: float,
) -> None:
    """Run webcam detection stream using YOLO only."""
    st.markdown("<div class='section-title'>System Signals</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔊 Audio Panel")
            audio_placeholder = st.empty()
        with col2:
            st.markdown("### 🐠 Object Panel")
            object_placeholder = st.empty()

    st.markdown("---")
    st.markdown("<div class='section-title'>Threat Intelligence</div>", unsafe_allow_html=True)
    with st.container():
        alert_placeholder = st.empty()
        blink_placeholder = st.empty()
        sound_placeholder = st.empty()
    st.markdown("<div class='section-title'>Video Stream</div>", unsafe_allow_html=True)
    with st.container():
        video_label_placeholder = st.empty()
        frame_placeholder = st.empty()
    st.markdown("<div class='section-title'>Logs & Analytics</div>", unsafe_allow_html=True)
    threat_log_placeholder = st.empty()

    with st.spinner("Processing video..."):
        yolo_model, _, _, _, yolo_device = load_models()

    st.markdown(
        "<div class='glass-card card history-card'>Webcam mode active. Turn off 🔴 Live Mode to end stream.</div>",
        unsafe_allow_html=True,
    )

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to access webcam.")
    object_history: ThreatHistory = []
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None
    webcam_start_time = time.time()

    try:
        max_frames = 600
        for _ in range(max_frames):
            if not st.session_state.get("webcam_active", False):
                break

            ok, frame = capture.read()
            if not ok:
                st.warning("Failed to read frame from webcam.")
                break

            annotated, current_objects, current_detections = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
            )

            if enable_threat:
                current_time = time.time() - webcam_start_time
                _update_object_history(
                    object_history=object_history,
                    current_time=current_time,
                    detected_objects=current_detections,
                )
                dominant_object = _dominant_object_by_confidence(object_history)

                if dominant_object == last_dominant_object:
                    if stable_start_time is None:
                        stable_start_time = current_time
                else:
                    last_dominant_object = dominant_object
                    stable_start_time = current_time

                stable_duration = (
                    current_time - stable_start_time
                    if stable_start_time is not None
                    else 0.0
                )

                _render_threat_alert(
                    alert_placeholder,
                    blink_placeholder,
                    dominant_object,
                    stable_duration,
                    stability_threshold,
                )

                is_stable = stable_duration >= stability_threshold
                high_threat_active = False
                if dominant_object and is_stable:
                    level = THREAT_LEVELS.get(dominant_object, 3)
                    high_threat_active = level >= 7

                    # Play sound once when a new stable high-threat object is detected.
                    if level >= 7 and enable_sound:
                        if dominant_object != last_alert_object:
                            _play_sound_alert(sound_placeholder)
                            last_alert_object = dominant_object

                    if level < 7:
                        sound_placeholder.empty()
                        last_alert_object = None

                    _append_threat_log(current_time, dominant_object, level)
                else:
                    sound_placeholder.empty()
                    last_alert_object = None
                    high_threat_active = False
            else:
                alert_placeholder.empty()
                blink_placeholder.empty()
                sound_placeholder.empty()
                last_alert_object = None
                high_threat_active = False

            _render_signal_card(
                audio_placeholder,
                "Audio Intelligence",
                "🔊 Webcam mode (N/A)",
                tone="neutral",
            )
            objects_str = ", ".join(sorted(current_objects)) if current_objects else "None"
            _render_signal_card(
                object_placeholder,
                "Object Intelligence",
                f"🐠 {objects_str}",
                tone="neutral",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            _render_threat_history(threat_log_placeholder)
    finally:
        capture.release()

    if st.session_state.get("webcam_active", False):
        st.session_state.webcam_active = False
        st.info("Webcam session ended. Start again if needed.")


def main() -> None:
    """Render Streamlit UI and route to uploaded-video or webcam mode."""
    st.set_page_config(page_title="Underwater AI Detection System", layout="wide")

    # Theme and animation logic is centralized here for easier maintenance.
    st.markdown(
        """
<style>
.stApp {
    background: linear-gradient(180deg, #0a2540, #001f3f, #000814);
    color: #ffffff;
}

h1 {
    color: #00e5ff;
    text-shadow: 0 0 14px rgba(0, 229, 255, 0.45);
}

h2, h3 {
    color: #4fc3f7;
}

.main .block-container {
    padding-top: 1.4rem;
}

.section-title {
    color: #00e5ff;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
    margin-bottom: 0.5rem;
}

.card:hover {
    transform: scale(1.02);
    transition: 0.3s ease;
}

.signal-card .card-title,
.threat-card .card-title,
.history-card {
    color: #00e5ff;
    font-size: 0.9rem;
    font-weight: 600;
}

.card-value {
    margin-top: 0.35rem;
    color: #ffffff;
    font-size: 1.08rem;
    font-weight: 600;
}

.tone-neutral {
    box-shadow: 0 0 16px rgba(79, 195, 247, 0.16);
}

.tone-low {
    box-shadow: 0 0 16px rgba(76, 245, 181, 0.24);
}

.threat-high {
    border-color: rgba(255, 77, 79, 0.8);
    box-shadow: 0 0 22px rgba(255, 77, 79, 0.62);
}

.threat-medium {
    border-color: rgba(255, 170, 0, 0.7);
    box-shadow: 0 0 20px rgba(255, 170, 0, 0.4);
}

.threat-low {
    border-color: rgba(79, 195, 247, 0.6);
    box-shadow: 0 0 18px rgba(79, 195, 247, 0.32);
}

.video-label {
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.video-label-safe {
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.22);
}

.video-label-danger {
    border-color: rgba(255, 77, 79, 0.85);
    box-shadow: 0 0 22px rgba(255, 77, 79, 0.68);
}

.signal-card,
.threat-card,
.history-card,
.video-label {
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
}

@keyframes pulse {
    0% { box-shadow: 0 0 5px red; }
    50% { box-shadow: 0 0 20px red; }
    100% { box-shadow: 0 0 5px red; }
}

.pulse-high {
    animation: pulse 2s infinite;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071628, #0a1f38);
    border-right: 1px solid rgba(79, 195, 247, 0.22);
}

[data-testid="stSidebar"] * {
    color: #e8f7ff;
}

[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [role="button"] {
    border-radius: 12px;
}

[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] [role="button"]:hover {
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.35);
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(79, 195, 247, 0.28);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}

.alert-blink {
    color: #ff4d4f;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.35rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("")
    st.markdown(
        "<h1 style='text-align: center;'>🌊 Underwater AI Monitoring System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Real-time Marine Object & Acoustic Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("---")

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 0.3
    if "threat_log" not in st.session_state:
        st.session_state.threat_log = []

    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("### 🎛 Input")
    uploaded_video = st.sidebar.file_uploader("Upload video (mp4)", type=["mp4"])
    webcam_mode = st.sidebar.checkbox("Use webcam mode")
    live_mode = st.sidebar.toggle("🔴 Live Mode", value=st.session_state.webcam_active)
    enable_threat = st.sidebar.toggle("🚨 Enable Threat Detection")
    enable_sound = st.sidebar.toggle("🔊 Enable Sound Alert")
    if st.sidebar.button("🧹 Clear Threat Log"):
        st.session_state.threat_log = []
    stability_threshold = st.sidebar.slider(
        "Stability Duration (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
    )
    st.sidebar.markdown("### 🎯 Detection")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.confidence_threshold),
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
                _render_uploaded_video(
                    temp_video_path,
                    enable_threat=enable_threat,
                    enable_sound=enable_sound,
                    stability_threshold=stability_threshold,
                )
            finally:
                temp_video_path.unlink(missing_ok=True)
        elif webcam_mode or live_mode:
            _run_webcam_mode(
                enable_threat=enable_threat,
                enable_sound=enable_sound,
                stability_threshold=stability_threshold,
            )
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
