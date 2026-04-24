"""Streamlit app for multimodal underwater detection and classification."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict, deque
import html
import json
import os
import sys
import tempfile
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import yaml

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

try:
    from src.config.config import AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH

AudioPrediction = Tuple[float, float, str, float]
ThreatHistory = List[Tuple[float, List[Tuple[str, float]]]]
MAX_CHAT_MESSAGES = 15
DEFAULT_TARGET_FPS = 18
DEFAULT_FRAME_SKIP = 2
DEFAULT_INFERENCE_SIZE = 640

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


def _class_count_from_dataset_folders() -> int:
    """Count classes from dataset_auto_labels/data.yaml names field."""
    if not AUTO_LABEL_DATA_YAML.exists():
        return 0

    try:
        parsed_yaml = yaml.safe_load(AUTO_LABEL_DATA_YAML.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return 0

    if not isinstance(parsed_yaml, dict):
        return 0

    names = parsed_yaml.get("names", [])
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    return 0


def _debug_log(debug_enabled: bool, message: str) -> None:
    """Print debug messages only when debug mode is enabled."""
    if debug_enabled:
        print(f"[DEBUG] {message}")


def _extract_ordered_names_from_data_yaml(parsed_yaml: dict) -> list[str]:
    """Extract class names in order from data.yaml names field."""
    names = parsed_yaml.get("names", [])
    if isinstance(names, list):
        return [str(name).strip() for name in names]
    if isinstance(names, dict):
        return [
            str(name).strip()
            for _, name in sorted(
                ((int(idx), value) for idx, value in names.items()),
                key=lambda item: item[0],
            )
        ]
    raise ValueError("'names' in data.yaml must be a list or dict.")


def _load_class_names_from_dataset_yaml() -> list[str]:
    """Load class names from dataset_auto_labels/data.yaml with strict validation."""
    if not AUTO_LABEL_DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset config: {AUTO_LABEL_DATA_YAML}")

    try:
        parsed_yaml = yaml.safe_load(AUTO_LABEL_DATA_YAML.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"Invalid YAML format in {AUTO_LABEL_DATA_YAML}: {err}") from err

    if not isinstance(parsed_yaml, dict):
        raise ValueError(f"Expected mapping in {AUTO_LABEL_DATA_YAML}, found: {type(parsed_yaml).__name__}")

    class_names = _extract_ordered_names_from_data_yaml(parsed_yaml)
    class_count = len(class_names)
    if class_count <= 0:
        raise ValueError("No classes found in dataset_auto_labels/data.yaml.")

    yaml_nc = parsed_yaml.get("nc")
    if isinstance(yaml_nc, int) and yaml_nc != class_count:
        raise ValueError(
            "Class count mismatch in data.yaml: "
            f"nc={yaml_nc}, names_count={class_count}"
        )

    return class_names


@st.cache_resource(show_spinner=False)
def _cached_model_class_count(model_path: str) -> int:
    """Load YOLO model once and return class count from model metadata."""
    model = YOLO(model_path)
    names = model.names
    if isinstance(names, (list, dict)):
        return len(names)
    raise ValueError(f"Unsupported YOLO model names type: {type(names).__name__}")


def _run_startup_validation(debug_enabled: bool) -> tuple[dict, str | None]:
    """Validate startup dependencies and return status map plus optional error."""
    status = {
        "model_loaded": False,
        "dataset_loaded": False,
        "classes_count": 0,
        "class_names": [],
    }

    try:
        if not YOLO_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Missing YOLO model file: {YOLO_WEIGHTS_PATH}")
        status["model_loaded"] = True
        _debug_log(debug_enabled, f"Model loaded path validated: {YOLO_WEIGHTS_PATH}")

        class_names = _load_class_names_from_dataset_yaml()
        status["dataset_loaded"] = True
        status["classes_count"] = len(class_names)
        status["class_names"] = class_names
        _debug_log(debug_enabled, f"Classes loaded: {class_names}")

        model_class_count = _cached_model_class_count(str(YOLO_WEIGHTS_PATH))
        if model_class_count != len(class_names):
            raise ValueError(
                "Model class count does not match data.yaml classes. "
                f"model={model_class_count}, data.yaml={len(class_names)}"
            )

        return status, None
    except Exception as err:
        return status, str(err)


def _saig_mode_label() -> str:
    """Return SaiG execution mode based on API key availability."""
    has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("SAIG_API_KEY"))
    return "API" if has_api_key else "Fallback"


def _render_system_status_panel(startup_status: dict | None = None) -> None:
    """Render top-level startup and system status cards."""
    startup_status = startup_status or {}
    model_ok = bool(startup_status.get("model_loaded", False))
    dataset_ok = bool(startup_status.get("dataset_loaded", False))
    classes_count = int(startup_status.get("classes_count", _class_count_from_dataset_folders()))

    saig_mode = _saig_mode_label()

    st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    _render_signal_card(
        col1.empty(),
        "Startup",
        "✔ Model Loaded" if model_ok else "✖ Model Not Loaded",
        tone="low" if model_ok else "neutral",
    )
    _render_signal_card(
        col2.empty(),
        "Dataset",
        "✔ Dataset Loaded" if dataset_ok else "✖ Dataset Not Loaded",
        tone="low" if dataset_ok else "neutral",
    )
    _render_signal_card(
        col3.empty(),
        "Classes / SaiG",
        f"✔ Classes: {classes_count} | 🧠 {saig_mode}",
        tone="neutral",
    )


def _render_performance_metrics(placeholder, fps: float, inference_ms: float) -> None:
    """Render live frame performance metrics."""
    _render_signal_card(
        placeholder,
        "Performance",
        f"⚡ FPS: {fps:.2f} | ⏱️ Inference: {inference_ms:.1f} ms",
        tone="neutral",
    )


def _append_chat_message(role: str, content: str) -> None:
    """Append one message and keep only recent chat turns in memory."""
    st.session_state.chat_history.append({"role": role, "content": content})
    if len(st.session_state.chat_history) > MAX_CHAT_MESSAGES:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_MESSAGES:]


def _open_webcam_with_retry(max_attempts: int = 3, delay_seconds: float = 0.4):
    """Try opening webcam more than once to handle transient device init failures."""
    for _ in range(max_attempts):
        capture = cv2.VideoCapture(0)
        if capture.isOpened():
            return capture
        capture.release()
        time.sleep(delay_seconds)
    return None


def encode_frame_to_base64(frame) -> str:
    """Convert a BGR frame to RGB and return base64-encoded JPEG bytes."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".jpg", rgb_frame)
    if not ok:
        raise RuntimeError("Failed to encode frame for SaiG analysis.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _estimate_threat_level(detected_objects: List[str]) -> int:
    """Estimate threat level from currently detected objects."""
    if not detected_objects:
        return 1
    return max(THREAT_LEVELS.get(obj, 3) for obj in detected_objects)


def _simulated_saig_response(
    user_prompt: str,
    detected_objects: List[str],
    chat_history: List[dict],
) -> str:
    """Generate conversational fallback reply using detections and user context."""
    unique_objects = sorted(set(detected_objects))
    level = _estimate_threat_level(unique_objects)
    recent_turns = max(len(chat_history) - 1, 0)

    if unique_objects:
        object_text = ", ".join(unique_objects)
        scene_description = (
            f"From the latest frame, I can see {object_text}. "
            f"For your question \"{user_prompt}\", this scene suggests a threat level around {level}/10."
        )
        if level >= 7:
            reasoning = (
                "The reason is that at least one high-risk marine object is present, "
                "so the environment should be treated as potentially dangerous."
            )
        elif level >= 4:
            reasoning = (
                "The detections indicate moderate risk, which means caution is recommended "
                "while continuing to monitor upcoming frames."
            )
        else:
            reasoning = (
                "Only low-risk classes are visible right now, so the immediate threat appears limited."
            )
    else:
        scene_description = (
            f"I do not see confident object detections in the latest frame, but for your question "
            f"\"{user_prompt}\" I can still provide a cautious interpretation."
        )
        reasoning = (
            "Since the detector did not lock onto clear targets, the threat estimate remains low for now, "
            "and it would help to keep streaming for additional visual evidence."
        )

    memory_context = (
        f" I am also using context from {recent_turns} recent conversation turn(s) to keep continuity."
        if recent_turns > 0
        else ""
    )
    return f"{scene_description} {reasoning}{memory_context}"


def query_saig(
    user_prompt: str,
    image_base64: str,
    detected_objects: List[str],
    chat_history: List[dict],
) -> str:
    """Query OpenAI-compatible API with conversation + frame context, with fallback mode."""
    if not image_base64:
        return "No visual data available yet"

    fallback_response = _simulated_saig_response(user_prompt, detected_objects, chat_history)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SAIG_API_KEY")
    if not api_key:
        return fallback_response

    endpoint = os.getenv("SAIG_API_BASE_URL", "https://api.openai.com/v1/chat/completions")
    model_name = os.getenv("SAIG_MODEL", "gpt-4o-mini")
    detected_text = ", ".join(sorted(set(detected_objects))) if detected_objects else "none"
    prompt = (
        "You are SaiG, an underwater AI expert assistant.\n\n"
        "You help users understand underwater scenes using:\n"
        "- visual detection results\n"
        "- context from conversation\n\n"
        f"Current detected objects:\n{detected_text}\n\n"
        f"User question:\n{user_prompt}\n\n"
        "Provide a detailed, clear explanation in paragraph form, not bullet points."
    )

    messages = [{"role": "system", "content": "You are SaiG, an underwater AI expert assistant."}]
    for message in chat_history[-8:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    )

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
    }

    request_body = json.dumps(payload).encode("utf-8")
    http_request = urllib_request.Request(
        endpoint,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(http_request, timeout=25) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(str(part["text"]))
            llm_text = "\n".join(text_parts).strip()
        else:
            llm_text = str(content).strip()

        return llm_text or fallback_response
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, KeyError, ValueError):
        return fallback_response


def _render_saig_chat_history(container, chat_history: List[dict]) -> None:
    """Render scrollable SaiG conversation with left/right chat bubbles."""
    with container.container():
        if not chat_history:
            st.markdown(
                (
                    "<div class='glass-card card saig-card'>"
                    "<div class='card-value'>Ask SaiG about the current scene to start the conversation.</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            return

        message_blocks: List[str] = []
        for message in chat_history:
            role = str(message.get("role", "assistant")).strip().lower()
            content = html.escape(str(message.get("content", "")).strip()).replace("\n", "<br>")
            if not content:
                continue

            if role == "user":
                message_blocks.append(
                    "<div class='chat-row chat-row-user'>"
                    f"<div class='chat-bubble chat-bubble-user'>{content}</div>"
                    "</div>"
                )
            else:
                message_blocks.append(
                    "<div class='chat-row chat-row-assistant'>"
                    f"<div class='chat-bubble chat-bubble-assistant'>{content}</div>"
                    "</div>"
                )

        chat_html = "".join(message_blocks)
        st.markdown(
            (
                "<div class='glass-card card saig-chat-shell'>"
                "<div class='saig-chat-history'>"
                f"{chat_html}"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _handle_saig_chat_send(
    send_requested: bool,
    prompt_text: str,
    latest_frame,
    detected_objects: List[str],
    warning_placeholder,
    debug_enabled: bool = False,
) -> None:
    """Handle one chat send event and append assistant response to session history."""
    if not send_requested:
        return

    user_prompt = prompt_text.strip()
    if not user_prompt:
        warning_placeholder.warning("Please enter a question before sending.")
        return

    _append_chat_message("user", user_prompt)

    if latest_frame is None:
        _append_chat_message("assistant", "No visual data available yet")
        return

    _debug_log(debug_enabled, "SaiG query triggered")
    with st.spinner("Analyzing..."):
        try:
            encoded_image = encode_frame_to_base64(latest_frame)
            response = query_saig(
                user_prompt=user_prompt,
                image_base64=encoded_image,
                detected_objects=detected_objects,
                chat_history=st.session_state.chat_history[:-1],
            )
            _append_chat_message("assistant", response)
        except Exception as err:
            fallback = _simulated_saig_response(
                user_prompt=user_prompt,
                detected_objects=detected_objects,
                chat_history=st.session_state.chat_history,
            )
            _append_chat_message("assistant", fallback)
            warning_placeholder.warning(f"SaiG API failed; using fallback analysis. Details: {err}")


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
    # Guard memory footprint during long sessions.
    if len(object_history) > 120:
        del object_history[:-120]


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
    target_fps: int,
    frame_skip: int,
    inference_size: int,
    debug_enabled: bool,
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
        performance_placeholder = st.empty()
        frame_placeholder = st.empty()
        progress_bar = st.progress(0.0)
    st.markdown("<div class='section-title'>SaiG Assistant</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("### 🧠 SaiG Assistant")
        chat_history_placeholder = st.empty()
        chat_prompt = st.text_input(
            "Ask SaiG about this scene...",
            key="saig_chat_input_uploaded",
            label_visibility="collapsed",
            placeholder="Ask SaiG about this scene...",
        )
        chat_action_col_1, chat_action_col_2 = st.columns([1, 1])
        with chat_action_col_1:
            send_chat = st.button("Send", key="saig_chat_send_uploaded")
        with chat_action_col_2:
            clear_chat = st.button("🧹 Clear Chat", key="saig_chat_clear_uploaded")

        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.saig_chat_input_uploaded = ""

        saig_warning_placeholder = st.empty()
        _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
    st.markdown("<div class='section-title'>Logs & Analytics</div>", unsafe_allow_html=True)
    threat_log_placeholder = st.empty()

    with st.spinner("Processing video..."):
        yolo_model, audio_model, class_to_idx, device, yolo_device = load_models()
        st.session_state.model_status = "Loaded"
        audio_predictions = process_audio_chunks(
            video_path=temp_video_path,
            audio_model=audio_model,
            class_to_idx=class_to_idx,
            device=device,
        )

    capture = cv2.VideoCapture(str(temp_video_path))
    if not capture.isOpened():
        st.warning("Invalid or unsupported video")
        return

    audio_history: Deque[str] = deque(maxlen=3)
    object_history: ThreatHistory = []
    latest_frame = None
    saig_chat_send_pending = send_chat
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None
    last_processed_time = 0.0
    frame_counter = 0
    target_interval = 1.0 / max(target_fps, 1)

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
            if frame is None:
                continue

            frame_counter += 1
            if frame_skip > 1 and frame_counter % frame_skip != 0:
                continue

            now = time.time()
            if last_processed_time and (now - last_processed_time) < target_interval:
                continue
            last_processed_time = now
            if frame_counter % 30 == 0:
                _debug_log(debug_enabled, "Detection running")

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

            inference_start = time.time()
            annotated, current_objects, current_detections = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
                inference_size=inference_size,
                copy_frame=False,
            )
            inference_ms = (time.time() - inference_start) * 1000.0

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

            objects_str = ", ".join(sorted(current_objects)) if current_objects else "No objects detected"
            _render_signal_card(
                object_placeholder,
                "Object Intelligence",
                f"🐠 {objects_str}",
                tone="neutral",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            writer.write(annotated)
            latest_frame = annotated
            st.session_state.latest_frame = latest_frame
            st.session_state.latest_detected_objects = sorted(current_objects)

            _handle_saig_chat_send(
                send_requested=saig_chat_send_pending,
                prompt_text=chat_prompt,
                latest_frame=latest_frame,
                detected_objects=st.session_state.get("latest_detected_objects", []),
                warning_placeholder=saig_warning_placeholder,
                debug_enabled=debug_enabled,
            )
            saig_chat_send_pending = False

            frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            frame_elapsed = max(time.time() - start_time, 1e-9)
            fps = 1.0 / frame_elapsed
            _render_performance_metrics(performance_placeholder, fps=fps, inference_ms=inference_ms)

            current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = current_frame / total_frames if total_frames > 0 else 0.0
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
            _render_threat_history(threat_log_placeholder)

        if saig_chat_send_pending:
            _handle_saig_chat_send(
                send_requested=True,
                prompt_text=chat_prompt,
                latest_frame=latest_frame,
                detected_objects=st.session_state.get("latest_detected_objects", []),
                warning_placeholder=saig_warning_placeholder,
                debug_enabled=debug_enabled,
            )
        _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
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
    target_fps: int,
    frame_skip: int,
    inference_size: int,
    debug_enabled: bool,
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
        performance_placeholder = st.empty()
        frame_placeholder = st.empty()
    st.markdown("<div class='section-title'>SaiG Assistant</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("### 🧠 SaiG Assistant")
        chat_history_placeholder = st.empty()
        chat_prompt = st.text_input(
            "Ask SaiG about this scene...",
            key="saig_chat_input_webcam",
            label_visibility="collapsed",
            placeholder="Ask SaiG about this scene...",
        )
        chat_action_col_1, chat_action_col_2 = st.columns([1, 1])
        with chat_action_col_1:
            send_chat = st.button("Send", key="saig_chat_send_webcam")
        with chat_action_col_2:
            clear_chat = st.button("🧹 Clear Chat", key="saig_chat_clear_webcam")

        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.saig_chat_input_webcam = ""

        saig_warning_placeholder = st.empty()
        _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
    st.markdown("<div class='section-title'>Logs & Analytics</div>", unsafe_allow_html=True)
    threat_log_placeholder = st.empty()

    with st.spinner("Processing video..."):
        yolo_model, _, _, _, yolo_device = load_models()
        st.session_state.model_status = "Loaded"

    st.markdown(
        "<div class='glass-card card history-card'>Webcam mode active. Turn off 🔴 Live Mode to end stream.</div>",
        unsafe_allow_html=True,
    )

    capture = _open_webcam_with_retry()
    if capture is None:
        st.warning("Camera not available")
        return
    object_history: ThreatHistory = []
    latest_frame = None
    saig_chat_send_pending = send_chat
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None
    webcam_start_time = time.time()
    last_processed_time = 0.0
    frame_counter = 0
    target_interval = 1.0 / max(target_fps, 1)

    try:
        max_frames = 600
        for _ in range(max_frames):
            if not st.session_state.get("webcam_active", False):
                break

            loop_start = time.time()

            ok, frame = capture.read()
            if not ok:
                st.warning("Failed to read frame from webcam.")
                break
            if frame is None:
                continue

            frame_counter += 1
            if frame_skip > 1 and frame_counter % frame_skip != 0:
                continue

            now = time.time()
            if last_processed_time and (now - last_processed_time) < target_interval:
                continue
            last_processed_time = now
            if frame_counter % 30 == 0:
                _debug_log(debug_enabled, "Detection running")

            inference_start = time.time()
            annotated, current_objects, current_detections = annotate_and_collect_objects(
                frame=frame,
                yolo_model=yolo_model,
                yolo_device=yolo_device,
                conf_threshold=st.session_state.get("confidence_threshold", 0.25),
                inference_size=inference_size,
                copy_frame=False,
            )
            inference_ms = (time.time() - inference_start) * 1000.0

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
            objects_str = ", ".join(sorted(current_objects)) if current_objects else "No objects detected"
            _render_signal_card(
                object_placeholder,
                "Object Intelligence",
                f"🐠 {objects_str}",
                tone="neutral",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            latest_frame = annotated
            st.session_state.latest_frame = latest_frame
            st.session_state.latest_detected_objects = sorted(current_objects)

            _handle_saig_chat_send(
                send_requested=saig_chat_send_pending,
                prompt_text=chat_prompt,
                latest_frame=latest_frame,
                detected_objects=st.session_state.get("latest_detected_objects", []),
                warning_placeholder=saig_warning_placeholder,
                debug_enabled=debug_enabled,
            )
            saig_chat_send_pending = False

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            frame_elapsed = max(time.time() - loop_start, 1e-9)
            fps = 1.0 / frame_elapsed
            _render_performance_metrics(performance_placeholder, fps=fps, inference_ms=inference_ms)
            _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
            _render_threat_history(threat_log_placeholder)

        if saig_chat_send_pending:
            _handle_saig_chat_send(
                send_requested=True,
                prompt_text=chat_prompt,
                latest_frame=latest_frame,
                detected_objects=st.session_state.get("latest_detected_objects", []),
                warning_placeholder=saig_warning_placeholder,
                debug_enabled=debug_enabled,
            )
        _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)
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
    padding-top: 1.6rem;
    padding-bottom: 2.0rem;
}

.section-title {
    color: #00e5ff;
    font-weight: 700;
    margin-top: 1.4rem;
    margin-bottom: 0.9rem;
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
    margin-bottom: 0.8rem;
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

.saig-card {
    border-color: rgba(0, 229, 255, 0.38);
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.22);
}

.saig-chat-shell {
    padding: 10px;
}

.saig-chat-history {
    max-height: 280px;
    overflow-y: auto;
    padding-right: 4px;
}

.chat-row {
    display: flex;
    margin-bottom: 0.45rem;
}

.chat-row-user {
    justify-content: flex-end;
}

.chat-row-assistant {
    justify-content: flex-start;
}

.chat-bubble {
    max-width: 85%;
    border-radius: 14px;
    padding: 10px 12px;
    line-height: 1.35;
    font-size: 0.96rem;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.chat-bubble-user {
    background: rgba(0, 229, 255, 0.16);
    border-color: rgba(0, 229, 255, 0.45);
}

.chat-bubble-assistant {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(79, 195, 247, 0.42);
}

.footer-note {
    text-align: center;
    color: rgba(232, 247, 255, 0.88);
    margin-top: 1.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
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
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    if "latest_detected_objects" not in st.session_state:
        st.session_state.latest_detected_objects = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_status" not in st.session_state:
        st.session_state.model_status = "Loaded" if YOLO_WEIGHTS_PATH.exists() else "Training"
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    st.sidebar.title("⚙️ Controls")
    debug_mode = st.sidebar.toggle("🪵 Debug Logs", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_mode

    startup_status, startup_error = _run_startup_validation(debug_mode)
    _render_system_status_panel(startup_status)
    if startup_error:
        st.error(f"Startup validation failed: {startup_error}")
        st.stop()

    st.sidebar.markdown("### 🎛 Input")
    uploaded_video = st.sidebar.file_uploader("Upload video (mp4)", type=["mp4"])
    webcam_mode = st.sidebar.checkbox("Use webcam mode")
    live_mode = st.sidebar.toggle("🔴 Live Mode", value=st.session_state.webcam_active)
    enable_threat = st.sidebar.toggle(
        "🚨 Enable Threat Detection",
        help="Threat levels: 1-3 low, 4-6 medium, 7-10 high. Alerts stabilize using the slider below.",
    )
    enable_sound = st.sidebar.toggle("🔊 Enable Sound Alert")
    if st.sidebar.button("🧹 Clear Threat Log"):
        st.session_state.threat_log = []
    stability_threshold = st.sidebar.slider(
        "Stability Duration (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Higher values reduce flicker by requiring the same dominant object to persist longer before alerting.",
    )
    st.sidebar.markdown("### 🎯 Detection")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.confidence_threshold),
        step=0.01,
    )
    st.sidebar.markdown("### ⚡ Performance")
    target_fps = st.sidebar.slider(
        "Target Processing FPS",
        min_value=10,
        max_value=30,
        value=DEFAULT_TARGET_FPS,
        step=1,
        help="Limits how often frames are processed to keep the UI responsive.",
    )
    frame_skip = st.sidebar.slider(
        "Process Every Nth Frame",
        min_value=1,
        max_value=4,
        value=DEFAULT_FRAME_SKIP,
        step=1,
        help="Higher values reduce compute by skipping intermediate frames.",
    )
    inference_size = st.sidebar.select_slider(
        "Inference Resolution",
        options=[320, 416, 512, 640],
        value=DEFAULT_INFERENCE_SIZE,
        help="Lower resolution improves speed at some accuracy cost.",
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
                    target_fps=target_fps,
                    frame_skip=frame_skip,
                    inference_size=inference_size,
                    debug_enabled=debug_mode,
                )
            finally:
                temp_video_path.unlink(missing_ok=True)
        elif webcam_mode or live_mode:
            _run_webcam_mode(
                enable_threat=enable_threat,
                enable_sound=enable_sound,
                stability_threshold=stability_threshold,
                target_fps=target_fps,
                frame_skip=frame_skip,
                inference_size=inference_size,
                debug_enabled=debug_mode,
            )
        else:
            st.info("Upload a video or enable live mode")
    except FileNotFoundError as err:
        st.error(f"Missing model or invalid file: {err}")
    except RuntimeError as err:
        st.error(f"Runtime error: {err}")
    except Exception as err:
        st.error(f"Unexpected error: {err}")

    st.markdown("---")
    st.markdown(
        "<div class='footer-note'>Built with YOLOv8, CNN Audio Model, and Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
