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
        icon="🧩",
    )
    _render_signal_card(
        col2.empty(),
        "Dataset",
        "✔ Dataset Loaded" if dataset_ok else "✖ Dataset Not Loaded",
        tone="low" if dataset_ok else "neutral",
        icon="🗂",
    )
    _render_signal_card(
        col3.empty(),
        "Classes / SaiG",
        f"✔ Classes: {classes_count} | 🧠 {saig_mode}",
        tone="neutral",
        icon="🧠",
    )


def _render_performance_metrics(placeholder, fps: float, inference_ms: float) -> None:
    """Render live frame performance metrics."""
    _render_signal_card(
        placeholder,
        "FPS",
        f"{fps:.1f}",
        tone="neutral",
        icon="⚡",
        meta=f"Inference {inference_ms:.1f} ms",
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
    previous_user_message = ""
    for message in reversed(chat_history):
        if str(message.get("role", "")).strip() == "user":
            previous_user_message = str(message.get("content", "")).strip()
            break

    object_reasoning_map = {
        "shark": "Detected a shark in the current frame. This species may represent elevated risk depending on distance and movement.",
        "submarine": "A submarine-like object is visible. This may indicate human-made underwater activity and should be monitored.",
        "human": "A human diver appears in the scene. Safety context matters, especially when large predators are nearby.",
        "whale": "A whale-like presence suggests large marine life nearby; typically low immediate threat but worth tracking.",
        "seal": "A seal is visible, which is generally moderate risk unless high-threat objects are also present.",
    }

    if unique_objects:
        object_text = ", ".join(unique_objects)
        scene_description = (
            f"From the latest frame, I can see {object_text}. "
            f"For your question \"{user_prompt}\", this scene suggests a threat level around {level}/10."
        )
        key_reasoning = []
        for obj in unique_objects:
            if obj in object_reasoning_map:
                key_reasoning.append(object_reasoning_map[obj])

        if not key_reasoning:
            if level >= 7:
                key_reasoning.append(
                    "At least one high-risk detection is present, so this frame should be treated as potentially dangerous."
                )
            elif level >= 4:
                key_reasoning.append(
                    "The detections indicate moderate risk, so continued monitoring is recommended."
                )
            else:
                key_reasoning.append(
                    "Only lower-risk objects are visible right now, so the immediate threat appears limited."
                )

        reasoning = " ".join(key_reasoning)
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
    follow_up_context = (
        f" Your previous question was \"{previous_user_message}\", so this answer keeps that context in mind."
        if previous_user_message and previous_user_message != user_prompt
        else ""
    )
    return f"{scene_description} {reasoning}{memory_context}{follow_up_context}"


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
        "Analyze this underwater frame and respond as SaiG.\n\n"
        f"Detected objects: {detected_text}\n"
        f"User request: {user_prompt}\n\n"
        "Explain scene interpretation, potential threat level, and recommended next monitoring action."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are SaiG, an intelligent underwater surveillance AI assistant. "
                "Analyze underwater scenes, marine life, submarines, threats, and explain clearly. "
                "Use the image and detected objects as context."
            ),
        }
    ]
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


def _handle_saig_quick_action(
    quick_requested: bool,
    latest_frame,
    detected_objects: List[str],
    warning_placeholder,
    debug_enabled: bool = False,
) -> None:
    """Run one-click SaiG scene analysis using latest frame and detections."""
    if not quick_requested:
        return

    if latest_frame is None:
        st.session_state.saig_quick_answer = "No visual data available yet"
        return

    quick_prompt = (
        "Provide an immediate scene analysis with detected objects, possible risks, "
        "and the most important monitoring recommendation."
    )
    _append_chat_message("user", "Analyze the current frame.")

    _debug_log(debug_enabled, "SaiG quick frame analysis triggered")
    with st.spinner("Analyzing frame with SaiG..."):
        try:
            encoded_image = encode_frame_to_base64(latest_frame)
            response = query_saig(
                user_prompt=quick_prompt,
                image_base64=encoded_image,
                detected_objects=detected_objects,
                chat_history=st.session_state.chat_history,
            )
            st.session_state.saig_quick_answer = response
            _append_chat_message("assistant", response)
        except Exception as err:
            fallback = _simulated_saig_response(
                user_prompt=quick_prompt,
                detected_objects=detected_objects,
                chat_history=st.session_state.chat_history,
            )
            st.session_state.saig_quick_answer = fallback
            _append_chat_message("assistant", fallback)
            warning_placeholder.warning(f"SaiG quick analysis failed; using fallback. Details: {err}")


def _render_signal_card(
    placeholder,
    title: str,
    value: str,
    tone: str = "neutral",
    icon: str = "",
    meta: str = "",
) -> None:
    """Render one dashboard signal card."""
    icon_html = f"<div class='card-icon'>{icon}</div>" if icon else ""
    meta_html = f"<div class='card-meta'>{meta}</div>" if meta else ""
    placeholder.markdown(
        (
            f"<div class='glass-card card signal-card tone-{tone}'>"
            f"{icon_html}"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-value'>{value}</div>"
            f"{meta_html}"
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


def _render_threat_history_list(log_placeholder) -> None:
    """Render recent threat events list (latest first)."""
    with log_placeholder.container():
        recent_events = st.session_state.threat_log[-10:]
        if not recent_events:
            st.markdown(
                "<div class='glass-card card history-shell'>No threat events logged yet.</div>",
                unsafe_allow_html=True,
            )
        else:
            rows = "".join(
                (
                    "<div class='history-row'>"
                    f"<span class='history-time'>[{event['time']:.1f}s]</span> "
                    f"<span class='history-object'>{event['object']}</span> "
                    f"<span class='history-level'>Level {event['level']}</span>"
                    "</div>"
                )
                for event in reversed(recent_events)
            )
            st.markdown(
                (
                    "<div class='glass-card card history-shell'>"
                    "<div class='history-scroll'>"
                    f"{rows}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def _render_threat_analytics(analytics_placeholder) -> None:
    """Render object-count analytics chart from threat log."""
    with analytics_placeholder.container():
        st.markdown("<div class='glass-card card analytics-shell'>", unsafe_allow_html=True)
        if st.session_state.threat_log:
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
            st.markdown("<div class='card-meta'>No analytics data available yet.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


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
    st.markdown("<div class='section-title'>Detection</div>", unsafe_allow_html=True)
    top_col_1, top_col_2, top_col_3, top_col_4 = st.columns(4)
    with top_col_1:
        audio_placeholder = st.empty()
    with top_col_2:
        object_placeholder = st.empty()
    with top_col_3:
        alert_placeholder = st.empty()
    with top_col_4:
        performance_placeholder = st.empty()

    blink_placeholder = st.empty()
    sound_placeholder = st.empty()

    st.markdown("<div class='section-title'>Detection Feed</div>", unsafe_allow_html=True)
    st.markdown("<div class='video-shell'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='video-title'>🎥 Live Detection Feed</div>", unsafe_allow_html=True)
        video_label_placeholder = st.empty()
        frame_placeholder = st.empty()
        progress_bar = st.progress(0.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>SaiG Assistant</div>", unsafe_allow_html=True)
    quick_action_col_1, quick_action_col_2, quick_action_col_3 = st.columns([2, 3, 2])
    with quick_action_col_1:
        st.empty()
    with quick_action_col_2:
        quick_saig = st.button("🧠 Ask SaiG About This Frame", key="upload_quick_saig", type="primary")
    with quick_action_col_3:
        st.empty()

    quick_answer_placeholder = st.empty()

    st.markdown("### 💬 Continue Conversation with SaiG")
    chat_history_placeholder = st.empty()
    _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)

    chat_col_1, chat_col_2, chat_col_3 = st.columns([6, 1, 1])
    with chat_col_1:
        chat_prompt = st.text_input(
            "Ask SaiG about this scene...",
            key="saig_chat_input_upload",
            label_visibility="collapsed",
            placeholder="Ask SaiG about this scene...",
        )
    with chat_col_2:
        send_chat = st.button("Send", key="saig_chat_send_upload", type="primary")
    with chat_col_3:
        clear_chat = st.button("Clear Chat", key="saig_chat_clear_upload", type="secondary")

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.saig_quick_answer = ""
        st.session_state.saig_chat_input_upload = ""

    warning_placeholder = st.empty()
    if st.session_state.get("saig_quick_answer"):
        quick_answer_placeholder.markdown(
            f"<div class='glass-card card saig-card'>{st.session_state.saig_quick_answer}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Detection Insights</div>", unsafe_allow_html=True)
    bottom_col_1, bottom_col_2 = st.columns(2)
    with bottom_col_1:
        st.markdown("### 📜 Threat History")
        threat_history_placeholder = st.empty()
    with bottom_col_2:
        st.markdown("### 📊 Object Frequency / Threat Trend")
        threat_analytics_placeholder = st.empty()

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
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None
    last_processed_time = 0.0
    frame_counter = 0
    target_interval = 1.0 / max(target_fps, 1)
    quick_saig_pending = quick_saig
    send_chat_pending = send_chat

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
                "Audio Detection",
                f"{smoothed_audio}",
                tone="neutral",
                icon="🔊",
                meta=f"Confidence {current_conf:.2f}",
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
                "Objects Detected",
                objects_str,
                tone="neutral",
                icon="🐠",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            writer.write(annotated)
            latest_frame = annotated
            st.session_state.latest_frame = latest_frame
            st.session_state.latest_detected_objects = sorted(current_objects)

            frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            frame_elapsed = max(time.time() - start_time, 1e-9)
            fps = 1.0 / frame_elapsed
            _render_performance_metrics(performance_placeholder, fps=fps, inference_ms=inference_ms)

            if quick_saig_pending:
                _handle_saig_quick_action(
                    quick_requested=True,
                    latest_frame=latest_frame,
                    detected_objects=st.session_state.latest_detected_objects,
                    warning_placeholder=warning_placeholder,
                    debug_enabled=debug_enabled,
                )
                quick_saig_pending = False
                if st.session_state.get("saig_quick_answer"):
                    quick_answer_placeholder.markdown(
                        f"<div class='glass-card card saig-card'>{st.session_state.saig_quick_answer}</div>",
                        unsafe_allow_html=True,
                    )

            if send_chat_pending:
                _handle_saig_chat_send(
                    send_requested=True,
                    prompt_text=chat_prompt,
                    latest_frame=latest_frame,
                    detected_objects=st.session_state.latest_detected_objects,
                    warning_placeholder=warning_placeholder,
                    debug_enabled=debug_enabled,
                )
                send_chat_pending = False
                _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)

            current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = current_frame / total_frames if total_frames > 0 else 0.0
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            _render_threat_history_list(threat_history_placeholder)
            _render_threat_analytics(threat_analytics_placeholder)
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
    st.markdown("<div class='section-title'>Detection</div>", unsafe_allow_html=True)
    top_col_1, top_col_2, top_col_3, top_col_4 = st.columns(4)
    with top_col_1:
        audio_placeholder = st.empty()
    with top_col_2:
        object_placeholder = st.empty()
    with top_col_3:
        alert_placeholder = st.empty()
    with top_col_4:
        performance_placeholder = st.empty()

    blink_placeholder = st.empty()
    sound_placeholder = st.empty()

    st.markdown("<div class='section-title'>Detection Feed</div>", unsafe_allow_html=True)
    st.markdown("<div class='video-shell'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='video-title'>🎥 Live Detection Feed</div>", unsafe_allow_html=True)
        video_label_placeholder = st.empty()
        frame_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>SaiG Assistant</div>", unsafe_allow_html=True)
    quick_action_col_1, quick_action_col_2, quick_action_col_3 = st.columns([2, 3, 2])
    with quick_action_col_1:
        st.empty()
    with quick_action_col_2:
        quick_saig = st.button("🧠 Ask SaiG About This Frame", key="webcam_quick_saig", type="primary")
    with quick_action_col_3:
        st.empty()

    quick_answer_placeholder = st.empty()

    st.markdown("### 💬 Continue Conversation with SaiG")
    chat_history_placeholder = st.empty()
    _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)

    chat_col_1, chat_col_2, chat_col_3 = st.columns([6, 1, 1])
    with chat_col_1:
        chat_prompt = st.text_input(
            "Ask SaiG about this scene...",
            key="saig_chat_input_webcam",
            label_visibility="collapsed",
            placeholder="Ask SaiG about this scene...",
        )
    with chat_col_2:
        send_chat = st.button("Send", key="saig_chat_send_webcam", type="primary")
    with chat_col_3:
        clear_chat = st.button("Clear Chat", key="saig_chat_clear_webcam", type="secondary")

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.saig_quick_answer = ""
        st.session_state.saig_chat_input_webcam = ""

    warning_placeholder = st.empty()
    if st.session_state.get("saig_quick_answer"):
        quick_answer_placeholder.markdown(
            f"<div class='glass-card card saig-card'>{st.session_state.saig_quick_answer}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Detection Insights</div>", unsafe_allow_html=True)
    bottom_col_1, bottom_col_2 = st.columns(2)
    with bottom_col_1:
        st.markdown("### 📜 Threat History")
        threat_history_placeholder = st.empty()
    with bottom_col_2:
        st.markdown("### 📊 Object Frequency / Threat Trend")
        threat_analytics_placeholder = st.empty()

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
    last_dominant_object: str | None = None
    stable_start_time: float | None = None
    last_alert_object: str | None = None
    webcam_start_time = time.time()
    last_processed_time = 0.0
    frame_counter = 0
    target_interval = 1.0 / max(target_fps, 1)
    quick_saig_pending = quick_saig
    send_chat_pending = send_chat

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
                "Audio Detection",
                "Webcam mode",
                tone="neutral",
                icon="🔊",
                meta="N/A",
            )
            objects_str = ", ".join(sorted(current_objects)) if current_objects else "No objects detected"
            _render_signal_card(
                object_placeholder,
                "Objects Detected",
                objects_str,
                tone="neutral",
                icon="🐠",
            )

            _render_video_label(video_label_placeholder, high_threat_active=high_threat_active)

            latest_frame = annotated
            st.session_state.latest_frame = latest_frame
            st.session_state.latest_detected_objects = sorted(current_objects)

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            frame_elapsed = max(time.time() - loop_start, 1e-9)
            fps = 1.0 / frame_elapsed
            _render_performance_metrics(performance_placeholder, fps=fps, inference_ms=inference_ms)

            if quick_saig_pending:
                _handle_saig_quick_action(
                    quick_requested=True,
                    latest_frame=latest_frame,
                    detected_objects=st.session_state.latest_detected_objects,
                    warning_placeholder=warning_placeholder,
                    debug_enabled=debug_enabled,
                )
                quick_saig_pending = False
                if st.session_state.get("saig_quick_answer"):
                    quick_answer_placeholder.markdown(
                        f"<div class='glass-card card saig-card'>{st.session_state.saig_quick_answer}</div>",
                        unsafe_allow_html=True,
                    )

            if send_chat_pending:
                _handle_saig_chat_send(
                    send_requested=True,
                    prompt_text=chat_prompt,
                    latest_frame=latest_frame,
                    detected_objects=st.session_state.latest_detected_objects,
                    warning_placeholder=warning_placeholder,
                    debug_enabled=debug_enabled,
                )
                send_chat_pending = False
                _render_saig_chat_history(chat_history_placeholder, st.session_state.chat_history)

            _render_threat_history_list(threat_history_placeholder)
            _render_threat_analytics(threat_analytics_placeholder)
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
    background: #F5F6FA;
    color: #4A4A4A;
}

:root {
    --bg: #F5F6FA;
    --surface: #FFFFFF;
    --surface-soft: #EEF0F8;
    --border: #E6E8F0;
    --text-primary: #4A4A4A;
    --text-muted: #707090;
    --accent: #0F0E47;
    --accent-2: #272757;
    --accent-3: #505081;
    --accent-4: #8686AC;
    --success: #5cb85c;
    --warning: #f0ad4e;
    --danger: #d9534f;
    --space-1: 8px;
    --space-2: 16px;
    --space-3: 24px;
    --space-4: 32px;
}

h1 {
    color: #0F0E47;
    font-size: 2.55rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: var(--space-1);
}

h2, h3 {
    color: #272757;
    font-weight: 700;
}

.main .block-container {
    padding-top: var(--space-3);
    padding-bottom: var(--space-4);
    max-width: 1320px;
    padding-left: var(--space-3);
    padding-right: var(--space-3);
}

.dashboard-header {
    text-align: center;
    margin-top: var(--space-1);
    margin-bottom: var(--space-3);
    background: linear-gradient(145deg, #ffffff 0%, #f8f9fe 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: var(--space-3) var(--space-2);
}

.dashboard-subtitle {
    color: #505081;
    font-size: 1rem;
    margin-top: var(--space-1);
    font-weight: 500;
    line-height: 1.45;
}

.section-title {
    color: #272757;
    font-weight: 600;
    margin-top: var(--space-3);
    margin-bottom: var(--space-2);
    font-size: 1.08rem;
}

.card,
.glass-card {
    background: var(--surface);
    border-radius: 12px;
    padding: 14px var(--space-2);
    margin-bottom: var(--space-2);
    border: 1px solid var(--border);
    box-shadow: 0 4px 14px rgba(15, 14, 71, 0.08);
}

.card:hover {
    border-color: #8686AC;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 8px 18px rgba(15, 14, 71, 0.12);
}

.signal-card {
    min-height: 144px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: var(--space-1);
    border-top: 3px solid #0F0E47;
}

.card-icon {
    font-size: 1.22rem;
    line-height: 1;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) .signal-card {
    border-top-color: #8686AC;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) .signal-card {
    border-top-color: #505081;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) .signal-card {
    border-top-color: #272757;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4) .signal-card {
    border-top-color: #0F0E47;
}

.signal-card .card-title,
.threat-card .card-title,
.history-card {
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.card-value {
    margin-top: 0;
    color: #272757;
    font-size: 1.56rem;
    font-weight: 700;
    line-height: 1.2;
}

.card-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
}

.threat-high {
    border-color: var(--danger);
}

.threat-medium {
    border-color: var(--warning);
}

.threat-low {
    border-color: var(--success);
}

.video-label {
    text-align: center;
    font-weight: 700;
    margin-bottom: var(--space-2);
    color: #272757;
}

.video-label-danger {
    border-color: var(--danger);
}

button[kind],
.stButton > button {
    border-radius: 10px;
    border: 1px solid #0F0E47;
    background: #0F0E47;
    color: #ffffff;
    min-height: 40px;
    height: 40px;
    padding: 0.4rem 0.95rem;
    font-weight: 600;
    width: 100%;
}

.stButton > button[kind="secondary"] {
    background: #8686AC;
    border-color: #8686AC;
    color: #FFFFFF;
}

button[kind]:hover,
.stButton > button:hover {
    background: #272757;
    border-color: #272757;
}

.stButton > button[kind="secondary"]:hover {
    background: #505081;
    border-color: #505081;
    color: #FFFFFF;
}

[data-baseweb="slider"],
[data-baseweb="select"],
[data-testid="stToggle"] {
    margin-top: var(--space-1);
    margin-bottom: var(--space-2);
}

label, .stSelectbox label, .stSlider label {
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
}

[data-testid="stTextInput"] input {
    min-height: 40px;
    height: 40px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    color: #272757;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-soft);
    border: 1px dashed var(--border);
    border-radius: 12px;
}

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 5px 14px rgba(15, 14, 71, 0.08);
}

.alert-blink {
    color: var(--danger);
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.35rem;
}

.saig-card {
    border-color: var(--border);
}

.saig-chat-shell {
    padding: 10px 8px;
    background: #FFFFFF;
}

.saig-chat-history {
    height: 340px;
    overflow-y: auto;
    padding-right: 6px;
}

.chat-row {
    display: flex;
    margin-bottom: 10px;
}

.chat-row-user {
    justify-content: flex-end;
}

.chat-row-assistant {
    justify-content: flex-start;
}

.chat-bubble {
    max-width: 82%;
    border-radius: 14px;
    padding: 10px 14px;
    line-height: 1.42;
    font-size: 0.95rem;
    border: 1px solid transparent;
}

.chat-bubble-user {
    background: #0F0E47;
    border-color: #0F0E47;
    color: #ffffff;
}

.chat-bubble-assistant {
    background: #E8EAF8;
    border-color: #DDE1F4;
    color: #272757;
}

[data-testid="column"] > div {
    gap: 10px;
}

.video-label {
    margin-top: var(--space-1);
    margin-bottom: var(--space-2);
}

.controls-shell {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: var(--space-2);
    margin-bottom: var(--space-1);
}

.video-shell {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: var(--space-3);
    margin-top: var(--space-2);
    margin-bottom: var(--space-3);
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(15, 14, 71, 0.06);
    border-top: 3px solid #0F0E47;
}

.video-title {
    text-align: center;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: var(--space-2);
    color: #272757;
}

.history-shell,
.analytics-shell {
    min-height: 320px;
    max-height: 320px;
}

.history-scroll {
    max-height: 280px;
    overflow-y: auto;
}

.history-row {
    display: flex;
    justify-content: space-between;
    gap: var(--space-1);
    padding: 8px 0;
    border-bottom: 1px solid #ECEEF6;
    color: var(--text-primary);
    font-size: 0.9rem;
}

.history-row:last-child {
    border-bottom: none;
}

.history-time {
    color: var(--text-muted);
}

.footer-note {
    text-align: center;
    color: #707090;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}

p,
.stCaption,
.stMarkdown small,
label {
    color: #4A4A4A;
    font-size: 0.98rem;
    line-height: 1.5;
}

section[data-testid="stSidebar"] {
    background: #0F0E47;
    color: #FFFFFF;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] .stButton > button {
    background: #272757;
    border-color: #272757;
    color: #FFFFFF;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #8686AC;
    border-color: #8686AC;
}

@media (max-width: 900px) {
    .main .block-container {
        padding-top: var(--space-2);
        padding-bottom: var(--space-3);
        padding-left: var(--space-2);
        padding-right: var(--space-2);
    }

    .section-title {
        margin-top: var(--space-3);
        margin-bottom: var(--space-2);
    }

    .chat-bubble {
        max-width: 100%;
    }

    .dashboard-header {
        padding: var(--space-2);
    }

    .signal-card {
        min-height: 132px;
    }

    .video-shell {
        padding: var(--space-2);
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            "<div class='dashboard-header'>"
            "<h1>🌊 Underwater AI Monitoring System</h1>"
            "<div class='dashboard-subtitle'>"
            "Real-time Detection • Threat Intelligence • SaiG Assistant"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "webcam_status" not in st.session_state:
        st.session_state.webcam_status = False
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "uploaded_video_signature" not in st.session_state:
        st.session_state.uploaded_video_signature = None
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 0.3
    if "enable_threat" not in st.session_state:
        st.session_state.enable_threat = False
    if "enable_sound" not in st.session_state:
        st.session_state.enable_sound = False
    if "stability_threshold" not in st.session_state:
        st.session_state.stability_threshold = 3.0
    if "target_fps" not in st.session_state:
        st.session_state.target_fps = DEFAULT_TARGET_FPS
    if "frame_skip" not in st.session_state:
        st.session_state.frame_skip = DEFAULT_FRAME_SKIP
    if "inference_size" not in st.session_state:
        st.session_state.inference_size = DEFAULT_INFERENCE_SIZE
    if "threat_log" not in st.session_state:
        st.session_state.threat_log = []
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    if "latest_detected_objects" not in st.session_state:
        st.session_state.latest_detected_objects = []
    if "detected_objects" not in st.session_state:
        st.session_state.detected_objects = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_status" not in st.session_state:
        st.session_state.model_status = "Loaded" if YOLO_WEIGHTS_PATH.exists() else "Training"
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "run_detection" not in st.session_state:
        st.session_state.run_detection = False
    if "saig_quick_answer" not in st.session_state:
        st.session_state.saig_quick_answer = ""

    debug_mode = bool(st.session_state.debug_mode)

    startup_status, startup_error = _run_startup_validation(debug_mode)
    st.markdown("<div class='section-title'>Controls Panel</div>", unsafe_allow_html=True)
    _render_system_status_panel(startup_status)
    if startup_error:
        st.error(f"Startup validation failed: {startup_error}")
        st.stop()

    st.markdown("<div class='controls-shell'>", unsafe_allow_html=True)
    with st.container(border=False):
        controls_left, controls_right = st.columns(2)

        with controls_left:
            st.markdown("### Input Panel")
            uploaded_video = st.file_uploader("Upload video (mp4)", type=["mp4"], key="singlepage_video_upload")
            if uploaded_video is not None:
                uploaded_bytes = uploaded_video.getvalue()
                signature = (uploaded_video.name, len(uploaded_bytes))
                if signature != st.session_state.uploaded_video_signature:
                    old_video_path = st.session_state.video_path
                    if old_video_path:
                        Path(old_video_path).unlink(missing_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_bytes)
                        st.session_state.video_path = str(Path(tmp_file.name))
                        st.session_state.uploaded_video_signature = signature
                    st.success("Video uploaded and ready.")

            st.session_state.webcam_status = st.toggle(
                "🔴 Live camera",
                value=st.session_state.webcam_status,
                key="singlepage_webcam_toggle",
            )

            if st.session_state.video_path:
                st.caption(f"Selected video: {Path(st.session_state.video_path).name}")

            action_col_1, action_col_2, action_col_3 = st.columns([2, 2, 2])
            with action_col_1:
                if st.button("Start Detection", key="singlepage_start_detection", type="primary"):
                    st.session_state.run_detection = True
                    st.session_state.webcam_active = st.session_state.webcam_status
            with action_col_2:
                if st.button("Stop Detection", key="singlepage_stop_detection", type="secondary"):
                    st.session_state.run_detection = False
                    st.session_state.webcam_active = False
            with action_col_3:
                if st.button("Clear Threat Log", key="singlepage_clear_threat_log", type="secondary"):
                    st.session_state.threat_log = []

        with controls_right:
            st.markdown("### Control Settings")
            st.session_state.enable_threat = st.toggle(
                "Enable Threat Detection",
                value=st.session_state.enable_threat,
                key="singlepage_enable_threat",
            )
            with st.expander("⚙ Advanced Settings", expanded=False):
                st.session_state.target_fps = st.slider(
                    "Target FPS",
                    min_value=10,
                    max_value=30,
                    value=int(st.session_state.target_fps),
                    step=1,
                    key="singlepage_target_fps",
                )
                st.session_state.frame_skip = st.slider(
                    "Frame Skip",
                    min_value=1,
                    max_value=4,
                    value=int(st.session_state.frame_skip),
                    step=1,
                    key="singlepage_frame_skip",
                )
                st.session_state.inference_size = st.select_slider(
                    "Inference Size",
                    options=[320, 416, 512, 640],
                    value=int(st.session_state.inference_size),
                    key="singlepage_inference_size",
                )
                st.session_state.enable_sound = st.toggle(
                    "Enable Sound Alert",
                    value=st.session_state.enable_sound,
                    key="singlepage_enable_sound",
                )
                st.session_state.confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.confidence_threshold),
                    step=0.01,
                    key="singlepage_confidence_threshold",
                )
                st.session_state.stability_threshold = st.slider(
                    "Stability Duration (seconds)",
                    min_value=1.0,
                    max_value=5.0,
                    value=float(st.session_state.stability_threshold),
                    step=0.5,
                    key="singlepage_stability_threshold",
                )
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.webcam_active = bool(
        st.session_state.webcam_status and st.session_state.run_detection
    )

    try:
        if st.session_state.run_detection:
            if st.session_state.webcam_status:
                _run_webcam_mode(
                    enable_threat=st.session_state.enable_threat,
                    enable_sound=st.session_state.enable_sound,
                    stability_threshold=float(st.session_state.stability_threshold),
                    target_fps=int(st.session_state.target_fps),
                    frame_skip=int(st.session_state.frame_skip),
                    inference_size=int(st.session_state.inference_size),
                    debug_enabled=debug_mode,
                )
            elif st.session_state.video_path:
                video_path_obj = Path(st.session_state.video_path)
                if not video_path_obj.exists():
                    st.error("Selected video path is missing. Re-upload the file.")
                else:
                    _render_uploaded_video(
                        video_path_obj,
                        enable_threat=st.session_state.enable_threat,
                        enable_sound=st.session_state.enable_sound,
                        stability_threshold=float(st.session_state.stability_threshold),
                        target_fps=int(st.session_state.target_fps),
                        frame_skip=int(st.session_state.frame_skip),
                        inference_size=int(st.session_state.inference_size),
                        debug_enabled=debug_mode,
                    )
            else:
                st.info("Upload a video or enable live camera, then click Start Detection.")
        else:
            st.markdown("<div class='section-title'>Detection Feed</div>", unsafe_allow_html=True)
            st.info("Detection is idle. Configure controls above and click Start Detection.")
            st.markdown("<div class='section-title'>SaiG Assistant</div>", unsafe_allow_html=True)
            st.info("Once detection starts, you can ask SaiG about the latest frame and continue the conversation.")
            st.markdown("<div class='section-title'>Logs / Analytics</div>", unsafe_allow_html=True)
            log_col_1, log_col_2 = st.columns(2)
            with log_col_1:
                st.markdown("### 📜 Threat History")
                _render_threat_history_list(st.empty())
            with log_col_2:
                st.markdown("### 📊 Object Frequency / Threat Trend")
                _render_threat_analytics(st.empty())
    except FileNotFoundError as err:
        st.error(f"Missing model or invalid file: {err}")
    except RuntimeError as err:
        st.error(f"Runtime error: {err}")
    except Exception as err:
        st.error(f"Unexpected error: {err}")

    st.session_state.detected_objects = st.session_state.latest_detected_objects

    st.markdown("---")
    st.markdown(
        "<div class='footer-note'>Built with YOLOv8 • Audio AI • Streamlit • SaiG</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
