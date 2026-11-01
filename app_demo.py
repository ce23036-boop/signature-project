from datetime import datetime
from pathlib import Path
import hmac
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from supabase import Client, create_client
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "artifacts" / "signature_model.keras"
CLASSES_FILE = BASE_DIR / "artifacts" / "class_names.json"
IMAGE_SIZE = (224, 224)


def load_users():
    try:
        configured_users = st.secrets["users"]
    except (FileNotFoundError, KeyError) as error:
        raise RuntimeError(
            "User codes are not configured. Add them in "
            "Streamlit Cloud Secrets."
        ) from error

    users = {
        str(person): str(code)
        for person, code in configured_users.items()
    }

    if not users:
        raise ValueError("The users section in secrets is empty.")

    return users


def get_admin_code():
    try:
        return str(st.secrets["admin_code"])
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_resource
def get_supabase_client() -> Client:
    try:
        supabase_url = str(st.secrets["SUPABASE_URL"])
        supabase_key = str(st.secrets["SUPABASE_SECRET_KEY"])
    except (FileNotFoundError, KeyError) as error:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and "
            "SUPABASE_SECRET_KEY in Streamlit secrets."
        ) from error

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY cannot be empty."
        )

    return create_client(supabase_url, supabase_key)


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="Signature Authentication",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Custom style
# --------------------------------------------------

def apply_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #f7f9fc 0%,
                #eef3f8 100%
            );
        }

        .main-title {
            color: #172033;
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #687386;
            font-size: 17px;
            text-align: center;
            margin-bottom: 30px;
        }

        .section-card,
        .status-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(31, 45, 61, 0.08);
            border: 1px solid #e8edf3;
        }

        .section-title {
            color: #172033;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .status-card {
            text-align: center;
        }

        .status-label {
            color: #718096;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .status-value {
            color: #172033;
            font-size: 25px;
            font-weight: 800;
        }

        .demo-label {
            background: #fff4d6;
            color: #8a6100;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            font-size: 14px;
            margin-bottom: 20px;
        }

        .footer {
            color: #7a8494;
            text-align: center;
            font-size: 13px;
            margin-top: 35px;
            margin-bottom: 15px;
        }

        button {
            border-radius: 10px !important;
            min-height: 44px;
        }

        button[kind="primary"] {
            background-color: #315efb;
            border: none;
            min-height: 48px;
            font-size: 16px;
            font-weight: 700;
        }

        @media only screen and (max-width: 700px) {
            .main-title {
                font-size: 29px;
            }

            .subtitle {
                font-size: 15px;
            }

            .section-card {
                padding: 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Session state
# --------------------------------------------------

def initialize_state():
    defaults = {
        "verification_result": "Waiting",
        "predicted_class": "-",
        "confidence": 0.0,
        "processing_time": 0.0,
        "signature_exists": False,
        "canvas_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_signature_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASSES_FILE}"
        )

    model = tf.keras.models.load_model(MODEL_FILE)

    with open(CLASSES_FILE, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


# --------------------------------------------------
# Predict signature
# --------------------------------------------------

def predict_signature(signature_image):
    model, class_names = load_signature_model()

    if signature_image.ndim != 3:
        raise ValueError("Invalid signature image.")

    # Convert RGBA to RGB.
    image = signature_image[:, :, :3].astype(np.float32)

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(
        image,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence


# --------------------------------------------------
# Save verification history
# --------------------------------------------------

def save_verification_log(
    selected_person,
    predicted_class,
    result,
    confidence,
    processing_time,
):
    client = get_supabase_client()
    client.table("verification_history").insert(
        {
            "person": selected_person,
            "predicted_class": predicted_class,
            "result": result,
            "confidence": round(confidence, 2),
            "processing_time": round(processing_time, 3),
        }
    ).execute()


# --------------------------------------------------
# Display result
# --------------------------------------------------

def show_result():
    result = st.session_state.verification_result
    predicted_class = st.session_state.predicted_class
    confidence = st.session_state.confidence
    processing_time = st.session_state.processing_time

    if result == "Original":
        result_color = "#198754"
    elif result == "Forged":
        result_color = "#dc3545"
    else:
        result_color = "#172033"

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">
                Verification Result
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_column, confidence_column, time_column = st.columns(3)

    with result_column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Result</div>
                <div class="status-value"
                     style="color: {result_color};">
                    {result}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with confidence_column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Confidence</div>
                <div class="status-value">
                    {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with time_column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Processing Time</div>
                <div class="status-value">
                    {processing_time:.3f}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predicted_class != "-":
        st.info(
            f"Model prediction: {predicted_class}"
        )


# --------------------------------------------------
# Display history
# --------------------------------------------------

def show_history():
    admin_code = get_admin_code()
    if not admin_code:
        return

    entered_code = st.sidebar.text_input(
        "Admin code",
        type="password",
    )

    if not hmac.compare_digest(entered_code, admin_code):
        return

    client = get_supabase_client()
    response = (
        client.table("verification_history")
        .select(
            "created_at, person, predicted_class, result, "
            "confidence, processing_time"
        )
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    rows = response.data or []

    if not rows:
        return

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">
                Verification History
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():
    apply_custom_style()
    initialize_state()
    users = load_users()

    st.markdown(
        """
        <div class="main-title">
            Signature Authentication System
        </div>

        <div class="subtitle">
            Verify whether a signature is original or forged
        </div>

        <div class="demo-label">
            Demonstration mode. Use the test names and codes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(
        [1, 1.35],
        gap="large",
    )

    with left_column:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    User Authentication
                </div>
            """,
            unsafe_allow_html=True,
        )

        selected_person = st.selectbox(
            "Select person",
            options=list(users.keys()),
        )

        code = st.text_input(
            "Access code",
            type="password",
            placeholder="Enter your code",
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    Verification Information
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Select the person, enter the correct code, "
            "then draw the signature."
        )

        st.write(
            "The model will compare the signature with "
            "that person's original and forged signatures."
        )

    with right_column:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    Signature Verification
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Draw the signature using the mouse, touch screen, "
            "or XP-Pen."
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=3,
            stroke_color="#172033",
            background_color="#ffffff",
            height=260,
            width=700,
            drawing_mode="freedraw",
            update_streamlit=True,
            return_image_data=True,
            key=(
                f"signature_canvas_"
                f"{st.session_state.canvas_version}"
            ),
        )

        if canvas_result.image_data is not None:
            image_data = canvas_result.image_data

            st.session_state.signature_exists = bool(
                np.any(image_data[:, :, 3] > 0)
            )

        clear_button = st.button(
            "Clear Signature",
            use_container_width=True,
        )

        if clear_button:
            st.session_state.signature_exists = False
            st.session_state.canvas_version += 1
            st.rerun()

        verify_button = st.button(
            "Verify Signature",
            type="primary",
            use_container_width=True,
        )

        if verify_button:
            expected_code = users[selected_person]

            if not hmac.compare_digest(code, expected_code):
                st.error("Incorrect access code.")

            elif not st.session_state.signature_exists:
                st.error("Please draw a signature.")

            elif canvas_result.image_data is None:
                st.error("Signature image is not available.")

            else:
                start_time = datetime.now()

                try:
                    predicted_class, confidence = (
                        predict_signature(
                            canvas_result.image_data
                        )
                    )

                    genuine_class = (
                        f"{selected_person}_genuine"
                    )

                    if predicted_class == genuine_class:
                        result = "Original"
                    else:
                        result = "Forged"

                    processing_time = (
                        datetime.now() - start_time
                    ).total_seconds()

                    st.session_state.verification_result = result
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.processing_time = processing_time

                    save_verification_log(
                        selected_person=selected_person,
                        predicted_class=predicted_class,
                        result=result,
                        confidence=confidence,
                        processing_time=processing_time,
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Model prediction failed: {error}"
                    )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    show_result()
    show_history()

    st.markdown(
        """
        <div class="footer">
            AI signature verification demonstration
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
