from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LOG_DIRECTORY = BASE_DIR / "verification_logs"
LOG_FILE = LOG_DIRECTORY / "demo_history.csv"


# --------------------------------------------------
# Page configuration
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
            background:
                linear-gradient(
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

        .section-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow:
                0 8px 24px
                rgba(31, 45, 61, 0.08);
            border: 1px solid #e8edf3;
        }

        .section-title {
            color: #172033;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .status-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 22px;
            text-align: center;
            box-shadow:
                0 8px 24px
                rgba(31, 45, 61, 0.08);
            border: 1px solid #e8edf3;
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

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        button[kind="primary"] {
            background-color: #315efb;
            border-radius: 10px;
            border: none;
            min-height: 48px;
            font-size: 16px;
            font-weight: 700;
        }

        button {
            border-radius: 10px !important;
            min-height: 44px;
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
        "confidence": 0.0,
        "processing_time": 0.0,
        "signature_exists": False,
        "canvas_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------
# Demo verification
# --------------------------------------------------

def calculate_demo_score(
    username,
    password,
    signature_image,
):
    """
    This is only a temporary demo score.

    It will be replaced later with the real CNN model.
    """

    image_bytes = signature_image.tobytes()

    source = (
        f"{username}|"
        f"{password}|"
        f"{len(image_bytes)}|"
        f"{image_bytes[:1000]}"
    ).encode(
        "utf-8",
        errors="ignore",
    )

    digest = hashlib.sha256(
        source
    ).hexdigest()

    numeric_value = int(
        digest[:8],
        16,
    )

    score = 70.0 + (
        numeric_value % 2900
    ) / 100.0

    return min(
        round(score, 2),
        99.0,
    )


# --------------------------------------------------
# Save verification history
# --------------------------------------------------

def save_demo_log(
    username,
    result,
    confidence,
    processing_time,
):
    LOG_DIRECTORY.mkdir(
        exist_ok=True
    )

    new_record = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "username": username,
                "result": result,
                "confidence": confidence,
                "processing_time": processing_time,
                "mode": "demo",
            }
        ]
    )

    if LOG_FILE.exists():
        old_records = pd.read_csv(
            LOG_FILE
        )

        all_records = pd.concat(
            [
                old_records,
                new_record,
            ],
            ignore_index=True,
        )
    else:
        all_records = new_record

    all_records.to_csv(
        LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# --------------------------------------------------
# Display result
# --------------------------------------------------

def show_result():
    result = st.session_state.verification_result
    confidence = st.session_state.confidence
    processing_time = st.session_state.processing_time

    if result == "Accepted":
        result_color = "#198754"
    elif result == "Rejected":
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

    result_column, confidence_column, time_column = (
        st.columns(3)
    )

    with result_column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">
                    Result
                </div>
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
                <div class="status-label">
                    Similarity
                </div>
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
                <div class="status-label">
                    Processing Time
                </div>
                <div class="status-value">
                    {processing_time:.3f}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Display history
# --------------------------------------------------

def show_history():
    if not LOG_FILE.exists():
        return

    history = pd.read_csv(
        LOG_FILE
    )

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
        history.tail(10),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():
    apply_custom_style()
    initialize_state()

    st.markdown(
        """
        <div class="main-title">
            Signature Authentication System
        </div>

        <div class="subtitle">
            Secure identity verification using handwritten signatures
        </div>

        <div class="demo-label">
            Demonstration mode is active.
            The AI model will be connected after dataset collection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(
        [1, 1.35],
        gap="large",
    )

    # --------------------------------------------------
    # User information
    # --------------------------------------------------

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

        with st.form("authentication_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
            )

            st.form_submit_button(
                "Continue",
                use_container_width=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # تم تعديل هذا القسم حتى لا تظهر وسوم HTML ككود
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
            "Enter your username and password, then draw your "
            "signature in the signature area."
        )

        st.write(
            "The final version will compare the signature with "
            "the trained CNN model."
        )

    # --------------------------------------------------
    # Signature area
    # --------------------------------------------------

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
            "Draw your signature using the mouse, touch screen, "
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

            # This fixes:
            # RuntimeError: image_data was not requested
            return_image_data=True,

            key=(
                f"signature_canvas_"
                f"{st.session_state.canvas_version}"
            ),
        )

        # Safely read the canvas image
        if canvas_result.image_data is not None:
            image_data = canvas_result.image_data

            if np.any(image_data[:, :, 3] > 0):
                st.session_state.signature_exists = True
            else:
                st.session_state.signature_exists = False

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
            if not username.strip():
                st.error(
                    "Please enter a username."
                )

            elif not password.strip():
                st.error(
                    "Please enter a password."
                )

            elif not st.session_state.signature_exists:
                st.error(
                    "Please draw a signature."
                )

            elif canvas_result.image_data is None:
                st.error(
                    "Signature image is not available."
                )

            else:
                start_time = datetime.now()

                score = calculate_demo_score(
                    username=username,
                    password=password,
                    signature_image=canvas_result.image_data,
                )

                if score >= 80.0:
                    result = "Accepted"
                else:
                    result = "Rejected"

                end_time = datetime.now()

                processing_time = (
                    end_time - start_time
                ).total_seconds()

                st.session_state.verification_result = (
                    result
                )

                st.session_state.confidence = (
                    score
                )

                st.session_state.processing_time = (
                    processing_time
                )

                save_demo_log(
                    username=username,
                    result=result,
                    confidence=score,
                    processing_time=processing_time,
                )

                st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # Result and history
    # --------------------------------------------------

    show_result()
    show_history()

    st.markdown(
        """
        <div class="footer">
            Demonstration interface for the AI signature verification project
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
