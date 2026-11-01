from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

from utils import preprocess_image


# ============================================================
# Configuration
# ============================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PORT = 5000

# مسار مجلد المشروع الذي يوجد فيه server.py
BASE_DIR = Path(__file__).resolve().parent

# مجلد التوقيع الأصلي
REFERENCE_FOLDER = (
    BASE_DIR
    / "dataset"
    / "person_1"
    / "genuine"
)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# يمكن تعديل القيمة لاحقًا بعد الاختبار
MATCH_THRESHOLD = 0.62


app = Flask(__name__)


# ============================================================
# Reference image
# ============================================================

def find_reference_image():
    """
    البحث عن أول صورة توقيع أصلية داخل:
    dataset/person_1/genuine
    """

    if not REFERENCE_FOLDER.exists():
        raise FileNotFoundError(
            "Reference folder was not found:\n"
            f"{REFERENCE_FOLDER}"
        )

    image_files = sorted(
        [
            file
            for file in REFERENCE_FOLDER.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not image_files:
        raise FileNotFoundError(
            "No reference image was found inside:\n"
            f"{REFERENCE_FOLDER}"
        )

    reference_image = image_files[0]

    print(
        "Using reference image: "
        f"{reference_image}"
    )

    return reference_image


def read_image_from_path(image_path):
    """
    قراءة صورة باستخدام OpenCV.
    يدعم المسارات التي تحتوي على أحرف عربية.
    """

    image_bytes = np.fromfile(
        str(image_path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            "Unable to read image:\n"
            f"{image_path}"
        )

    return image


def load_reference_image():
    """
    تحميل ومعالجة صورة التوقيع الأصلي.
    """

    reference_path = find_reference_image()

    reference_image = read_image_from_path(
        reference_path
    )

    reference_processed = preprocess_image(
        reference_image
    )

    return reference_processed, reference_path


# ============================================================
# Convert online points to image
# ============================================================

def render_points_to_image(points):
    """
    تحويل نقاط التوقيع القادمة من الهاتف
    إلى صورة مؤقتة داخل Python.
    """

    if not points:
        raise ValueError(
            "No signature points were received."
        )

    valid_points = []

    for point in points:
        try:
            x = float(point["x"])
            y = float(point["y"])

            stroke = int(
                point.get("stroke", 1)
            )

            valid_points.append(
                {
                    "x": x,
                    "y": y,
                    "stroke": stroke,
                }
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    if len(valid_points) < 2:
        raise ValueError(
            "At least two valid points are required."
        )

    x_values = [
        point["x"]
        for point in valid_points
    ]

    y_values = [
        point["y"]
        for point in valid_points
    ]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    original_width = max_x - min_x
    original_height = max_y - min_y

    if original_width <= 0:
        original_width = 1.0

    if original_height <= 0:
        original_height = 1.0

    raw_width = 1000
    raw_height = 600

    # خلفية سوداء مؤقتة
    raw_image = np.zeros(
        (
            raw_height,
            raw_width,
        ),
        dtype=np.uint8,
    )

    previous_point = None
    previous_stroke = None

    for point in valid_points:
        normalized_x = (
            point["x"] - min_x
        ) / original_width

        normalized_y = (
            point["y"] - min_y
        ) / original_height

        current_x = int(
            normalized_x * (raw_width - 1)
        )

        current_y = int(
            normalized_y * (raw_height - 1)
        )

        current_stroke = point["stroke"]

        if (
            previous_point is not None
            and previous_stroke == current_stroke
        ):
            cv2.line(
                raw_image,
                previous_point,
                (
                    current_x,
                    current_y,
                ),
                color=255,
                thickness=8,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                raw_image,
                (
                    current_x,
                    current_y,
                ),
                radius=4,
                color=255,
                thickness=-1,
            )

        previous_point = (
            current_x,
            current_y,
        )

        previous_stroke = current_stroke

    # تحويل الصورة إلى خلفية بيضاء وتوقيع أسود
    rendered_image = cv2.bitwise_not(
        raw_image
    )

    return rendered_image


def preprocess_online_points(points):
    """
    معالجة توقيع الهاتف بنفس طريقة
    معالجة صورة التوقيع الأصلي.
    """

    rendered_image = render_points_to_image(
        points
    )

    return preprocess_image(
        rendered_image
    )


# ============================================================
# Similarity
# ============================================================

def calculate_similarity(
    reference,
    candidate,
):
    """
    حساب نسبة التشابه باستخدام Dice.
    """

    reference_mask = (
        reference[:, :, 0] > 0.5
    ).astype(np.float32)

    candidate_mask = (
        candidate[:, :, 0] > 0.5
    ).astype(np.float32)

    intersection = np.sum(
        reference_mask * candidate_mask
    )

    reference_area = np.sum(
        reference_mask
    )

    candidate_area = np.sum(
        candidate_mask
    )

    denominator = (
        reference_area + candidate_area
    )

    if denominator == 0:
        return 0.0

    dice_score = (
        2.0
        * intersection
        / denominator
    )

    return float(
        np.clip(
            dice_score,
            0.0,
            1.0,
        )
    )


def verify_signature(points):
    """
    مقارنة التوقيع القادم من الهاتف
    مع التوقيع الأصلي فقط.
    """

    reference, reference_path = (
        load_reference_image()
    )

    candidate = preprocess_online_points(
        points
    )

    score = calculate_similarity(
        reference,
        candidate,
    )

    if score >= MATCH_THRESHOLD:
        result = "Genuine"
    else:
        result = "Not matching"

    return {
        "result": result,
        "score": round(score, 4),
        "threshold": MATCH_THRESHOLD,
        "reference": str(reference_path),
    }


# ============================================================
# API
# ============================================================

@app.get("/")
def home():
    try:
        reference_path = (
            find_reference_image()
        )

        return jsonify(
            {
                "status": "online",
                "message": (
                    "Signature verification "
                    "server is running."
                ),
                "reference": str(
                    reference_path
                ),
            }
        ), 200

    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


@app.post("/verify")
def verify():
    try:
        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):
            return jsonify(
                {
                    "error": (
                        "Request body must be "
                        "a JSON object."
                    )
                }
            ), 400

        points = data.get("points")

        if not isinstance(points, list):
            return jsonify(
                {
                    "error": (
                        "The JSON must contain "
                        "a points list."
                    )
                }
            ), 400

        result = verify_signature(
            points
        )

        print(
            "Verification result: "
            f"{result}"
        )

        return jsonify(result), 200

    except Exception as error:
        print(
            "Verification error: "
            f"{error}"
        )

        return jsonify(
            {
                "error": str(error)
            }
        ), 500


# ============================================================
# Start server
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Signature verification server")
    print("=" * 60)

    print(
        "Reference folder:"
    )
    print(REFERENCE_FOLDER)

    try:
        reference_path = (
            find_reference_image()
        )

        print(
            "Selected reference image:"
        )
        print(reference_path)

    except Exception as error:
        print(
            "Reference image problem:"
        )
        print(error)

    print(
        "Starting server on port "
        f"{PORT}..."
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
    )