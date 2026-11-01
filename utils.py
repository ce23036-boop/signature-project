from pathlib import Path

import cv2
import numpy as np


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def read_image_from_path(image_path):
    image_path = Path(image_path)

    image_data = np.fromfile(
        str(image_path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


def read_image_from_bytes(image_bytes):
    image_data = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            "Unable to read uploaded image."
        )

    return image


def crop_signature(image):
    if image is None:
        raise ValueError(
            "The image is empty."
        )

    blurred_image = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    _, binary_image = cv2.threshold(
        blurred_image,
        220,
        255,
        cv2.THRESH_BINARY_INV,
    )

    points = cv2.findNonZero(
        binary_image
    )

    if points is None:
        raise ValueError(
            "No signature was detected."
        )

    x, y, width, height = cv2.boundingRect(
        points
    )

    padding = 12

    x_start = max(
        0,
        x - padding,
    )

    y_start = max(
        0,
        y - padding,
    )

    x_end = min(
        binary_image.shape[1],
        x + width + padding,
    )

    y_end = min(
        binary_image.shape[0],
        y + height + padding,
    )

    cropped_image = binary_image[
        y_start:y_end,
        x_start:x_end,
    ]

    return cropped_image


def preprocess_image(image):
    cropped_image = crop_signature(
        image
    )

    resized_image = cv2.resize(
        cropped_image,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    normalized_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )

    processed_image = np.stack(
        [
            normalized_image,
            normalized_image,
            normalized_image,
        ],
        axis=-1,
    )

    return processed_image


def preprocess_image_path(image_path):
    image = read_image_from_path(
        image_path
    )

    return preprocess_image(image)


def preprocess_uploaded_bytes(
    image_bytes,
):
    image = read_image_from_bytes(
        image_bytes
    )

    return preprocess_image(image)


def collect_image_files(dataset_directory):
    dataset_directory = Path(
        dataset_directory
    )

    image_files = []

    for file_path in dataset_directory.rglob(
        "*"
    ):
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            image_files.append(file_path)

    return sorted(image_files)