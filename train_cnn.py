from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


CLASS_NAMES = [
    "person_1_genuine",
    "person_1_forged",
    "person_2_genuine",
    "person_2_forged",
]

CLASS_PATHS = {
    "person_1_genuine": DATASET_DIR / "person_1" / "genuine",
    "person_1_forged": DATASET_DIR / "person_1" / "forged",
    "person_2_genuine": DATASET_DIR / "person_2" / "genuine",
    "person_2_forged": DATASET_DIR / "person_2" / "forged",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images():
    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        folder = CLASS_PATHS[class_name]

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder}"
            )

        files = sorted(
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        )

        if not files:
            raise ValueError(
                f"No images found in: {folder}"
            )

        print(f"{class_name}: {len(files)} images")

        image_paths.extend(
            str(file) for file in files
        )
        labels.extend(
            [label] * len(files)
        )

    return image_paths, labels


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )


image_paths, labels = collect_images()

image_paths = np.array(image_paths)
labels = np.array(labels)

train_paths = []
train_labels = []
validation_paths = []
validation_labels = []

for class_id in range(len(CLASS_NAMES)):
    indices = np.where(labels == class_id)[0]
    np.random.shuffle(indices)

    validation_count = max(1, int(len(indices) * 0.25))

    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]

    train_paths.extend(image_paths[train_indices])
    train_labels.extend(labels[train_indices])

    validation_paths.extend(image_paths[validation_indices])
    validation_labels.extend(labels[validation_indices])

print()
print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(validation_paths)}")

train_dataset = make_dataset(
    train_paths,
    train_labels,
    training=True,
)

validation_dataset = make_dataset(
    validation_paths,
    validation_labels,
)


augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.05, 0.05),
    ],
    name="augmentation",
)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False


inputs = keras.Input(
    shape=IMAGE_SIZE + (3,)
)

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x * 255.0
)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


ARTIFACTS_DIR.mkdir(exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=ARTIFACTS_DIR / "signature_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.000001,
    ),
]


model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1,
)

print()
print(f"Validation accuracy: {accuracy * 100:.2f}%")

with open(
    ARTIFACTS_DIR / "class_names.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        CLASS_NAMES,
        file,
        ensure_ascii=False,
        indent=2,
    )

print()
print("Training completed.")
print(
    f"Model saved to: "
    f"{ARTIFACTS_DIR / 'signature_model.keras'}"
)
print(
    f"Classes saved to: "
    f"{ARTIFACTS_DIR / 'class_names.json'}"
)