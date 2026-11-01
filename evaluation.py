from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    true_labels,
    predicted_labels,
):
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def calculate_far_frr(
    true_labels,
    predicted_labels,
    class_names,
):
    genuine_count = 0
    forged_count = 0

    false_acceptances = 0
    false_rejections = 0

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels,
    ):
        true_class = class_names[
            int(true_label)
        ]

        predicted_class = class_names[
            int(predicted_label)
        ]

        true_is_genuine = (
            true_class.endswith(
                "_Genuine"
            )
        )

        predicted_is_genuine = (
            predicted_class.endswith(
                "_Genuine"
            )
        )

        if true_is_genuine:
            genuine_count += 1

            if predicted_class != true_class:
                false_rejections += 1

        else:
            forged_count += 1

            if predicted_is_genuine:
                false_acceptances += 1

    far = (
        false_acceptances / forged_count
        if forged_count > 0
        else 0.0
    )

    frr = (
        false_rejections / genuine_count
        if genuine_count > 0
        else 0.0
    )

    return {
        "far": float(far),
        "frr": float(frr),
        "false_acceptances": int(
            false_acceptances
        ),
        "false_rejections": int(
            false_rejections
        ),
    }