import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "predictive_maintenance_dataset.csv"
LIVE_DATASET_PATH = PROJECT_ROOT / "labeled_motor_sensor_readings.csv"
OUTPUT_PATH = Path(__file__).with_name("embedded_model.h")
SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "motor_fault_random_forest.joblib"
ADC_COUNTS_PER_VIBRATION_UNIT = 200
FEATURE_NAMES = ("vibration", "temperature")
RANDOM_STATE = 42
TREE_COUNT = 100
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 2


def _read_dataset(path: Path, dataset_name: str) -> list[tuple[datetime, list[float], int]]:
    samples: list[tuple[datetime, list[float], int]] = []
    with path.open(encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required = {"timestamp", *FEATURE_NAMES, "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{dataset_name} must contain timestamp, vibration, temperature, "
                "and label columns."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
                vibration = float(row["vibration"])
                temperature = float(row["temperature"])
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {dataset_name} values at line {line_number}."
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"{dataset_name} label at line {line_number} must be 0 or 1."
                )
            if not math.isfinite(vibration) or not math.isfinite(temperature):
                raise ValueError(
                    f"Non-finite {dataset_name} feature at line {line_number}."
                )
            samples.append((timestamp, [vibration, temperature], label))
    samples.sort(key=lambda sample: sample[0])
    return samples


def load_historical_dataset() -> tuple[list[list[float]], list[int]]:
    samples = _read_dataset(DATASET_PATH, "Historical dataset")
    features = [sample[1] for sample in samples]
    labels = [sample[2] for sample in samples]
    if not features or set(labels) != {0, 1}:
        raise ValueError(
            "Historical dataset must include samples for both labels 0 and 1."
        )
    return features, labels


def load_historical_machine_ids() -> list[str]:
    samples: list[tuple[datetime, int, str]] = []
    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames is None or not {"timestamp", "machine_id"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Historical dataset must contain timestamp and machine_id.")
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid historical timestamp at line {line_number}."
                ) from error
            machine_id = row["machine_id"].strip()
            if not machine_id:
                raise ValueError(
                    f"Historical machine_id at line {line_number} cannot be empty."
                )
            samples.append((timestamp, line_number, machine_id))
    samples.sort(key=lambda sample: (sample[0], sample[1]))
    machine_ids = [sample[2] for sample in samples]
    if len(machine_ids) < 2 or len(set(machine_ids)) < 2:
        raise ValueError(
            "Leave-one-machine-out evaluation requires at least two machines."
        )
    return machine_ids


def load_labeled_live_dataset() -> tuple[list[list[float]], list[int]]:
    if not LIVE_DATASET_PATH.is_file():
        return [], []
    samples = _read_dataset(LIVE_DATASET_PATH, "Labeled live dataset")
    if not samples:
        raise ValueError(f"Labeled live dataset is empty: {LIVE_DATASET_PATH}")
    return [sample[1] for sample in samples], [sample[2] for sample in samples]


def load_dataset() -> tuple[list[list[float]], list[int]]:
    historical_features, historical_labels = load_historical_dataset()
    live_features, live_labels = load_labeled_live_dataset()
    return (
        historical_features + live_features,
        historical_labels + live_labels,
    )


def append_labeled_live_sample(
    timestamp: str,
    sequence: int,
    vibration_delta: int,
    temperature_c: float,
    label: int,
) -> bool:
    if label not in (0, 1):
        raise ValueError("A confirmed sample label must be 0 (normal) or 1 (fault).")
    if sequence < 0 or vibration_delta < 0 or not math.isfinite(temperature_c):
        raise ValueError("Live sample values must be finite and non-negative.")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("Live sample timestamp must be ISO 8601.") from error

    LIVE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["timestamp", "sequence", "vibration", "temperature", "label"]
    if LIVE_DATASET_PATH.exists() and LIVE_DATASET_PATH.stat().st_size:
        with LIVE_DATASET_PATH.open(encoding="utf-8-sig", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            if reader.fieldnames is None or not set(columns).issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Labeled live dataset has an invalid schema: {LIVE_DATASET_PATH}"
                )
            if any(
                row["timestamp"] == timestamp
                and row["sequence"] == str(sequence)
                for row in reader
            ):
                return False
    write_header = (
        not LIVE_DATASET_PATH.exists() or LIVE_DATASET_PATH.stat().st_size == 0
    )
    with LIVE_DATASET_PATH.open("a", encoding="utf-8", newline="") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "sequence": sequence,
                "vibration": vibration_delta / ADC_COUNTS_PER_VIBRATION_UNIT,
                "temperature": temperature_c,
                "label": label,
            }
        )
    return True


def _format_values(values: list[str]) -> str:
    return ", ".join(values)


def _format_float(value: float) -> str:
    literal = f"{value:.17g}"
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal


def _forest_arrays(model: RandomForestClassifier) -> dict[str, list[Any]]:
    features: list[int] = []
    thresholds: list[str] = []
    left_children: list[int] = []
    right_children: list[int] = []
    fault_fractions: list[str] = []
    tree_roots: list[int] = []

    for estimator in model.estimators_:
        tree = estimator.tree_
        offset = len(features)
        tree_roots.append(offset)
        for node in range(tree.node_count):
            feature = int(tree.feature[node])
            features.append(feature)
            thresholds.append(_format_float(float(tree.threshold[node])))
            left = int(tree.children_left[node])
            right = int(tree.children_right[node])
            left_children.append(offset + left if left >= 0 else -1)
            right_children.append(offset + right if right >= 0 else -1)
            counts = tree.value[node][0]
            fault_fraction = float(counts[1] / counts.sum())
            fault_fractions.append(_format_float(fault_fraction))
    return {
        "features": features,
        "thresholds": thresholds,
        "left_children": left_children,
        "right_children": right_children,
        "fault_fractions": fault_fractions,
        "tree_roots": tree_roots,
    }


def generate_header(model: RandomForestClassifier, features: list[list[float]]) -> None:
    arrays = _forest_arrays(model)
    feature_columns = list(zip(*features))
    minimums = [_format_float(min(values)) for values in feature_columns]
    maximums = [_format_float(max(values)) for values in feature_columns]
    node_count = len(arrays["features"])
    tree_count = len(arrays["tree_roots"])
    header = f"""#pragma once

#include <stdint.h>

namespace MotorFaultModel {{
constexpr uint16_t TREE_COUNT = {tree_count};
constexpr uint16_t NODE_COUNT = {node_count};
constexpr float ADC_COUNTS_PER_VIBRATION_UNIT = {ADC_COUNTS_PER_VIBRATION_UNIT}.0F;
constexpr float VIBRATION_MIN = {minimums[0]};
constexpr float VIBRATION_MAX = {maximums[0]};
constexpr float TEMPERATURE_MIN_C = {minimums[1]};
constexpr float TEMPERATURE_MAX_C = {maximums[1]};
constexpr uint16_t TREE_ROOT[TREE_COUNT] = {{{_format_values([str(value) for value in arrays["tree_roots"]])}}};
constexpr int8_t FEATURE[NODE_COUNT] = {{{_format_values([str(value) for value in arrays["features"]])}}};
constexpr double THRESHOLD[NODE_COUNT] = {{{_format_values(arrays["thresholds"])}}};
constexpr int16_t LEFT_CHILD[NODE_COUNT] = {{{_format_values([str(value) for value in arrays["left_children"]])}}};
constexpr int16_t RIGHT_CHILD[NODE_COUNT] = {{{_format_values([str(value) for value in arrays["right_children"]])}}};
constexpr double FAULT_FRACTION[NODE_COUNT] = {{{_format_values(arrays["fault_fractions"])}}};
}}  // namespace MotorFaultModel
"""
    OUTPUT_PATH.write_text(header, encoding="utf-8")


def train_model() -> dict[str, Any]:
    historical_features, historical_labels = load_historical_dataset()
    machine_ids = load_historical_machine_ids()
    live_features, live_labels = load_labeled_live_dataset()
    if len(machine_ids) != len(historical_features):
        raise ValueError(
            "Historical machine IDs do not align with timestamp-sorted samples."
        )
    validation_model = RandomForestClassifier(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    validation_groups = sorted(set(machine_ids))
    for machine_id in validation_groups:
        training_labels = {
            label
            for label, group in zip(historical_labels, machine_ids)
            if group != machine_id
        }
        if training_labels != {0, 1}:
            raise ValueError(
                "Each leave-one-machine-out training fold must contain normal "
                "and fault samples."
            )
    predictions = cross_val_predict(
        validation_model,
        historical_features,
        historical_labels,
        groups=machine_ids,
        cv=LeaveOneGroupOut(),
        n_jobs=1,
    )
    features, labels = load_dataset()
    metrics: dict[str, Any] = {
        "algorithm": "RandomForestClassifier",
        "features": list(FEATURE_NAMES),
        "historical_samples": len(historical_features),
        "labeled_live_samples": len(live_features),
        "final_training_samples": len(features),
        "validation_strategy": "leave-one-machine-out cross-validation",
        "validation_groups": validation_groups,
        "validation_samples": len(historical_features),
        "fault_samples": sum(labels),
        "normal_samples": len(labels) - sum(labels),
        "random_state": RANDOM_STATE,
        "trees": TREE_COUNT,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "accuracy": float(accuracy_score(historical_labels, predictions)),
        "confusion_matrix": confusion_matrix(
            historical_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            historical_labels,
            predictions,
            labels=[0, 1],
            target_names=["normal", "fault"],
            output_dict=True,
            zero_division=0,
        ),
        "feature_ranges": {
            "vibration": [
                min(row[0] for row in features),
                max(row[0] for row in features),
            ],
            "temperature": [
                min(row[1] for row in features),
                max(row[1] for row in features),
            ],
        },
    }

    final_model = RandomForestClassifier(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    final_model.fit(features, labels)
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, SAVED_MODEL_PATH)
    generate_header(final_model, features)
    return metrics


def main() -> None:
    metrics = train_model()
    report_path = PROJECT_ROOT / "models" / "training_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"Trained Random Forest on {metrics['historical_samples']} historical "
        f"rows and {metrics['labeled_live_samples']} labeled live rows "
        f"({metrics['final_training_samples']} total)."
    )
    print(
        f"Leave-one-machine-out accuracy: {metrics['accuracy']:.2%}; "
        f"confusion matrix [[TN, FP], [FN, TP]]: "
        f"{metrics['confusion_matrix']}."
    )
    print(f"Saved Python model: {SAVED_MODEL_PATH}")
    print(f"Generated ESP32 model: {OUTPUT_PATH}")
    print(f"Saved evaluation metrics: {report_path}")
    print(
        "Features: vibration and temperature. Live vibration is converted from "
        f"ADC delta using {ADC_COUNTS_PER_VIBRATION_UNIT} counts per unit."
    )


if __name__ == "__main__":
    main()
