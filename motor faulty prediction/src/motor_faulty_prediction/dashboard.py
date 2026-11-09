import csv
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import serial
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from serial.tools import list_ports

from motor_faulty_prediction.serial_receiver import parse_telemetry
from motor_faulty_prediction.serial_receiver import save_csv_row
from motor_faulty_prediction.train_model import (
    ADC_COUNTS_PER_VIBRATION_UNIT,
    LIVE_DATASET_PATH,
    append_labeled_live_sample,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "predictive_maintenance_dataset.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
LIVE_TELEMETRY_PATH = PROJECT_ROOT / "motor_sensor_readings.csv"
MAX_HISTORY = 500
MAX_CONFIGURABLE_RPM = 100_000_000


def open_esp32_serial_connection(port: str) -> serial.Serial:
    connection = serial.Serial(port=None, baudrate=115200, timeout=0.1)
    connection.dtr = False
    connection.rts = False
    connection.port = port.strip()
    connection.open()
    return connection


@st.cache_data
def read_training_ranges(
    dataset_path: str,
    metrics_path: str,
    metrics_revision: int,
) -> dict[str, tuple[float, float]]:
    metrics_file = Path(metrics_path)
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("Training metrics must be a JSON object.")
        feature_ranges = metrics.get("feature_ranges")
        if not isinstance(feature_ranges, dict):
            raise ValueError("Training metrics do not contain model feature ranges.")
        ranges: dict[str, tuple[float, float]] = {}
        for feature in ("vibration", "temperature"):
            values = feature_ranges.get(feature)
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in values
                )
                or any(not math.isfinite(value) for value in values)
                or values[0] > values[1]
            ):
                raise ValueError(
                    f"Training metrics contain an invalid {feature} feature range."
                )
            ranges[feature] = (float(values[0]), float(values[1]))
        return ranges

    vibration_values: list[float] = []
    temperature_values: list[float] = []

    with Path(dataset_path).open(encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = {"vibration", "temperature"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "The dataset must contain vibration and temperature columns."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                vibration = float(row["vibration"])
                temperature = float(row["temperature"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid vibration or temperature at CSV line {line_number}."
                ) from error
            vibration_values.append(vibration)
            temperature_values.append(temperature)

    if not vibration_values:
        raise ValueError("The dataset contains no training rows.")
    return {
        "vibration": (min(vibration_values), max(vibration_values)),
        "temperature": (min(temperature_values), max(temperature_values)),
    }


def get_out_of_training_range(
    vibration_delta: int,
    temperature_c: float,
    training_ranges: dict[str, tuple[float, float]],
) -> list[str]:
    vibration_feature = vibration_delta / ADC_COUNTS_PER_VIBRATION_UNIT
    out_of_range = []
    if not training_ranges["vibration"][0] <= vibration_feature <= training_ranges[
        "vibration"
    ][1]:
        out_of_range.append("vibration")
    if not training_ranges["temperature"][0] <= temperature_c <= training_ranges[
        "temperature"
    ][1]:
        out_of_range.append("temperature")
    return out_of_range


def create_pdf_report(
    records: list[dict[str, Any]],
    prediction: int | None,
    fault_fraction: float | None,
    out_of_range: list[str],
) -> bytes:
    if not records:
        raise ValueError("There is no sensor data to export.")

    latest = records[-1]
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Motor Fault Prediction - Live Sensor Report", styles["Title"]),
        Paragraph(f"Generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
                  styles["Normal"]),
        Spacer(1, 12),
        Paragraph(
            "Random Forest estimate trained from historical data and any "
            "independently confirmed live labels. ESP32 sensor feature scaling is not calibrated "
            "to the historical dataset. Do not use this output for safety-critical "
            "decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    if prediction is None or fault_fraction is None:
        model_result = "Not available"
    else:
        model_result = "Fault" if prediction else "Normal"
        model_result += f" ({fault_fraction:.1%} RF fault score)"
    if out_of_range:
        model_result += f" (out-of-training-range: {', '.join(out_of_range)})"
    summary = [
        ["Reading", "Value"],
        ["Model estimate", model_result],
        ["Temperature", f"{latest['temperature_c']:.1f} C"],
        ["Vibration delta", f"{latest['vibration_delta']} ADC counts"],
        ["RPM", str(latest["rpm"])],
        [
            "RPM measurement",
            (
                "Invalid: Hall pulse rate exceeded 2,000,000 RPM filter"
                if latest["rpm_pulse_too_fast"]
                else "Valid"
            ),
        ],
        ["ESP32 alarms", ", ".join(latest["active_alarms"]) or "None"],
    ]
    story.append(Table(summary, hAlign="LEFT"))
    story.append(Spacer(1, 16))

    history = [["Time", "Temperature C", "Vibration ADC", "RPM", "ML fault %"]]
    for record in records[-30:]:
        history.append(
            [
                record["timestamp"],
                f"{record['temperature_c']:.1f}",
                str(record["vibration_delta"]),
                str(record["rpm"]),
                (
                    f"{record['fault_fraction']:.1%}"
                    if record["fault_fraction"] is not None
                    else "N/A"
                ),
            ]
        )
    history_table = Table(history, repeatRows=1, hAlign="LEFT")
    history_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ]
        )
    )
    story.extend([Paragraph("Recent sensor history", styles["Heading2"]), history_table])
    document.build(story)
    return output.getvalue()


def create_csv_report(records: list[dict[str, Any]]) -> bytes:
    columns = [
        "timestamp",
        "sequence",
        "accel_x_adc",
        "accel_y_adc",
        "accel_z_adc",
        "vibration_x_delta",
        "vibration_y_delta",
        "vibration_z_delta",
        "temperature_c",
        "rpm",
        "rpm_pulse_too_fast",
        "hall_do",
        "alarm_temperature",
        "alarm_rpm",
        "alarm_vibration",
        "buzzer_on",
        "ml_prediction",
        "ml_fault_fraction",
        "ml_out_of_training_range",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for record in records:
        row = {
            "timestamp": record["timestamp"],
            "sequence": record["sequence"],
            "temperature_c": record["temperature_c"],
            "rpm": record["rpm"],
            "rpm_pulse_too_fast": record["rpm_pulse_too_fast"],
            "hall_do": record["hall_do"],
            "alarm_temperature": "temperature" in record["active_alarms"],
            "alarm_rpm": "rpm" in record["active_alarms"],
            "alarm_vibration": "vibration" in record["active_alarms"],
            "buzzer_on": record["buzzer_on"],
            "ml_prediction": (
                "fault" if record["prediction"] == 1
                else "normal" if record["prediction"] == 0
                else "unavailable"
            ),
            "ml_fault_fraction": record["fault_fraction"],
            "ml_out_of_training_range": record["ml_out_of_range"],
        }
        row.update(
            {f"accel_{axis}_adc": record["accel_adc"][axis] for axis in ("x", "y", "z")}
        )
        row.update(
            {
                f"vibration_{axis}_delta": record["vibration_delta_by_axis"][axis]
                for axis in ("x", "y", "z")
            }
        )
        writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")


def _read_available_telemetry(
    connection: serial.Serial,
    training_ranges: dict[str, tuple[float, float]],
) -> None:
    history: list[dict[str, Any]] = st.session_state.sensor_history
    for _ in range(20):
        if connection.in_waiting <= 0:
            break
        raw_line = connection.readline()
        try:
            line = raw_line.decode("utf-8").strip()
            if line:
                serial_log: list[str] = st.session_state.serial_log
                serial_log.append(line)
                del serial_log[:-50]
            if not line.startswith("{"):
                if line.startswith("ERROR"):
                    st.session_state.last_serial_error = line
                continue
            telemetry = parse_telemetry(line)
        except (UnicodeDecodeError, ValueError) as error:
            st.session_state.last_serial_error = str(error)
            continue
        try:
            save_csv_row(LIVE_TELEMETRY_PATH, telemetry)
        except OSError as error:
            st.session_state.telemetry_log_error = (
                f"Live telemetry could not be saved to {LIVE_TELEMETRY_PATH}: {error}"
            )

        (
            sequence,
            readings,
            vibration_delta,
            temperature_c,
            rpm,
            lm393_do,
            alarms,
            buzzer_on,
            ml_prediction,
            ml_fault_fraction,
            ml_out_of_range,
            rpm_pulse_too_fast,
        ) = telemetry
        maximum_vibration_delta = max(vibration_delta.values())
        previous_sequence = st.session_state.last_sequence
        missed_samples = (
            max(0, sequence - previous_sequence - 1)
            if previous_sequence is not None and sequence > previous_sequence
            else 0
        )
        st.session_state.missed_samples += missed_samples
        st.session_state.last_sequence = sequence
        out_of_range = get_out_of_training_range(
            maximum_vibration_delta,
            temperature_c,
            training_ranges,
        )
        active_alarms = [name for name, active in alarms.items() if active]
        history.append(
            {
                "timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="microseconds"),
                "sequence": sequence,
                "sequence_gap_before": missed_samples,
                "temperature_c": temperature_c,
                "vibration_delta": maximum_vibration_delta,
                "vibration_delta_by_axis": vibration_delta,
                "rpm": rpm,
                "rpm_pulse_too_fast": rpm_pulse_too_fast,
                "hall_do": "HIGH" if lm393_do else "LOW",
                "active_alarms": active_alarms,
                "buzzer_on": buzzer_on,
                "prediction": ml_prediction,
                "fault_fraction": ml_fault_fraction,
                "ml_out_of_range": ml_out_of_range or bool(out_of_range),
                "out_of_range": out_of_range,
                "accel_adc": readings,
            }
        )
        del history[:-MAX_HISTORY]


def _render_live_panel(
    training_ranges: dict[str, tuple[float, float]],
    temperature_range: tuple[float, float],
) -> None:
    connection: serial.Serial | None = st.session_state.serial_connection
    if connection is not None and connection.is_open:
        try:
            _read_available_telemetry(
                connection, training_ranges
            )
        except serial.SerialException as error:
            st.error(f"Lost connection to the ESP32: {error}")
            connection.close()
            st.session_state.serial_connection = None

    history: list[dict[str, Any]] = st.session_state.sensor_history
    if not history:
        st.info("Connect the ESP32 to see live readings.")
        return

    latest = history[-1]
    columns = st.columns(5)
    columns[0].metric("Temperature", f"{latest['temperature_c']:.1f} °C")
    columns[1].metric("Vibration delta", f"{latest['vibration_delta']} ADC")
    columns[2].metric("Speed", f"{latest['rpm']} RPM")
    rpm_is_fault = "rpm" in latest["active_alarms"]
    columns[3].metric("RPM range status", "FAULT" if rpm_is_fault else "NORMAL")
    columns[4].metric(
        "Threshold alarm",
        "ACTIVE" if latest["active_alarms"] else "CLEAR",
    )
    if latest["rpm_pulse_too_fast"]:
        st.error(
            "RPM rule: FAULT. The Hall pulse rate exceeded the ESP32's "
            "2,000,000 RPM measurement-filter limit; the displayed RPM is "
            "not a valid speed measurement."
        )
    elif rpm_is_fault:
        st.warning(
            f"RPM rule: FAULT. Measured speed {latest['rpm']} RPM is outside "
            "the ESP32-configured allowed range."
        )
    if latest["buzzer_on"]:
        alarm_sources = list(latest["active_alarms"])
        if latest["prediction"] and not latest["ml_out_of_range"]:
            alarm_sources.append("ML fault prediction")
        alarm_names = ", ".join(alarm_sources) or "unspecified ESP32 alarm"
        st.warning(
            f"Physical buzzer is ON because the ESP32 reports: {alarm_names}. "
            f"Current readings: {latest['temperature_c']:.1f} °C, "
            f"{latest['vibration_delta']} ADC vibration, {latest['rpm']} RPM. "
            "The buzzer follows threshold alarms; out-of-training-range ML "
            "predictions are not valid fault decisions."
        )
    if "temperature" in latest["active_alarms"]:
        minimum_temperature, maximum_temperature = temperature_range
        st.warning(
            f"Temperature threshold alarm: {latest['temperature_c']:.1f} °C "
            f"is outside the configured {minimum_temperature:.1f}–"
            f"{maximum_temperature:.1f} °C range."
        )
    else:
        st.caption("Physical buzzer: OFF.")
    st.caption(
        "Motor stop control is not connected: this setup has no relay, "
        "contactor, or motor driver. Stop the motor using its normal controls."
    )
    if latest["rpm"] == 0 and not latest["rpm_pulse_too_fast"]:
        st.warning(
            "No Hall pulses were counted. If the motor is rotating, check the "
            "magnet alignment, Hall-module DO-to-GPIO25 wiring, common ground, "
            "and that the Hall output never exceeds 3.3 V."
        )
    elif latest["rpm"] > MAX_CONFIGURABLE_RPM:
        st.error(
            f"The Hall input produced an implausible {latest['rpm']} RPM reading. "
            "Treat RPM as invalid and inspect the Hall wiring and signal noise."
        )

    if latest["prediction"] is None:
        st.info("ML result unavailable. Update the ESP32 firmware to the trained model.")
    elif latest["ml_out_of_range"]:
        st.warning(
            f"ML estimate is UNVALIDATED: the raw model output is "
            f"{'FAULT' if latest['prediction'] else 'NORMAL'} "
            f"({latest['fault_fraction']:.1%} RF fault score), but these sensor "
            "values are outside the model's training range. Do not treat this "
            "estimate as a confirmed fault or normal condition."
        )
    elif latest["prediction"]:
        st.warning(
            f"ESP32 ML result: fault ({latest['fault_fraction']:.1%} "
            "RF fault score)."
        )
    else:
        st.success(
            f"ESP32 ML result: normal ({latest['fault_fraction']:.1%} "
            "RF fault score)."
        )
    if latest["out_of_range"]:
        st.error(
            "Live input outside the CSV training range for "
            f"{', '.join(latest['out_of_range'])}; this prediction is not reliable."
        )
    if st.session_state.missed_samples:
        st.warning(
            f"Serial sequence gaps detected: {st.session_state.missed_samples} "
            "sample(s) were missed since this dashboard connection."
        )
    if st.session_state.last_serial_error:
        st.caption(f"Skipped serial line: {st.session_state.last_serial_error}")
        st.session_state.last_serial_error = ""

    with st.expander("ESP32 serial monitor output"):
        st.code("\n".join(st.session_state.serial_log[-20:]) or "Waiting for serial data")

    st.dataframe(
        [
            {
                "Time": record["timestamp"],
                "Temperature (°C)": record["temperature_c"],
                "Vibration delta (ADC)": record["vibration_delta"],
                "RPM": record["rpm"],
                "Hall DO": record["hall_do"],
                "ESP32 alarms": ", ".join(record["active_alarms"]) or "None",
                "ML output": (
                    "unavailable"
                    if record["prediction"] is None
                    else "fault"
                    if record["prediction"]
                    else "normal"
                ),
                "RF fault score": (
                    f"{record['fault_fraction']:.1%}"
                    if record["fault_fraction"] is not None
                    else "N/A"
                ),
                "ML outside training range": record["ml_out_of_range"],
            }
            for record in reversed(history[-50:])
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Add a confirmed real-data label")
    st.caption(
        "Only label a reading when the motor condition is independently known. "
        "The ESP32 ML result and alarm state are not ground-truth labels. "
        "The Random Forest uses vibration and temperature; RPM range status is "
        "a separate deterministic rule because the historical training data has no RPM feature."
    )
    labelable_records = history[-MAX_HISTORY:]
    labelable_samples = {
        (record["timestamp"], record["sequence"]): record
        for record in labelable_records
    }
    sample_options = list(labelable_samples)
    latest_sample = (
        labelable_records[-1]["timestamp"],
        labelable_records[-1]["sequence"],
    )
    with st.form("label_live_sample"):
        selected_sample = st.selectbox(
            "Sensor sample to label",
            options=sample_options,
            index=sample_options.index(latest_sample),
            format_func=lambda sample_key: (
                f"{labelable_samples[sample_key]['timestamp']} "
                f"(sequence {labelable_samples[sample_key]['sequence']}, "
                f"{labelable_samples[sample_key]['temperature_c']:.1f} C, "
                f"{labelable_samples[sample_key]['vibration_delta']} ADC, "
                f"{labelable_samples[sample_key]['rpm']} RPM)"
            ),
        )
        confirmed_label = st.selectbox(
            "Verified motor condition",
            ["Select a confirmed condition", "Normal", "Fault"],
        )
        save_label = st.form_submit_button("Save confirmed sample")
    if save_label:
        if confirmed_label == "Select a confirmed condition":
            st.error("Select a verified normal or fault condition before saving.")
        else:
            try:
                selected_record = labelable_samples[selected_sample]
                saved = append_labeled_live_sample(
                    timestamp=selected_record["timestamp"],
                    sequence=selected_record["sequence"],
                    vibration_delta=selected_record["vibration_delta"],
                    temperature_c=selected_record["temperature_c"],
                    label=int(confirmed_label == "Fault"),
                )
            except (OSError, ValueError) as error:
                st.error(f"Could not save the confirmed sample: {error}")
            else:
                if saved:
                    st.success(
                        f"Saved {confirmed_label.lower()} label to "
                        f"{LIVE_DATASET_PATH.name}."
                    )
                else:
                    st.info("This sensor sample already has a confirmed label.")
    st.caption(
        "The training command combines confirmed live samples with historical "
        "data. Retrain, rebuild, and upload the firmware to update ESP32 inference."
    )
    if st.session_state.telemetry_log_error:
        st.error(st.session_state.telemetry_log_error)
    st.caption(f"All parsed readings are appended to `{LIVE_TELEMETRY_PATH.name}`.")
    report = create_pdf_report(
        history,
        latest["prediction"],
        latest["fault_fraction"],
        latest["out_of_range"],
    )
    st.download_button(
        "Export latest readings as PDF",
        data=report,
        file_name="motor_fault_prediction_report.pdf",
        mime="application/pdf",
    )
    st.download_button(
        "Export live readings as CSV",
        data=create_csv_report(history),
        file_name="motor_sensor_readings.csv",
        mime="text/csv",
    )


def launch() -> None:
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__)), *sys.argv[1:]]
    streamlit_cli.main()


def main() -> None:
    st.set_page_config(page_title="Motor Fault Dashboard", layout="wide")
    st.title("ESP32 Motor Monitoring Dashboard")
    st.caption("ADXL335 vibration · LM35 temperature · LM393 Hall RPM · ESP32 ML")
    st.warning(
        "The ESP32 runs a Random Forest trained from historical data plus any "
        "manually confirmed live labels. Sensor units are not calibrated to the training "
        "data; out-of-range results are marked unvalidated. This is experimental "
        "and not a safety-rated diagnosis."
    )

    try:
        metrics_revision = (
            METRICS_PATH.stat().st_mtime_ns if METRICS_PATH.is_file() else 0
        )
        training_ranges = read_training_ranges(
            str(DATASET_PATH),
            str(METRICS_PATH),
            metrics_revision,
        )
    except (OSError, ValueError) as error:
        st.error(f"Unable to read model training ranges: {error}")
        st.stop()

    if "serial_connection" not in st.session_state:
        st.session_state.serial_connection = None
    if "sensor_history" not in st.session_state:
        st.session_state.sensor_history = []
    if "serial_log" not in st.session_state:
        st.session_state.serial_log = []
    if "last_serial_error" not in st.session_state:
        st.session_state.last_serial_error = ""
    if "telemetry_log_error" not in st.session_state:
        st.session_state.telemetry_log_error = ""
    if "last_sequence" not in st.session_state:
        st.session_state.last_sequence = None
    if "missed_samples" not in st.session_state:
        st.session_state.missed_samples = 0

    st.sidebar.header("ESP32 connection")
    available_ports = [port.device for port in list_ports.comports()]
    default_port = available_ports[0] if available_ports else ""
    port = st.sidebar.text_input(
        "Serial port",
        value=default_port,
        placeholder="Connect ESP32 to detect its COM port",
    )
    if available_ports:
        st.sidebar.caption(f"Detected serial ports: {', '.join(available_ports)}")
    else:
        st.sidebar.warning(
            "No serial ports detected. Connect the ESP32 with a USB data cable "
            "and check Device Manager."
        )
    connection = st.session_state.serial_connection
    if connection is None or not connection.is_open:
        if st.sidebar.button("Connect", type="primary"):
            selected_port = port.strip().upper()
            if selected_port not in {
                detected_port.upper() for detected_port in available_ports
            }:
                detected = ", ".join(available_ports) or "none"
                st.sidebar.error(
                    f"{selected_port or 'No port selected'} is not currently "
                    f"detected. Detected ports: {detected}."
                )
            else:
                try:
                    st.session_state.serial_connection = open_esp32_serial_connection(
                        selected_port
                    )
                    st.session_state.sensor_history = []
                    st.session_state.serial_log = []
                    st.session_state.last_sequence = None
                    st.session_state.missed_samples = 0
                    st.rerun()
                except (serial.SerialException, ValueError) as error:
                    st.sidebar.error(f"Could not open {selected_port}: {error}")
    elif st.sidebar.button("Disconnect"):
        connection.close()
        st.session_state.serial_connection = None
        st.rerun()

    st.sidebar.caption(
        "Close PlatformIO's serial monitor and the serial receiver before connecting."
    )
    st.sidebar.caption(
        "Model input ranges: vibration "
        f"{training_ranges['vibration'][0]:.3f}–"
        f"{training_ranges['vibration'][1]:.3f}; temperature "
        f"{training_ranges['temperature'][0]:.1f}–"
        f"{training_ranges['temperature'][1]:.1f} °C. "
        f"ESP32 uses {ADC_COUNTS_PER_VIBRATION_UNIT} ADC counts per vibration unit."
    )

    st.sidebar.header("ESP32 alarm thresholds")
    minimum_temperature = st.sidebar.number_input(
        "Minimum temperature (°C)", min_value=-55.0, max_value=500.0,
        value=50.0, step=0.5
    )
    maximum_temperature = st.sidebar.number_input(
        "Maximum temperature (°C)", min_value=-55.0, max_value=500.0,
        value=200.0, step=0.5
    )
    minimum_rpm = st.sidebar.number_input(
        "Minimum RPM",
        min_value=0,
        max_value=MAX_CONFIGURABLE_RPM,
        value=10000,
        step=100,
    )
    maximum_rpm = st.sidebar.number_input(
        "Maximum RPM",
        min_value=1,
        max_value=MAX_CONFIGURABLE_RPM,
        value=300000,
        step=100,
    )
    minimum_vibration = st.sidebar.number_input(
        "Minimum vibration delta (ADC)", min_value=0, max_value=4094, value=0
    )
    maximum_vibration = st.sidebar.number_input(
        "Maximum vibration delta (ADC)", min_value=1, max_value=4095, value=200
    )
    if st.sidebar.button("Apply thresholds"):
        if (
            minimum_temperature >= maximum_temperature
            or minimum_rpm >= maximum_rpm
            or minimum_vibration >= maximum_vibration
        ):
            st.sidebar.error("Each minimum must be lower than its maximum.")
        elif st.session_state.serial_connection is None:
            st.sidebar.error("Connect to the ESP32 before applying thresholds.")
        else:
            commands = (
                f"TEMP_RANGE={minimum_temperature},{maximum_temperature}",
                f"RPM_RANGE={minimum_rpm},{maximum_rpm}",
                f"VIB_RANGE={minimum_vibration},{maximum_vibration}",
            )
            try:
                for command in commands:
                    st.session_state.serial_connection.write(
                        (command + "\n").encode("ascii")
                    )
                st.sidebar.success("Threshold commands sent to the ESP32.")
            except (serial.SerialException, UnicodeEncodeError) as error:
                st.sidebar.error(f"Could not send thresholds: {error}")

    @st.fragment(run_every=1)
    def live_panel() -> None:
        _render_live_panel(
            training_ranges,
            (minimum_temperature, maximum_temperature),
        )

    live_panel()


if __name__ == "__main__":
    main()
