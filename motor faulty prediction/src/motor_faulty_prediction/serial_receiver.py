import argparse
import csv
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports


def parse_telemetry(
    line: str,
) -> tuple[
    int,
    dict[str, int],
    dict[str, int],
    float,
    int,
    bool,
    dict[str, bool],
    bool,
    int | None,
    float | None,
    bool,
    bool,
]:
    try:
        message: Any = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error

    if not isinstance(message, dict) or message.get("type") != "telemetry":
        raise ValueError("expected a telemetry object")

    sequence = message.get("sequence")
    readings = message.get("accel_adc")
    vibration_delta = message.get("vibration_delta")
    temperature_c = message.get("temperature_c")
    rpm = message.get("rpm")
    lm393_do = message.get("lm393_do")
    alarms = message.get("alarms")
    buzzer_on = message.get("buzzer_on")
    ml_prediction = message.get("ml_prediction")
    ml_fault_fraction = message.get("ml_fault_fraction")
    ml_out_of_range = message.get("ml_out_of_range", False)
    rpm_pulse_too_fast = message.get("rpm_pulse_too_fast", False)
    if type(sequence) is not int or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if not isinstance(readings, dict) or set(readings) != {"x", "y", "z"}:
        raise ValueError("accel_adc must contain x, y, and z readings")
    if not isinstance(vibration_delta, dict) or set(vibration_delta) != {
        "x",
        "y",
        "z",
    }:
        raise ValueError("vibration_delta must contain x, y, and z readings")
    if (
        isinstance(temperature_c, bool)
        or not isinstance(temperature_c, (int, float))
        or not -100 <= temperature_c <= 500
    ):
        raise ValueError("temperature_c must be a valid temperature in Celsius")
    if type(rpm) is not int or rpm < 0:
        raise ValueError("rpm must be a non-negative integer")
    if type(lm393_do) is not bool:
        raise ValueError("lm393_do must be a boolean")
    if not isinstance(alarms, dict) or set(alarms) != {
        "temperature",
        "rpm",
        "vibration",
    } or any(type(value) is not bool for value in alarms.values()):
        raise ValueError("alarms must contain temperature, rpm, and vibration booleans")
    if type(buzzer_on) is not bool:
        raise ValueError("buzzer_on must be a boolean")
    if "ml_prediction" in message or "ml_fault_fraction" in message:
        if type(ml_prediction) is not int or ml_prediction not in (0, 1):
            raise ValueError("ml_prediction must be 0 or 1")
        if (
            isinstance(ml_fault_fraction, bool)
            or not isinstance(ml_fault_fraction, (int, float))
            or not 0 <= ml_fault_fraction <= 1
        ):
            raise ValueError("ml_fault_fraction must be between 0 and 1")
    elif "ml_out_of_range" in message:
        raise ValueError("ML prediction fields must be provided together")
    if type(ml_out_of_range) is not bool:
        raise ValueError("ml_out_of_range must be a boolean")
    if type(rpm_pulse_too_fast) is not bool:
        raise ValueError("rpm_pulse_too_fast must be a boolean")

    validated_readings = {}
    for axis in ("x", "y", "z"):
        value = readings[axis]
        if type(value) is not int or not 0 <= value <= 4095:
            raise ValueError("accel_adc readings must be 12-bit integer values")
        validated_readings[axis] = value

    validated_deltas = {}
    for axis in ("x", "y", "z"):
        value = vibration_delta[axis]
        if type(value) is not int or not 0 <= value <= 4095:
            raise ValueError("vibration_delta readings must be 12-bit integer values")
        validated_deltas[axis] = value

    return (
        sequence,
        validated_readings,
        validated_deltas,
        float(temperature_c),
        rpm,
        lm393_do,
        alarms,
        buzzer_on,
        ml_prediction,
        (
            float(ml_fault_fraction)
            if ml_fault_fraction is not None
            else None
        ),
        ml_out_of_range,
        rpm_pulse_too_fast,
    )


def format_telemetry(
    sequence: int,
    readings: dict[str, int],
    vibration_delta: dict[str, int],
    temperature_c: float,
    rpm: int,
    lm393_do: bool,
    alarms: dict[str, bool],
    buzzer_on: bool,
    ml_prediction: int | None,
    ml_fault_fraction: float | None,
    ml_out_of_range: bool,
    rpm_pulse_too_fast: bool,
) -> str:
    do_level = "HIGH" if lm393_do else "LOW"
    buzzer_state = "ON" if buzzer_on else "OFF"
    if ml_prediction is None or ml_fault_fraction is None:
        ml_status = "unavailable"
    elif ml_out_of_range:
        ml_status = "UNVALIDATED (outside training data)"
    else:
        ml_status = (
            f"{'FAULT' if ml_prediction else 'NORMAL'} "
            f"({ml_fault_fraction:.1%} RF fault score)"
        )
    return (
        f"sample={sequence} | ADXL335 ADC GPIO34 X={readings['x']} "
        f"GPIO35 Y={readings['y']} GPIO32 Z={readings['z']} "
        f"| vibration delta X={vibration_delta['x']} "
        f"Y={vibration_delta['y']} Z={vibration_delta['z']} counts "
        f"| LM35 GPIO33={temperature_c:.1f} C "
        f"| Hall/LM393 GPIO25={rpm} RPM (DO={do_level}) "
        f"| alarms T/R/V={int(alarms['temperature'])}/"
        f"{int(alarms['rpm'])}/{int(alarms['vibration'])} "
        f"| Hall overspeed signal={rpm_pulse_too_fast} "
        f"| buzzer/BC547 GPIO13={buzzer_state} | ML={ml_status}"
    )


def save_csv_row(
    csv_path: Path,
    telemetry: tuple[
        int,
        dict[str, int],
        dict[str, int],
        float,
        int,
        bool,
        dict[str, bool],
        bool,
        int | None,
        float | None,
        bool,
        bool,
    ],
) -> None:
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
        "hall_do",
        "alarm_temperature",
        "alarm_rpm",
        "alarm_vibration",
        "buzzer_on",
        "ml_prediction",
        "ml_fault_fraction",
        "ml_out_of_training_range",
    ]
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sequence": sequence,
        "accel_x_adc": readings["x"],
        "accel_y_adc": readings["y"],
        "accel_z_adc": readings["z"],
        "vibration_x_delta": vibration_delta["x"],
        "vibration_y_delta": vibration_delta["y"],
        "vibration_z_delta": vibration_delta["z"],
        "temperature_c": temperature_c,
        "rpm": rpm,
        "hall_do": "HIGH" if lm393_do else "LOW",
        "alarm_temperature": alarms["temperature"],
        "alarm_rpm": alarms["rpm"],
        "alarm_vibration": alarms["vibration"],
        "buzzer_on": buzzer_on,
        "ml_prediction": ml_prediction,
        "ml_fault_fraction": ml_fault_fraction,
        "ml_out_of_training_range": ml_out_of_range,
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_serial(connection: serial.Serial, csv_path: Path) -> None:
    csv_logging_enabled = True
    while True:
        raw_line = connection.readline()
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            print(f"Skipping invalid serial text: {error}", file=sys.stderr)
            continue
        if not line:
            continue
        if not line.startswith("{"):
            print(f"ESP32: {line}", flush=True)
            continue

        try:
            telemetry = parse_telemetry(line)
        except ValueError as error:
            print(f"ESP32: {line} ({error})", flush=True)
            continue
        if csv_logging_enabled:
            try:
                save_csv_row(csv_path, telemetry)
            except OSError as error:
                print(f"CSV logging error for {csv_path}: {error}", file=sys.stderr)
                csv_logging_enabled = False
        print(format_telemetry(*telemetry), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor ESP32 motor sensors and adjust temperature, RPM, and "
            "vibration limits."
        )
    )
    parser.add_argument("--port", required=True, help="serial port, for example COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("motor_sensor_readings.csv"),
        help="append live telemetry to this CSV file",
    )
    args = parser.parse_args()

    if args.baud <= 0:
        parser.error("--baud must be positive")
    if args.csv.exists() and args.csv.is_dir():
        parser.error("--csv must be a file path, not a directory")

    available_ports = [port.device for port in list_ports.comports()]
    if args.port not in available_ports:
        if available_ports:
            details = f"Available serial ports: {', '.join(available_ports)}."
        else:
            details = (
                "No serial ports were detected. Check the USB connection and driver. "
                "For a CP2102 adapter on Windows, install the Silicon Labs CP210x "
                "VCP driver, then unplug and reconnect the board."
            )
        parser.exit(2, f"Error: {args.port} is not available. {details}\n")

    print(f"Opening ESP32 on {args.port} at {args.baud} baud.")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as connection:
            reader = threading.Thread(
                target=read_serial, args=(connection, args.csv), daemon=True
            )
            reader.start()
            print(
                f"Appending telemetry to {args.csv.resolve()}.\n"
                "Enter TEMP_RANGE=min,max | RPM_RANGE=min,max | "
                "VIB_RANGE=min,max | CALIBRATE | STATUS. Type EXIT to stop."
            )
            while True:
                try:
                    command = input("serial> ").strip()
                except EOFError:
                    break
                if command.upper() in {"EXIT", "QUIT"}:
                    break
                if command:
                    connection.write((command + "\n").encode("ascii"))
    except serial.SerialException as error:
        parser.exit(2, f"Error: could not open {args.port}: {error}\n")
    except UnicodeEncodeError as error:
        parser.exit(2, f"Error: commands must contain ASCII characters: {error}\n")


if __name__ == "__main__":
    main()
