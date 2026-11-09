# ESP32 motor monitoring

The PlatformIO firmware reads an ADXL335 accelerometer, an LM35 temperature
sensor, and an LM393 Hall-effect module. It reports JSON telemetry over USB
serial, checks configurable thresholds, and drives a buzzer through a BC547
when a threshold alarm or ML fault prediction is active. A 16x2 LCD is
connected in parallel 4-bit mode (no I2C backpack). The current wiring does
not include a motor relay, contactor, or driver, so the ESP32 cannot stop the
motor.

## Wiring

| Component | Connection |
| --- | --- |
| ADXL335 VCC / GND | ESP32 3V3 / GND |
| ADXL335 X-OUT / Y-OUT / Z-OUT | GPIO 34 / 35 / 32 |
| LM35 Vs / GND / Vout | ESP32 3V3 / GND / GPIO 33 |
| Hall module VCC / GND / DO | ESP32 3V3 / GND / GPIO 25 |
| LCD VSS / VDD | GND / display supply |
| LCD VO | Wiper of a 10 kΩ contrast potentiometer |
| LCD RW | GND |
| LCD RS / E | GPIO 16 / 17 |
| LCD D4 / D5 / D6 / D7 | GPIO 18 / 19 / 21 / 22 |
| BC547 base | GPIO 13 through a 1 kΩ resistor |
| BC547 emitter / collector | GND / buzzer negative |
| Buzzer positive | Supply appropriate for the buzzer |

Use a 3.3V-compatible LCD input or suitable level shifting if required by the
display. Never connect a 5V signal to an ESP32 GPIO. The Hall module must
produce one LM393 DO pulse per shaft revolution by default; change
`HALL_PULSES_PER_REVOLUTION` in firmware if it produces a different number.
The sensor output must not exceed 3.3V. The ESP32 accepts configured RPM
thresholds up to 100,000,000 RPM, but its Hall pulse filter only measures up to
2,000,000 RPM. A faster pulse marks RPM as invalid and trips the RPM alarm.
The default operating range is 10,000–300,000 RPM; below or above it is a
rule-based fault and activates the buzzer. If the measured RPM is unexpected,
stop the motor before checking magnet gap, DO-to-GPIO25 signal, module power,
common ground, and pulses per revolution.

## Train the model and build the firmware

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m motor_faulty_prediction.train_model
pio run
pio run --target upload
```

If PowerShell blocks venv activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. PlatformIO
installs the LCD library specified in `platformio.ini` during the build. The
training command trains a 100-tree Random Forest using the historical CSV and
any operator-labeled rows in `labeled_motor_sensor_readings.csv`. It evaluates
on held-out machines using leave-one-machine-out cross-validation, then saves
the joblib model and metrics under `models\` and generates
`src\motor_faulty_prediction\embedded_model.h` for ESP32 inference. Run it again
after adding confirmed labels, then rebuild and re-upload the ESP32.

For notebook cells in VS Code, choose **Select Kernel** and select the project's
`.venv` Python interpreter. `ipykernel` is included in the project install. If
the notebook is using another interpreter and reports that `ipykernel` is
missing, switch it to `.venv\Scripts\python.exe` or install `ipykernel` into
the interpreter shown in the error.

The ESP32 performs Random Forest inference locally once per second. Its JSON
serial telemetry includes the model label, averaged forest fault score, and
an out-of-training-range flag. The LCD shows the model result alongside
temperature, vibration, and RPM. This same serial stream powers the dashboard;
the dashboard's **ESP32 serial monitor output** panel displays the raw stream.

## Dashboard

Install the requirements if you did not use the editable project install, then
launch the Streamlit dashboard:

```powershell
python -m pip install -r requirements.txt
streamlit run src\motor_faulty_prediction\dashboard.py
```

Select the ESP32 COM port in the sidebar and click **Connect**. The dashboard
shows live readings, ESP32 alarms and ML output, raw serial data, plus PDF and
CSV download buttons. The CSV contains the most recent 500 samples from that
dashboard connection. Use the sidebar controls to send threshold changes to
the ESP32. Close PlatformIO's serial monitor and the serial receiver before
connecting; the USB COM port can only be opened by one program at a time.

The dashboard appends every valid telemetry sample to
`motor_sensor_readings.csv`; the on-screen history is only a rolling view of
the latest 500 samples. Use **Add a confirmed real-data label** to select a
specific captured sample and label it only when the motor condition is
independently verified. Never use the ESP32 prediction or alarm as the label.
Confirmed samples are written to `labeled_motor_sensor_readings.csv` and are
combined with the historical data at the next training run. Collect verified
examples of both classes before retraining. The reported accuracy is a
historical, machine-held-out score and does not establish accuracy on the
physical motor; real-world validation requires labeled readings from that
hardware.

To view decoded readings in a terminal and continuously append a CSV instead
of using the dashboard, run:

```powershell
python -m motor_faulty_prediction.serial_receiver --port COM5
```

Replace `COM5` with the ESP32 COM port. By default the receiver appends all
readings to `motor_sensor_readings.csv` in the current folder. Choose a
different file with `--csv path\to\readings.csv`. This terminal receiver and
the dashboard cannot both connect to the same USB serial port at once.

Default limits are 50–200 °C, 10,000–300,000 RPM, and 0–200 ADC counts of
vibration delta. Vibration and RPM alarm when their measured values are
outside the inclusive configured ranges; RPM zero is therefore a fault.
Temperature and vibration thresholds remain configurable. The ESP32 calibrates the
ADXL335 stationary baseline during startup; keep the motor still during that
calibration. Firmware commands can also be entered through the serial receiver:
temperature thresholds may be configured up to 500 °C and RPM thresholds up
to 100,000,000 RPM. These are configuration limits, not sensor specifications;
the Hall measurement filter remains limited to 2,000,000 RPM.

```text
TEMP_RANGE=50,200
RPM_RANGE=10000,300000
VIB_RANGE=0,200
STATUS
CALIBRATE
```

Limits set at runtime reset when the ESP32 reboots.

## ML output limitations

The Random Forest uses only `vibration` and `temperature` from the historical
CSV and operator-labeled real readings so it can also run on the connected
ESP32 sensors. RPM is not a feature in the historical dataset, so the real-time
RPM fault/normal result is a separate deterministic range rule, not a Random
Forest prediction. Live telemetry is logged; only independently confirmed,
operator-labeled samples are added to Random Forest training. It does
not use `acoustic`, `current`, or `IMF_*`, because the current hardware does
not measure those features. Firmware maps ADXL335 ADC delta to the dataset's
vibration feature using a fixed 200-count-per-unit scale, which has not been
calibrated against a physical vibration reference. The historical temperature
range (about 58–85 °C) does not match the configured 50–200 °C operating range,
so live readings outside the training ranges are marked unvalidated. The RF
fault score is an average of tree class scores, not a calibrated fault
probability. Use representative labeled data from this exact sensor setup and
calibrate its units before relying on model output. The leave-one-machine-out
score measures generalization across the historical dataset's machines, not
across physical ESP32 installations. The dashboard distinguishes threshold alarms and ML estimates; either a
threshold alarm or an in-training-range ML fault prediction drives the buzzer.
ML predictions outside the training range are unvalidated and do not trigger
the buzzer. The buzzer
does not stop the motor; the installed hardware has no motor control interface.
Do not connect a motor directly to an ESP32 GPIO. The LM35 reading is
median-filtered to reduce isolated ADC spikes;
this cannot compensate for incorrect wiring or sensor calibration. ML status
does not replace the independent ESP32 threshold alarms.

If Windows does not show a COM port and Device Manager reports Code 28 for a
CP2102 USB-to-UART adapter, install the
[Silicon Labs CP210x VCP driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers),
then reconnect the board and use its COM port.