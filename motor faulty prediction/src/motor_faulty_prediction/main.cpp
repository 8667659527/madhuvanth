#include <Arduino.h>
#include <LiquidCrystal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "embedded_model.h"

#ifndef ADXL335_X_PIN
#define ADXL335_X_PIN 34
#endif

#ifndef ADXL335_Y_PIN
#define ADXL335_Y_PIN 35
#endif

#ifndef ADXL335_Z_PIN
#define ADXL335_Z_PIN 32
#endif

#ifndef LM35_PIN
#define LM35_PIN 33
#endif

#ifndef LM393_DO_PIN
#define LM393_DO_PIN 25
#endif

#ifndef BUZZER_DRIVER_PIN
#define BUZZER_DRIVER_PIN 13
#endif

#ifndef LCD_RS_PIN
#define LCD_RS_PIN 16
#endif

#ifndef LCD_ENABLE_PIN
#define LCD_ENABLE_PIN 17
#endif

#ifndef LCD_D4_PIN
#define LCD_D4_PIN 18
#endif

#ifndef LCD_D5_PIN
#define LCD_D5_PIN 19
#endif

#ifndef LCD_D6_PIN
#define LCD_D6_PIN 21
#endif

#ifndef LCD_D7_PIN
#define LCD_D7_PIN 22
#endif

#ifndef HALL_PULSES_PER_REVOLUTION
#define HALL_PULSES_PER_REVOLUTION 1
#endif

constexpr uint32_t SERIAL_BAUD_RATE = 115200;
constexpr uint32_t SAMPLE_INTERVAL_MS = 1000;
constexpr uint16_t CALIBRATION_SAMPLES = 64;
constexpr uint32_t CALIBRATION_SAMPLE_DELAY_MS = 5;
constexpr uint8_t TEMPERATURE_SAMPLES = 17;
constexpr float MAX_CONFIGURABLE_TEMPERATURE_C = 500.0F;
constexpr uint32_t MAX_CONFIGURABLE_RPM = 100000000UL;
constexpr uint32_t MAX_HALL_MEASURABLE_RPM = 2000000UL;
constexpr uint32_t MIN_HALL_PULSE_INTERVAL_US =
    static_cast<uint32_t>(60000000ULL /
                          (static_cast<uint64_t>(MAX_HALL_MEASURABLE_RPM) *
                           HALL_PULSES_PER_REVOLUTION));
static_assert(HALL_PULSES_PER_REVOLUTION > 0,
              "HALL_PULSES_PER_REVOLUTION must be positive");
static_assert(MIN_HALL_PULSE_INTERVAL_US > 0,
              "HALL_PULSES_PER_REVOLUTION is too high for RPM filtering");

float minimumTemperatureC = 50.0F;
float maximumTemperatureC = 200.0F;
uint32_t minimumRpm = 10000;
uint32_t maximumRpm = 300000;
uint16_t minimumVibrationDeltaLimit = 0;
uint16_t maximumVibrationDeltaLimit = 200;

uint32_t sequenceNumber = 0;
uint32_t lastSampleTime = 0;
uint16_t baselineX = 0;
uint16_t baselineY = 0;
uint16_t baselineZ = 0;
volatile uint32_t rpmPulseCount = 0;
volatile uint32_t lastRpmPulseTimeUs = 0;
volatile bool hallPulseTooFast = false;
String serialCommand;
LiquidCrystal lcd(LCD_RS_PIN, LCD_ENABLE_PIN, LCD_D4_PIN, LCD_D5_PIN,LCD_D6_PIN, LCD_D7_PIN);

struct MlResult {
  bool fault;
  float faultFraction;
  bool outOfTrainingRange;
};

void IRAM_ATTR onRpmPulse() {
  const uint32_t pulseTimeUs = micros();
  if (pulseTimeUs - lastRpmPulseTimeUs >= MIN_HALL_PULSE_INTERVAL_US) {
    lastRpmPulseTimeUs = pulseTimeUs;
    rpmPulseCount = rpmPulseCount + 1;
  } else {
    hallPulseTooFast = true;
  }
}

MlResult predictMotorFault(float vibrationFeature, float temperatureC) {
  const bool outOfTrainingRange =
      vibrationFeature < MotorFaultModel::VIBRATION_MIN ||
      vibrationFeature > MotorFaultModel::VIBRATION_MAX ||
      temperatureC < MotorFaultModel::TEMPERATURE_MIN_C ||
      temperatureC > MotorFaultModel::TEMPERATURE_MAX_C;
  const float features[] = {vibrationFeature, temperatureC};
  double faultFractionSum = 0.0;
  for (uint16_t tree = 0; tree < MotorFaultModel::TREE_COUNT; ++tree) {
    int16_t node = MotorFaultModel::TREE_ROOT[tree];
    while (MotorFaultModel::FEATURE[node] >= 0) {
      const uint8_t feature = MotorFaultModel::FEATURE[node];
      node = features[feature] <= MotorFaultModel::THRESHOLD[node]
                 ? MotorFaultModel::LEFT_CHILD[node]
                 : MotorFaultModel::RIGHT_CHILD[node];
    }
    faultFractionSum += MotorFaultModel::FAULT_FRACTION[node];
  }
  const double faultFraction =
      faultFractionSum / static_cast<double>(MotorFaultModel::TREE_COUNT);
  return {faultFraction > 0.5, static_cast<float>(faultFraction),
          outOfTrainingRange};
}

void writeLcdLine(uint8_t row, const char *text) {
  lcd.setCursor(0, row);
  lcd.print(text);
  const size_t length = strlen(text);
  for (size_t column = length; column < 16; ++column) {
    lcd.print(' ');
  }
}

void updateLcd(float temperatureC, uint16_t vibrationDelta, uint32_t rpm,
               bool temperatureAlarm, bool rpmAlarm, bool vibrationAlarm,
               const MlResult &mlResult) {
  char line[17];
  snprintf(line, sizeof(line), "T:%4.1fC V:%3u", temperatureC,
           vibrationDelta);
  writeLcdLine(0, line);

  const char *mlStatus = mlResult.outOfTrainingRange
                             ? "CHECK"
                             : (mlResult.fault ? "FAULT" : "NORM");
  snprintf(line, sizeof(line), "R%4lu ML:%s",
           static_cast<unsigned long>(rpm), mlStatus);
  writeLcdLine(1, line);

  if (temperatureAlarm || rpmAlarm || vibrationAlarm) {
    lcd.setCursor(15, 0);
    lcd.print('!');
  }
}

void calibrateVibrationBaseline() {
  uint32_t sumX = 0;
  uint32_t sumY = 0;
  uint32_t sumZ = 0;

  for (uint16_t sample = 0; sample < CALIBRATION_SAMPLES; ++sample) {
    sumX += analogRead(ADXL335_X_PIN);
    sumY += analogRead(ADXL335_Y_PIN);
    sumZ += analogRead(ADXL335_Z_PIN);
    delay(CALIBRATION_SAMPLE_DELAY_MS);
  }

  baselineX = sumX / CALIBRATION_SAMPLES;
  baselineY = sumY / CALIBRATION_SAMPLES;
  baselineZ = sumZ / CALIBRATION_SAMPLES;
}

float readTemperatureC() {
  uint32_t milliVoltSamples[TEMPERATURE_SAMPLES];
  for (uint8_t sample = 0; sample < TEMPERATURE_SAMPLES; ++sample) {
    milliVoltSamples[sample] = analogReadMilliVolts(LM35_PIN);
    delay(1);
  }
  for (uint8_t sample = 1; sample < TEMPERATURE_SAMPLES; ++sample) {
    const uint32_t value = milliVoltSamples[sample];
    int8_t index = static_cast<int8_t>(sample) - 1;
    while (index >= 0 && milliVoltSamples[index] > value) {
      milliVoltSamples[index + 1] = milliVoltSamples[index];
      --index;
    }
    milliVoltSamples[index + 1] = value;
  }
  return static_cast<float>(milliVoltSamples[TEMPERATURE_SAMPLES / 2]) / 10.0F;
}

bool parseFloatValue(const String &text, float &value) {
  char *end = nullptr;
  value = strtof(text.c_str(), &end);
  while (end && *end == ' ') {
    ++end;
  }
  return end != text.c_str() && end && *end == '\0' && isfinite(value);
}

bool parseUnsignedValue(const String &text, uint32_t &value) {
  if (text.isEmpty()) {
    return false;
  }

  char *end = nullptr;
  const long parsed = strtol(text.c_str(), &end, 10);
  while (end && *end == ' ') {
    ++end;
  }
  if (end == text.c_str() || !end || *end != '\0' || parsed < 0) {
    return false;
  }
  value = static_cast<uint32_t>(parsed);
  return true;
}

void printSettings() {
  Serial.printf(
      "Settings: TEMP_RANGE=%.1f,%.1f C RPM_RANGE=%lu,%lu "
      "VIB_RANGE=%u,%u ADC counts PULSES_PER_REV=%u\n",
      minimumTemperatureC, maximumTemperatureC,
      static_cast<unsigned long>(minimumRpm),
      static_cast<unsigned long>(maximumRpm), minimumVibrationDeltaLimit,
      maximumVibrationDeltaLimit,
      HALL_PULSES_PER_REVOLUTION);
}

void processSerialCommand(String command) {
  command.trim();
  command.toUpperCase();
  if (command.isEmpty()) {
    return;
  }

  if (command == "HELP") {
    Serial.println(
        "Commands: TEMP_RANGE=min,max (C), RPM_RANGE=min,max, "
        "VIB_RANGE=min,max (ADC counts), VIB_DELTA=max (legacy), "
        "CALIBRATE, STATUS");
    return;
  }

  if (command == "STATUS") {
    printSettings();
    return;
  }

  if (command == "CALIBRATE") {
    Serial.println("Keep the motor still; calibrating ADXL335 baseline...");
    calibrateVibrationBaseline();
    Serial.printf("OK baseline X=%u Y=%u Z=%u ADC\n", baselineX, baselineY,
                  baselineZ);
    return;
  }

  const int separator = command.indexOf('=');
  if (separator <= 0) {
    Serial.println("ERROR Invalid command. Type HELP for commands.");
    return;
  }

  const String name = command.substring(0, separator);
  const String value = command.substring(separator + 1);
  const int comma = value.indexOf(',');

  if (name == "TEMP_RANGE") {
    if (comma < 0) {
      Serial.println("ERROR Use TEMP_RANGE=min,max");
      return;
    }
    float newMinimum = 0.0F;
    float newMaximum = 0.0F;
    if (!parseFloatValue(value.substring(0, comma), newMinimum) ||
        !parseFloatValue(value.substring(comma + 1), newMaximum) ||
        newMinimum >= newMaximum || newMinimum < -55.0F ||
        newMaximum > MAX_CONFIGURABLE_TEMPERATURE_C) {
      Serial.println(
          "ERROR Temperature range must be valid, increasing, and within "
          "-55..500 C");
      return;
    }
    minimumTemperatureC = newMinimum;
    maximumTemperatureC = newMaximum;
    Serial.println("OK Temperature range updated.");
    printSettings();
    return;
  }

  if (name == "RPM_RANGE") {
    if (comma < 0) {
      Serial.println("ERROR Use RPM_RANGE=min,max");
      return;
    }
    uint32_t newMinimum = 0;
    uint32_t newMaximum = 0;
    if (!parseUnsignedValue(value.substring(0, comma), newMinimum) ||
        !parseUnsignedValue(value.substring(comma + 1), newMaximum) ||
        newMinimum >= newMaximum || newMaximum > MAX_CONFIGURABLE_RPM) {
      Serial.println(
            "ERROR RPM range must be increasing and maximum must be <=100000000");
      return;
    }
    minimumRpm = newMinimum;
    maximumRpm = newMaximum;
    Serial.println("OK RPM range updated.");
    printSettings();
    return;
  }

  if (name == "VIB_RANGE") {
    if (comma < 0) {
      Serial.println("ERROR Use VIB_RANGE=min,max");
      return;
    }
    uint32_t newMinimum = 0;
    uint32_t newMaximum = 0;
    if (!parseUnsignedValue(value.substring(0, comma), newMinimum) ||
        !parseUnsignedValue(value.substring(comma + 1), newMaximum) ||
        newMinimum >= newMaximum || newMaximum > 4095) {
      Serial.println(
          "ERROR Vibration range must be increasing within 0..4095 ADC counts");
      return;
    }
    minimumVibrationDeltaLimit = static_cast<uint16_t>(newMinimum);
    maximumVibrationDeltaLimit = static_cast<uint16_t>(newMaximum);
    Serial.println("OK Vibration range updated.");
    printSettings();
    return;
  }

  if (name == "VIB_DELTA") {
    uint32_t newThreshold = 0;
    if (!parseUnsignedValue(value, newThreshold) || newThreshold == 0 ||
        newThreshold > 4095) {
      Serial.println("ERROR VIB_DELTA must be between 1 and 4095 ADC counts");
      return;
    }
    minimumVibrationDeltaLimit = 0;
    maximumVibrationDeltaLimit = static_cast<uint16_t>(newThreshold);
    Serial.println("OK Vibration threshold updated.");
    printSettings();
    return;
  }

  Serial.println("ERROR Unknown command. Type HELP for commands.");
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\n' || character == '\r') {
      if (!serialCommand.isEmpty()) {
        processSerialCommand(serialCommand);
        serialCommand = "";
      }
    } else if (serialCommand.length() < 80) {
      serialCommand += character;
    } else {
      serialCommand = "";
      Serial.println("ERROR Command too long.");
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  analogReadResolution(12);
  analogSetPinAttenuation(ADXL335_X_PIN, ADC_11db);
  analogSetPinAttenuation(ADXL335_Y_PIN, ADC_11db);
  analogSetPinAttenuation(ADXL335_Z_PIN, ADC_11db);
  analogSetPinAttenuation(LM35_PIN, ADC_11db);

  lcd.begin(16, 2);
  writeLcdLine(0, "Motor monitor");
  writeLcdLine(1, "Calibrating...");

  pinMode(ADXL335_X_PIN, INPUT);
  pinMode(ADXL335_Y_PIN, INPUT);
  pinMode(ADXL335_Z_PIN, INPUT);
  pinMode(LM35_PIN, INPUT);
  pinMode(LM393_DO_PIN, INPUT_PULLUP);
  pinMode(BUZZER_DRIVER_PIN, OUTPUT);
  digitalWrite(BUZZER_DRIVER_PIN, LOW);
  attachInterrupt(digitalPinToInterrupt(LM393_DO_PIN), onRpmPulse, FALLING);

  Serial.println("Calibrating ADXL335; keep the motor still...");
  calibrateVibrationBaseline();
  Serial.printf("Baseline X=%u Y=%u Z=%u ADC\n", baselineX, baselineY,
                baselineZ);
  printSettings();
  Serial.println("Type HELP for runtime threshold commands.");
  lastSampleTime = millis();
}

void loop() {
  readSerialCommands();

  const uint32_t now = millis();
  const uint32_t elapsedMs = now - lastSampleTime;
  if (elapsedMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTime = now;

  const uint16_t xReading = analogRead(ADXL335_X_PIN);
  const uint16_t yReading = analogRead(ADXL335_Y_PIN);
  const uint16_t zReading = analogRead(ADXL335_Z_PIN);
  const uint16_t xDelta = abs(static_cast<int>(xReading) - baselineX);
  const uint16_t yDelta = abs(static_cast<int>(yReading) - baselineY);
  const uint16_t zDelta = abs(static_cast<int>(zReading) - baselineZ);
  const uint16_t maximumVibrationDelta = max(xDelta, max(yDelta, zDelta));
  const float temperatureC = readTemperatureC();

  noInterrupts();
  const uint32_t pulses = rpmPulseCount;
  rpmPulseCount = 0;
  const bool rpmPulseTooFast = hallPulseTooFast;
  hallPulseTooFast = false;
  interrupts();
  const uint32_t rpm =
      static_cast<uint32_t>(
          (static_cast<uint64_t>(pulses) * 60000ULL) /
          (static_cast<uint64_t>(elapsedMs) * HALL_PULSES_PER_REVOLUTION));

  const bool temperatureOutOfRange =
      temperatureC < minimumTemperatureC || temperatureC > maximumTemperatureC;
  const bool rpmOutOfRange =
      rpmPulseTooFast || rpm < minimumRpm || rpm > maximumRpm;
  const bool vibrationOutOfRange =
      maximumVibrationDelta < minimumVibrationDeltaLimit ||
      maximumVibrationDelta > maximumVibrationDeltaLimit;
  const float vibrationFeature =
      static_cast<float>(maximumVibrationDelta) /
      MotorFaultModel::ADC_COUNTS_PER_VIBRATION_UNIT;
  const MlResult mlResult = predictMotorFault(vibrationFeature, temperatureC);
  const bool thresholdAlarm =
      temperatureOutOfRange || rpmOutOfRange || vibrationOutOfRange;
  const bool validatedMlFault = mlResult.fault && !mlResult.outOfTrainingRange;
  const bool buzzerOn = thresholdAlarm || validatedMlFault;

  digitalWrite(BUZZER_DRIVER_PIN, buzzerOn ? HIGH : LOW);
  updateLcd(temperatureC, maximumVibrationDelta, rpm,
            temperatureOutOfRange, rpmOutOfRange, vibrationOutOfRange,
            mlResult);

  Serial.printf(
      "{\"type\":\"telemetry\",\"sequence\":%lu,\"accel_adc\":{"
      "\"x\":%u,\"y\":%u,\"z\":%u},\"vibration_delta\":{"
      "\"x\":%u,\"y\":%u,\"z\":%u},\"temperature_c\":%.1f,"
      "\"rpm\":%lu,\"lm393_do\":%s,\"alarms\":{"
      "\"temperature\":%s,\"rpm\":%s,\"vibration\":%s},"
      "\"rpm_pulse_too_fast\":%s,"
      "\"buzzer_on\":%s,\"ml_prediction\":%u,"
      "\"ml_fault_fraction\":%.4f,\"ml_out_of_range\":%s}\n",
      static_cast<unsigned long>(sequenceNumber++), xReading, yReading,
      zReading, xDelta, yDelta, zDelta, temperatureC,
      static_cast<unsigned long>(rpm),
      digitalRead(LM393_DO_PIN) == HIGH ? "true" : "false",
      temperatureOutOfRange ? "true" : "false",
      rpmOutOfRange ? "true" : "false",
      vibrationOutOfRange ? "true" : "false",
      rpmPulseTooFast ? "true" : "false",
      buzzerOn ? "true" : "false", mlResult.fault ? 1 : 0,
      mlResult.faultFraction,
      mlResult.outOfTrainingRange ? "true" : "false");
}
