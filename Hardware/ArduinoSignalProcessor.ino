#include <Arduino.h>
#include "SignalProcessor.h"

const uint8_t MAIN_SIGNAL_PIN = A0;
const uint8_t MIC_NOISE_PIN = A1;
const uint8_t SENSOR_NOISE_PIN = A2;
const uint8_t RANDOM_NOISE_PIN = A3;
const uint8_t OUTPUT_PIN = 9;

SignalProcessor processor;

float analogToUnitFloat(uint8_t pin) {
    const int raw = analogRead(pin);
    return ((float)raw - 512.0f) / 512.0f;
}

int unitFloatToPwm(float sample) {
    const float clamped = signal_processor_clamp(sample, -1.0f, 1.0f);
    const float shifted = (clamped + 1.0f) * 127.5f;
    return (int)shifted;
}

void setup() {
    analogReference(DEFAULT);
    pinMode(OUTPUT_PIN, OUTPUT);

    signal_processor_init(&processor, 8000.0f, 80.0f, 3400.0f, 60.0f, 20.0f);
    signal_processor_set_reference_gains(&processor, 0.12f, 0.10f);
    signal_processor_set_noise_gate(&processor, 2.5f, 0.15f, 0.10f);
}

void loop() {
    SignalProcessorInput input;
    float cleaned;

    input.sample = analogToUnitFloat(MAIN_SIGNAL_PIN);
    input.microphone_noise_reference = analogToUnitFloat(MIC_NOISE_PIN);
    input.sensor_noise_reference = analogToUnitFloat(SENSOR_NOISE_PIN);
    input.random_noise_estimate = analogToUnitFloat(RANDOM_NOISE_PIN);

    cleaned = signal_processor_process(&processor, input);
    analogWrite(OUTPUT_PIN, unitFloatToPwm(cleaned));

    delayMicroseconds(125);
}
