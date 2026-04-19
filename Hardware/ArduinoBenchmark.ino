#include <Arduino.h>
#include "SignalProcessor.h"

const unsigned long BAUD_RATE = 115200;
const unsigned int BENCHMARK_ITERATIONS = 2000;
const float SAMPLE_RATE_HZ = 8000.0f;

SignalProcessor processor;

float synthSample(unsigned int index, float frequency_hz, float amplitude) {
    const float phase = 2.0f * PI * frequency_hz * ((float)index / SAMPLE_RATE_HZ);
    return amplitude * sinf(phase);
}

SignalProcessorInput makeSyntheticInput(unsigned int index) {
    SignalProcessorInput input;

    input.sample =
        synthSample(index, 260.0f, 0.65f) +
        synthSample(index, 430.0f, 0.20f) +
        synthSample(index, 60.0f, 0.15f);

    input.microphone_noise_reference = synthSample(index, 60.0f, 0.10f);
    input.sensor_noise_reference = synthSample(index, 180.0f, 0.05f);
    input.random_noise_estimate = synthSample(index, 1200.0f, 0.03f);

    return input;
}

void runBenchmark() {
    unsigned long totalTimeUs = 0;
    unsigned long maxTimeUs = 0;
    volatile float sink = 0.0f;

    for (unsigned int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        const SignalProcessorInput input = makeSyntheticInput(i);
        const unsigned long startUs = micros();
        sink += signal_processor_process(&processor, input);
        const unsigned long elapsedUs = micros() - startUs;

        totalTimeUs += elapsedUs;
        if (elapsedUs > maxTimeUs) {
            maxTimeUs = elapsedUs;
        }
    }

    const float averageTimeUs = (float)totalTimeUs / (float)BENCHMARK_ITERATIONS;
    const float sampleBudgetUs = 1000000.0f / SAMPLE_RATE_HZ;
    const float cpuUsagePercent = (averageTimeUs / sampleBudgetUs) * 100.0f;

    Serial.println("Signal Processor Benchmark");
    Serial.println("--------------------------");
    Serial.print("Iterations: ");
    Serial.println(BENCHMARK_ITERATIONS);
    Serial.print("Sample rate: ");
    Serial.println(SAMPLE_RATE_HZ);
    Serial.print("Average process time (us): ");
    Serial.println(averageTimeUs, 4);
    Serial.print("Max process time (us): ");
    Serial.println(maxTimeUs);
    Serial.print("Per-sample budget (us): ");
    Serial.println(sampleBudgetUs, 4);
    Serial.print("Estimated CPU usage at target sample rate (%): ");
    Serial.println(cpuUsagePercent, 2);
    Serial.print("Accumulator guard: ");
    Serial.println(sink, 6);
}

void setup() {
    Serial.begin(BAUD_RATE);
    while (!Serial) {
    }

    signal_processor_init(&processor, SAMPLE_RATE_HZ, 80.0f, 3400.0f, 60.0f, 20.0f);
    signal_processor_set_reference_gains(&processor, 0.12f, 0.10f);
    signal_processor_set_noise_gate(&processor, 2.5f, 0.15f, 0.10f);

    delay(250);
    runBenchmark();
}

void loop() {
}
