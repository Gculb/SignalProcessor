#include "SignalProcessor.h"

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float compute_one_pole_alpha(float cutoff_hz, float sample_rate_hz) {
    if (cutoff_hz <= 0.0f || sample_rate_hz <= 0.0f) {
        return 0.0f;
    }

    return expf(-2.0f * (float)M_PI * cutoff_hz / sample_rate_hz);
}

static void configure_notch(
    SignalProcessor *processor,
    float sample_rate_hz,
    float notch_hz,
    float notch_q
) {
    float omega;
    float alpha;
    float b0;
    float b1;
    float b2;
    float a0;
    float a1;
    float a2;

    processor->notch_b0 = 1.0f;
    processor->notch_b1 = 0.0f;
    processor->notch_b2 = 0.0f;
    processor->notch_a1 = 0.0f;
    processor->notch_a2 = 0.0f;

    if (notch_hz <= 0.0f || notch_q <= 0.0f || sample_rate_hz <= (2.0f * notch_hz)) {
        return;
    }

    omega = 2.0f * (float)M_PI * notch_hz / sample_rate_hz;
    alpha = sinf(omega) / (2.0f * notch_q);

    b0 = 1.0f;
    b1 = -2.0f * cosf(omega);
    b2 = 1.0f;
    a0 = 1.0f + alpha;
    a1 = -2.0f * cosf(omega);
    a2 = 1.0f - alpha;

    processor->notch_b0 = b0 / a0;
    processor->notch_b1 = b1 / a0;
    processor->notch_b2 = b2 / a0;
    processor->notch_a1 = a1 / a0;
    processor->notch_a2 = a2 / a0;
}

float signal_processor_clamp(float value, float minimum, float maximum) {
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

void signal_processor_init(
    SignalProcessor *processor,
    float sample_rate_hz,
    float highpass_hz,
    float lowpass_hz,
    float notch_hz,
    float notch_q
) {
    if (processor == NULL) {
        return;
    }

    processor->sample_rate_hz = sample_rate_hz;

    processor->dc_alpha = compute_one_pole_alpha(10.0f, sample_rate_hz);
    processor->dc_state = 0.0f;
    processor->dc_previous_input = 0.0f;

    processor->ambient_adapt = 0.08f;
    processor->sensor_adapt = 0.08f;
    processor->ambient_reference_state = 0.0f;
    processor->sensor_reference_state = 0.0f;

    processor->hp_alpha = compute_one_pole_alpha(highpass_hz, sample_rate_hz);
    processor->hp_state = 0.0f;
    processor->hp_previous_input = 0.0f;

    processor->lp_alpha = compute_one_pole_alpha(lowpass_hz, sample_rate_hz);
    processor->lp_state = 0.0f;

    processor->notch_x1 = 0.0f;
    processor->notch_x2 = 0.0f;
    processor->notch_y1 = 0.0f;
    processor->notch_y2 = 0.0f;
    configure_notch(processor, sample_rate_hz, notch_hz, notch_q);

    processor->smoother_alpha = 0.20f;
    processor->smoother_state = 0.0f;

    processor->noise_floor_alpha = 0.005f;
    processor->noise_floor = 0.0f;
    processor->gate_threshold = 2.5f;
    processor->gate_release = 0.15f;
    processor->gain_floor = 0.10f;
}

void signal_processor_set_reference_gains(
    SignalProcessor *processor,
    float microphone_reference_gain,
    float sensor_reference_gain
) {
    if (processor == NULL) {
        return;
    }

    processor->ambient_adapt = signal_processor_clamp(microphone_reference_gain, 0.0f, 1.0f);
    processor->sensor_adapt = signal_processor_clamp(sensor_reference_gain, 0.0f, 1.0f);
}

void signal_processor_set_noise_gate(
    SignalProcessor *processor,
    float threshold_multiplier,
    float release,
    float gain_floor
) {
    if (processor == NULL) {
        return;
    }

    processor->gate_threshold = signal_processor_clamp(threshold_multiplier, 1.0f, 10.0f);
    processor->gate_release = signal_processor_clamp(release, 0.0f, 1.0f);
    processor->gain_floor = signal_processor_clamp(gain_floor, 0.0f, 1.0f);
}

static float remove_dc_offset(SignalProcessor *processor, float sample) {
    float filtered = sample - processor->dc_previous_input + processor->dc_alpha * processor->dc_state;
    processor->dc_previous_input = sample;
    processor->dc_state = filtered;
    return filtered;
}

static float subtract_reference_noise(
    float sample,
    float reference_sample,
    float adapt,
    float *reference_state
) {
    float tracked_reference = (0.95f * (*reference_state)) + (0.05f * reference_sample);
    *reference_state = tracked_reference;
    return sample - (adapt * tracked_reference);
}

static float apply_highpass(SignalProcessor *processor, float sample) {
    float output = processor->hp_alpha * (processor->hp_state + sample - processor->hp_previous_input);
    processor->hp_previous_input = sample;
    processor->hp_state = output;
    return output;
}

static float apply_lowpass(SignalProcessor *processor, float sample) {
    float output = (1.0f - processor->lp_alpha) * sample + processor->lp_alpha * processor->lp_state;
    processor->lp_state = output;
    return output;
}

static float apply_notch(SignalProcessor *processor, float sample) {
    float output =
        processor->notch_b0 * sample +
        processor->notch_b1 * processor->notch_x1 +
        processor->notch_b2 * processor->notch_x2 -
        processor->notch_a1 * processor->notch_y1 -
        processor->notch_a2 * processor->notch_y2;

    processor->notch_x2 = processor->notch_x1;
    processor->notch_x1 = sample;
    processor->notch_y2 = processor->notch_y1;
    processor->notch_y1 = output;

    return output;
}

static float suppress_random_noise(
    SignalProcessor *processor,
    float sample,
    float random_noise_estimate
) {
    float noise_magnitude = fabsf(random_noise_estimate);
    float signal_magnitude = fabsf(sample);
    float gain = 1.0f;

    processor->noise_floor =
        ((1.0f - processor->noise_floor_alpha) * processor->noise_floor) +
        (processor->noise_floor_alpha * noise_magnitude);

    if (signal_magnitude < (processor->noise_floor * processor->gate_threshold)) {
        gain = processor->gain_floor;
    } else if (signal_magnitude < (processor->noise_floor * (processor->gate_threshold + 1.0f))) {
        float numerator = signal_magnitude - (processor->noise_floor * processor->gate_threshold);
        float denominator = processor->noise_floor + 1e-6f;
        float ramp = signal_processor_clamp(numerator / denominator, 0.0f, 1.0f);
        gain = processor->gain_floor + ((1.0f - processor->gain_floor) * ramp);
    }

    sample *= gain;
    sample -= 0.5f * random_noise_estimate;

    processor->smoother_state =
        ((1.0f - processor->smoother_alpha) * processor->smoother_state) +
        (processor->smoother_alpha * sample);

    return ((1.0f - processor->gate_release) * sample) +
           (processor->gate_release * processor->smoother_state);
}

float signal_processor_process(
    SignalProcessor *processor,
    SignalProcessorInput input
) {
    float sample;

    if (processor == NULL) {
        return 0.0f;
    }

    sample = input.sample;
    sample = remove_dc_offset(processor, sample);
    sample = subtract_reference_noise(
        sample,
        input.microphone_noise_reference,
        processor->ambient_adapt,
        &processor->ambient_reference_state
    );
    sample = subtract_reference_noise(
        sample,
        input.sensor_noise_reference,
        processor->sensor_adapt,
        &processor->sensor_reference_state
    );
    sample = apply_highpass(processor, sample);
    sample = apply_lowpass(processor, sample);
    sample = apply_notch(processor, sample);
    sample = suppress_random_noise(processor, sample, input.random_noise_estimate);

    return signal_processor_clamp(sample, -1.0f, 1.0f);
}

float signal_processor_process_sample(
    SignalProcessor *processor,
    float raw_sample
) {
    SignalProcessorInput input;

    input.sample = raw_sample;
    input.microphone_noise_reference = 0.0f;
    input.sensor_noise_reference = 0.0f;
    input.random_noise_estimate = 0.0f;

    return signal_processor_process(processor, input);
}
