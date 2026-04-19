#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float sample_rate_hz;

    float dc_alpha;
    float dc_state;
    float dc_previous_input;

    float ambient_adapt;
    float sensor_adapt;
    float ambient_reference_state;
    float sensor_reference_state;

    float hp_alpha;
    float hp_state;
    float hp_previous_input;

    float lp_alpha;
    float lp_state;

    float notch_b0;
    float notch_b1;
    float notch_b2;
    float notch_a1;
    float notch_a2;
    float notch_x1;
    float notch_x2;
    float notch_y1;
    float notch_y2;

    float smoother_alpha;
    float smoother_state;

    float noise_floor_alpha;
    float noise_floor;
    float gate_threshold;
    float gate_release;
    float gain_floor;
} SignalProcessor;

typedef struct {
    float sample;
    float microphone_noise_reference;
    float sensor_noise_reference;
    float random_noise_estimate;
} SignalProcessorInput;

void signal_processor_init(
    SignalProcessor *processor,
    float sample_rate_hz,
    float highpass_hz,
    float lowpass_hz,
    float notch_hz,
    float notch_q
);

void signal_processor_set_reference_gains(
    SignalProcessor *processor,
    float microphone_reference_gain,
    float sensor_reference_gain
);

void signal_processor_set_noise_gate(
    SignalProcessor *processor,
    float threshold_multiplier,
    float release,
    float gain_floor
);

float signal_processor_process(
    SignalProcessor *processor,
    SignalProcessorInput input
);

float signal_processor_process_sample(
    SignalProcessor *processor,
    float raw_sample
);

float signal_processor_clamp(float value, float minimum, float maximum);

#ifdef __cplusplus
}
#endif

#endif
