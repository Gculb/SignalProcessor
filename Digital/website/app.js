const API_BASE_URL = (() => {
  if (window.location.protocol === "file:") {
    return "http://127.0.0.1:8000";
  }

  if (window.location.origin.includes(":8001")) {
    return "http://127.0.0.1:8000";
  }

  return window.location.origin;
})();

const elements = {
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  status: document.querySelector("#status"),
  statusText: document.querySelector("#statusText"),
  highpassInput: document.querySelector("#highpassInput"),
  lowpassInput: document.querySelector("#lowpassInput"),
  noiseToggle: document.querySelector("#noiseToggle"),
  signalTypeSelect: document.querySelector("#signalTypeSelect"),
  deviceTypeSelect: document.querySelector("#deviceTypeSelect"),
  deviceCustomWrapper: document.querySelector("#deviceCustomWrapper"),
  deviceCustomInput: document.querySelector("#deviceCustomInput"),
  classifyButton: document.querySelector("#classifyButton"),
  resetButton: document.querySelector("#resetButton"),
  deviceTypeLabel: document.querySelector("#deviceTypeLabel"),
  noiseDetailsLabel: document.querySelector("#noiseDetailsLabel"),
  rmsMetric: document.querySelector("#rmsMetric"),
  peakMetric: document.querySelector("#peakMetric"),
  spectrogramCanvas: document.querySelector("#spectrogramCanvas"),
  dbfsMetric: document.querySelector("#dbfsMetric"),
  frequencyMetric: document.querySelector("#frequencyMetric"),
  zcrMetric: document.querySelector("#zcrMetric"),
  latencyMetric: document.querySelector("#latencyMetric"),
  durationLabel: document.querySelector("#durationLabel"),
  filterLabel: document.querySelector("#filterLabel"),
  noiseLabel: document.querySelector("#noiseLabel"),
  classificationLabel: document.querySelector("#classificationLabel"),
  classificationStatus: document.querySelector("#classificationStatus"),
  waveformCanvas: document.querySelector("#waveformCanvas"),
  spectrumCanvas: document.querySelector("#spectrumCanvas"),
};

let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let sampleBuffer = [];
let lastChunk = null;
let inFlight = false;
let lastSendAt = 0;

const CHUNK_SECONDS = 0.35;

elements.startButton.addEventListener("click", startMicrophone);
elements.stopButton.addEventListener("click", stopMicrophone);
elements.classifyButton.addEventListener("click", classifyCurrentChunk);
elements.resetButton.addEventListener("click", resetInterface);
elements.deviceTypeSelect.addEventListener("change", onDeviceTypeChange);

drawWaveform([]);
drawSpectrum([]);

function onDeviceTypeChange() {
  const showCustom = elements.deviceTypeSelect.value === "other";
  elements.deviceCustomWrapper.style.display = showCustom ? "grid" : "none";
}

async function startMicrophone() {
  try {
    setStatus("Requesting microphone", false);
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    audioContext = new AudioContext();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      sampleBuffer.push(...input);

      const targetLength = Math.floor(audioContext.sampleRate * CHUNK_SECONDS);
      if (sampleBuffer.length >= targetLength && !inFlight) {
        const chunk = sampleBuffer.slice(-targetLength);
        sampleBuffer = sampleBuffer.slice(-targetLength);
        const typed = new Float32Array(chunk);
        lastChunk = { samples: typed, sampleRate: audioContext.sampleRate };
        sendAudioChunk(typed, audioContext.sampleRate);
      }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    elements.startButton.disabled = true;
    elements.stopButton.disabled = false;
    setStatus("Connected", true);
  } catch (error) {
    setStatus(error.message || "Microphone blocked", false);
  }
}

function stopMicrophone() {
  if (processorNode) {
    processorNode.disconnect();
  }
  if (sourceNode) {
    sourceNode.disconnect();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
  if (audioContext) {
    audioContext.close();
  }

  audioContext = null;
  mediaStream = null;
  sourceNode = null;
  processorNode = null;
  sampleBuffer = [];
  inFlight = false;

  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  setStatus("Disconnected", false);
}

async function sendAudioChunk(samples, sampleRate) {
  inFlight = true;
  lastSendAt = performance.now();

  try {
    const response = await fetch(`${API_BASE_URL}/api/process`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sampleRate,
        samples: float32ToBase64(samples),
        signalType: elements.signalTypeSelect.value || "generic",
        deviceType:
          elements.deviceTypeSelect.value === "other"
            ? elements.deviceCustomInput.value || "other"
            : elements.deviceTypeSelect.value || "generic",
        settings: {
          highpass: Number(elements.highpassInput.value),
          lowpass: Number(elements.lowpassInput.value),
          detectNoise: elements.noiseToggle.checked,
        },
      }),
    });

    const payload = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(payload.error || `Server rejected the audio chunk (${response.status})`);
    }

    renderMetrics(payload, performance.now() - lastSendAt);
  } catch (error) {
    setStatus(error.message || "Processing error", false);
  } finally {
    inFlight = false;
  }
}

async function classifyCurrentChunk() {
  if (!lastChunk) {
    setClassifierBadge("No chunk", "pill-error");
    setStatus("No audio chunk available yet. Speak to the microphone first.", false);
    return;
  }

  try {
    setStatus("Classifying current chunk…", false);
    setClassifierBadge("Working", "pill-neutral");
    const response = await fetch(`${API_BASE_URL}/api/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sampleRate: lastChunk.sampleRate,
        samples: float32ToBase64(lastChunk.samples),
        signalType: elements.signalTypeSelect.value || "generic",
        deviceType:
          elements.deviceTypeSelect.value === "other"
            ? elements.deviceCustomInput.value || "other"
            : elements.deviceTypeSelect.value || "generic",
      }),
    });

    const payload = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(payload.error || `Server rejected the classification request (${response.status})`);
    }

    elements.classificationLabel.textContent = payload.classification
      ? `${payload.classification.label} (${payload.classification.confidence})`
      : "Unknown";
    elements.deviceTypeLabel.textContent = payload.classification?.deviceTypeHint ||
      elements.deviceTypeSelect.value ||
      "Generic";
    setClassifierBadge("Success", "pill-success");
    setStatus("Classification complete", true);
  } catch (error) {
    setClassifierBadge("Failed", "pill-error");
    setStatus(error.message || "Classification failed", false);
  }
}

function resetInterface() {
  elements.rmsMetric.textContent = "0.000000";
  elements.peakMetric.textContent = "0.000000";
  elements.dbfsMetric.textContent = "-120.00";
  elements.frequencyMetric.textContent = "0.00";
  elements.zcrMetric.textContent = "0.00000";
  elements.latencyMetric.textContent = "0 ms";
  elements.durationLabel.textContent = "0.00 s";
  elements.filterLabel.textContent = "Waiting for audio";
  elements.noiseLabel.textContent = "None";
  elements.classificationLabel.textContent = "Unknown";
  elements.deviceTypeLabel.textContent = elements.deviceTypeSelect.value || "Generic";
  setClassifierBadge("Waiting", "pill-neutral");
  drawWaveform([]);
  drawSpectrum([]);
  drawSpectrogram([]);
  setStatus("Ready", true);
}

function setClassifierBadge(text, variantClass) {
  if (!elements.classificationStatus) {
    return;
  }

  elements.classificationStatus.textContent = text;
  elements.classificationStatus.classList.remove("pill-success", "pill-error", "pill-neutral");
  elements.classificationStatus.classList.add(variantClass);
}

function renderMetrics(payload, latencyMs) {
  const metrics = payload.processed;

  elements.rmsMetric.textContent = metrics.rms.toFixed(6);
  elements.peakMetric.textContent = metrics.peak.toFixed(6);
  elements.dbfsMetric.textContent = metrics.dbfs.toFixed(2);
  elements.frequencyMetric.textContent = metrics.dominantFrequencyHz.toFixed(2);
  elements.zcrMetric.textContent = metrics.zeroCrossingRate.toFixed(5);
  elements.latencyMetric.textContent = `${Math.round(latencyMs)} ms`;
  elements.durationLabel.textContent = `${payload.durationSeconds.toFixed(2)} s`;
  elements.filterLabel.textContent = payload.filters.length ? payload.filters.join(", ") : "None";
  elements.noiseLabel.textContent = payload.noisePeaksHz.length
    ? payload.noisePeaksHz.map((frequency) => `${frequency} Hz`).join(", ")
    : "None";
  elements.noiseDetailsLabel.textContent = payload.noisePeakDetails?.length
    ? payload.noisePeakDetails
        .map(
          (detail) =>
            `${detail.frequency}Hz (ratio ${detail.ratio}, persistence ${detail.persistence}, ${detail.bandwidthHz}Hz)`
        )
        .join("; ")
    : "None";
  drawWaveform(payload.waveform);
  drawSpectrum(payload.spectrum);
  drawSpectrogram(payload.spectrogram);
  elements.classificationLabel.textContent = payload.classification
    ? `${payload.classification.label} (${payload.classification.confidence})`
    : "Unknown";
  elements.deviceTypeLabel.textContent = payload.classification?.deviceTypeHint || elements.deviceTypeSelect.value || "Generic";
  setStatus("Connected", true);
}

function drawWaveform(points) {
  const canvas = elements.waveformCanvas;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const centerY = height / 2;

  context.clearRect(0, 0, width, height);
  drawGrid(context, width, height);
  context.beginPath();
  context.strokeStyle = "#48d1a8";
  context.lineWidth = 2;

  if (!points.length) {
    context.moveTo(0, centerY);
    context.lineTo(width, centerY);
  } else {
    points.forEach((value, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = centerY - value * (height * 0.42);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
  }

  context.stroke();
}

function drawSpectrum(bins) {
  const canvas = elements.spectrumCanvas;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const barWidth = width / Math.max(bins.length, 1);

  context.clearRect(0, 0, width, height);
  drawGrid(context, width, height);
  context.fillStyle = "#f4c95d";

  bins.forEach((bin, index) => {
    const barHeight = Math.max(2, bin.magnitude * (height - 24));
    context.fillRect(index * barWidth, height - barHeight, Math.max(1, barWidth - 2), barHeight);
  });
}

function drawSpectrogram(matrix) {
  const canvas = elements.spectrogramCanvas;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  context.clearRect(0, 0, width, height);
  drawGrid(context, width, height);

  if (!matrix?.length) {
    context.fillStyle = "rgba(255, 255, 255, 0.04)";
    context.fillRect(0, 0, width, height);
    return;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellWidth = width / cols;
  const cellHeight = height / rows;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = Math.min(Math.max(matrix[row][col], 0), 1);
      const intensity = Math.round(value * 255);
      const color = `rgb(${intensity}, ${Math.round(180 + value * 55)}, ${255 - intensity})`;
      context.fillStyle = color;
      context.fillRect(
        col * cellWidth,
        height - (row + 1) * cellHeight,
        Math.max(1, cellWidth),
        Math.max(1, cellHeight)
      );
    }
  }
}

function drawGrid(context, width, height) {
  context.fillStyle = "#1f282b";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(157, 176, 172, 0.16)";
  context.lineWidth = 1;

  for (let x = 0; x <= width; x += width / 6) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }

  for (let y = 0; y <= height; y += height / 4) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function setStatus(message, connected) {
  elements.status.classList.toggle("connected", connected);
  elements.statusText.textContent = message;
}

async function parseJsonResponse(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(`Invalid JSON response from server: ${text.slice(0, 200)}`);
  }
}

function float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = "";
  const chunkSize = 0x8000;

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return btoa(binary);
}
