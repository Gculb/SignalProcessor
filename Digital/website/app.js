const elements = {
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  status: document.querySelector("#status"),
  statusText: document.querySelector("#statusText"),
  highpassInput: document.querySelector("#highpassInput"),
  lowpassInput: document.querySelector("#lowpassInput"),
  noiseToggle: document.querySelector("#noiseToggle"),
  rmsMetric: document.querySelector("#rmsMetric"),
  peakMetric: document.querySelector("#peakMetric"),
  dbfsMetric: document.querySelector("#dbfsMetric"),
  frequencyMetric: document.querySelector("#frequencyMetric"),
  zcrMetric: document.querySelector("#zcrMetric"),
  latencyMetric: document.querySelector("#latencyMetric"),
  durationLabel: document.querySelector("#durationLabel"),
  filterLabel: document.querySelector("#filterLabel"),
  noiseLabel: document.querySelector("#noiseLabel"),
  waveformCanvas: document.querySelector("#waveformCanvas"),
  spectrumCanvas: document.querySelector("#spectrumCanvas"),
};

let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let sampleBuffer = [];
let inFlight = false;
let lastSendAt = 0;

const CHUNK_SECONDS = 0.35;

elements.startButton.addEventListener("click", startMicrophone);
elements.stopButton.addEventListener("click", stopMicrophone);

drawWaveform([]);
drawSpectrum([]);

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
        sendAudioChunk(new Float32Array(chunk), audioContext.sampleRate);
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
    const response = await fetch("/api/process", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sampleRate,
        samples: float32ToBase64(samples),
        settings: {
          highpass: Number(elements.highpassInput.value),
          lowpass: Number(elements.lowpassInput.value),
          detectNoise: elements.noiseToggle.checked,
        },
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Server rejected the audio chunk");
    }

    renderMetrics(payload, performance.now() - lastSendAt);
  } catch (error) {
    setStatus(error.message || "Processing error", false);
  } finally {
    inFlight = false;
  }
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

  drawWaveform(payload.waveform);
  drawSpectrum(payload.spectrum);
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

function float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = "";
  const chunkSize = 0x8000;

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return btoa(binary);
}
