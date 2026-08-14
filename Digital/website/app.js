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
  trainingLabelSelect: document.querySelector("#trainingLabelSelect"),
  trainingSampleRateInput: document.querySelector("#trainingSampleRateInput"),
  trainingSamplesInput: document.querySelector("#trainingSamplesInput"),
  trainingBatchInput: document.querySelector("#trainingBatchInput"),
  trainingFileInput: document.querySelector("#trainingFileInput"),
  addTrainingExampleButton: document.querySelector("#addTrainingExampleButton"),
  addTrainingBatchButton: document.querySelector("#addTrainingBatchButton"),
  generateDemoExamplesButton: document.querySelector("#generateDemoExamplesButton"),
  trainModelButton: document.querySelector("#trainModelButton"),
  trainingSummary: document.querySelector("#trainingSummary"),
  unlabeledWavInput: document.querySelector("#unlabeledWavInput"),
  analyzeUnlabeledButton: document.querySelector("#analyzeUnlabeledButton"),
  applyClusterLabelsButton: document.querySelector("#applyClusterLabelsButton"),
  saveReviewedLabelsButton: document.querySelector("#saveReviewedLabelsButton"),
  clusterSummary: document.querySelector("#clusterSummary"),
  clusterReviewList: document.querySelector("#clusterReviewList"),
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
elements.addTrainingExampleButton.addEventListener("click", addTrainingExample);
elements.addTrainingBatchButton.addEventListener("click", addTrainingBatchExamples);
elements.generateDemoExamplesButton.addEventListener("click", generateDemoExamples);
elements.trainModelButton.addEventListener("click", trainModelFromExamples);
elements.trainingFileInput.addEventListener("change", handleTrainingFileUpload);
elements.analyzeUnlabeledButton.addEventListener("click", analyzeUnlabeledFolder);
elements.applyClusterLabelsButton.addEventListener("click", applyClusterLabelsToTrainingQueue);
elements.saveReviewedLabelsButton.addEventListener("click", saveReviewedLabels);

drawWaveform([]);
drawSpectrum([]);

let trainingExamples = [];
let clusteredFiles = [];

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

function addTrainingExample() {
  const label = elements.trainingLabelSelect.value;
  const text = elements.trainingSamplesInput.value.trim();
  if (!text) {
    setTrainingSummary("Add numeric samples or upload a CSV/WAV before adding an example.");
    return;
  }

  const samples = parseNumericSamples(text);
  if (!samples.length) {
    setTrainingSummary("No valid numeric values were found. Use comma-separated values or a CSV file.");
    return;
  }

  trainingExamples.push({
    label,
    sampleRate: Number(elements.trainingSampleRateInput.value) || 48000,
    samples: Array.from(samples.slice(0, 96000)),
  });

  elements.trainingSamplesInput.value = "";
  elements.trainingFileInput.value = "";
  setTrainingSummary(
    `${trainingExamples.length} labeled example(s) ready. Click “Train model” to retrain the classifier.`
  );
}

function addTrainingBatchExamples() {
  const text = elements.trainingBatchInput.value.trim();
  if (!text) {
    setTrainingSummary("Paste batch rows in the format label,sampleRate,samples before adding them.");
    return;
  }

  const rows = parseBatchCsvRows(text);
  if (!rows.length) {
    setTrainingSummary("No valid batch rows were found. Use label,sampleRate,samples on each line.");
    return;
  }

  trainingExamples.push(...rows);
  elements.trainingBatchInput.value = "";
  setTrainingSummary(`${trainingExamples.length} labeled example(s) ready. Click “Train model” to retrain the classifier.`);
}

function generateDemoExamples() {
  const sampleRate = Number(elements.trainingSampleRateInput.value) || 48000;
  const palette = [
    { label: "speech_like", generator: makeSpeechDemoSignal },
    { label: "ecg_like", generator: makeEcgDemoSignal },
    { label: "noise", generator: makeNoiseDemoSignal },
  ];

  for (const entry of palette) {
    trainingExamples.push({
      label: entry.label,
      sampleRate,
      samples: entry.generator(sampleRate),
    });
  }

  setTrainingSummary(
    `Generated 3 quick demo examples (${palette.map((item) => item.label).join(", ")}). Click “Train model” to retrain.`
  );
}

async function trainModelFromExamples() {
  if (!trainingExamples.length) {
    setTrainingSummary("No training samples were added yet. Paste values or upload a file first.");
    return;
  }

  try {
    setTrainingSummary("Training model with the current labeled examples…");
    const response = await fetch(`${API_BASE_URL}/api/train`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        examples: trainingExamples.map((example) => ({
          sampleRate: example.sampleRate,
          samples: float32ToBase64(new Float32Array(example.samples)),
          label: example.label,
        })),
      }),
    });

    const payload = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || "Training failed.");
    }

    const trained = payload.training ?? {};
    const summary = `Training complete: ${trained.trainedExamples ?? trainingExamples.length} examples processed. Labels: ${((trained.labels || []).join(", ") || "n/a")}.`;
    setTrainingSummary(summary);
    setStatus("Model retrained", true);
    setClassifierBadge(payload.classification?.label || "Updated", "pill-success");
    elements.classificationLabel.textContent = payload.classification
      ? `${payload.classification.label} (${payload.classification.confidence})`
      : "Unknown";
  } catch (error) {
    setTrainingSummary(error.message || "Training failed.");
    setStatus(error.message || "Training failed", false);
    setClassifierBadge("Failed", "pill-error");
  }
}

async function handleTrainingFileUpload(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }

  try {
    const name = file.name.toLowerCase();

    if (name.endsWith(".csv") || name.endsWith(".txt")) {
      const text = await file.text();
      const rows = parseBatchCsvRows(text);
      if (rows.length) {
        trainingExamples.push(...rows);
        setTrainingSummary(`Imported ${rows.length} labeled rows from ${file.name}. Click “Train model” to retrain.`);
        return;
      }

      const samples = parseNumericSamples(text);
      if (!samples.length) {
        throw new Error("The selected file did not contain readable numeric samples or labeled CSV rows.");
      }

      elements.trainingSamplesInput.value = samples.slice(0, 96).map((value) => value.toFixed(6)).join(", ");
      setTrainingSummary(`Loaded ${samples.length} samples from ${file.name}. Click “Add example” to label and queue it for training.`);
      return;
    }

    if (name.endsWith(".wav")) {
      const buffer = await file.arrayBuffer();
      const samples = decodeWavToFloat32(buffer);
      if (!samples.length) {
        throw new Error("The selected WAV file did not contain readable samples.");
      }

      elements.trainingSamplesInput.value = samples.slice(0, 96).map((value) => value.toFixed(6)).join(", ");
      setTrainingSummary(`Loaded ${samples.length} samples from ${file.name}. Click “Add example” to label and queue it for training.`);
      return;
    }

    throw new Error("Unsupported file type. Please upload a CSV, TXT, or WAV file.");
  } catch (error) {
    setTrainingSummary(error.message || "Could not read the uploaded file.");
  }
}

function parseBatchCsvRows(text) {
  const lines = String(text)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    return [];
  }

  const parsedRows = [];
  for (const line of lines) {
    const cells = splitCsvLine(line);
    if (!cells.length) {
      continue;
    }

    const first = cells[0].toLowerCase();
    const hasHeader = first.includes("label") || first.includes("sample") || first.includes("samples");
    if (hasHeader) {
      continue;
    }

    const labelValue = cells[0]?.trim();
    const sampleRateValue = cells[1]?.trim();
    const samplesText = cells.slice(2).join(",");

    if (!labelValue || !sampleRateValue || !samplesText) {
      continue;
    }

    const label = labelValue.toLowerCase();
    const sampleRate = Number(sampleRateValue);
    const samples = parseNumericSamples(samplesText);
    if (!samples.length || !Number.isFinite(sampleRate)) {
      continue;
    }

    parsedRows.push({
      label,
      sampleRate,
      samples: Array.from(samples.slice(0, 96000)),
    });
  }

  return parsedRows;
}

function splitCsvLine(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (character === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current.trim());
  return cells.filter((part) => part !== "");
}

function parseNumericSamples(value) {
  const matches = String(value)
    .split(/[\s,\r\n]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));

  return matches;
}

function makeSpeechDemoSignal(sampleRate) {
  const size = 2048;
  const time = Array.from({ length: size }, (_, index) => index / sampleRate);
  const envelope = time.map((t) => 0.35 + 0.65 * Math.sin(2 * Math.PI * 2.5 * t));
  return time.map((t, index) => Math.sin(2 * Math.PI * 220 * t) * envelope[index] + (Math.random() - 0.5) * 0.06);
}

function makeEcgDemoSignal(sampleRate) {
  const size = 2048;
  const samples = new Array(size).fill(0);
  for (let index = 0; index < size; index += 1) {
    const t = index / sampleRate;
    const beat = 60 * t;
    if (Math.abs((beat % 1) - 0.2) < 0.1) {
      const localIndex = index % 160;
      const pulse = Math.max(0, 1 - Math.abs(localIndex - 80) / 60);
      samples[index] = pulse * 1.2;
    }
  }
  return samples.map((value) => value * 0.9 + (Math.random() - 0.5) * 0.05);
}

function makeNoiseDemoSignal(sampleRate) {
  const size = 2048;
  return Array.from({ length: size }, () => (Math.random() - 0.5) * 0.8 + (Math.random() - 0.5) * 0.12);
}

function decodeWavToFloat32(buffer) {
  const view = new DataView(buffer);
  const riffHeader = String.fromCharCode(...new Uint8Array(buffer.slice(0, 4)));
  if (riffHeader !== "RIFF") {
    throw new Error("The uploaded WAV file is not valid RIFF data.");
  }

  let offset = 12;
  let dataOffset = -1;
  let dataSize = 0;
  let bitsPerSample = 16;
  let channels = 1;

  while (offset + 8 <= view.byteLength) {
    const chunkId = String.fromCharCode(...new Uint8Array(buffer.slice(offset, offset + 4)));
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;

    if (chunkId === "fmt ") {
      channels = view.getUint16(chunkStart + 2, true);
      bitsPerSample = view.getUint16(chunkStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkStart;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize;
  }

  if (dataOffset < 0 || dataSize <= 0) {
    throw new Error("The uploaded WAV file does not contain a data chunk.");
  }

  const raw = new Uint8Array(buffer.slice(dataOffset, dataOffset + dataSize));
  const sampleCount = raw.length / (bitsPerSample / 8) / channels;
  const samples = new Float32Array(sampleCount);

  if (bitsPerSample === 8) {
    for (let index = 0; index < samples.length; index += 1) {
      const byte = raw[index * channels] ?? 0;
      samples[index] = (byte - 128) / 128;
    }
  } else if (bitsPerSample === 16) {
    for (let index = 0; index < samples.length; index += 1) {
      const offsetIndex = (index * channels * 2);
      const value = view.getInt16(dataOffset + offsetIndex, true);
      samples[index] = value / 32768;
    }
  } else if (bitsPerSample === 24) {
    for (let index = 0; index < samples.length; index += 1) {
      const offsetIndex = index * channels * 3;
      let value = raw[offsetIndex] | (raw[offsetIndex + 1] << 8) | (raw[offsetIndex + 2] << 16);
      if (value & 0x800000) {
        value -= 1 << 24;
      }
      samples[index] = value / 8388608;
    }
  } else {
    for (let index = 0; index < samples.length; index += 1) {
      const offsetIndex = (index * channels * 4);
      const value = view.getInt32(dataOffset + offsetIndex, true);
      samples[index] = value / 2147483648;
    }
  }

  return Array.from(samples);
}

function setTrainingSummary(message) {
  elements.trainingSummary.textContent = message;
}

function setClusterSummary(message) {
  elements.clusterSummary.textContent = message;
}

function analyzeUnlabeledFolder() {
  const files = Array.from(elements.unlabeledWavInput.files || []);
  if (!files.length) {
    setClusterSummary("Choose one or more WAV files before analyzing the folder.");
    return;
  }

  Promise.all(
    files.map(async (file) => {
      const arrayBuffer = await file.arrayBuffer();
      const samples = decodeWavToFloat32(arrayBuffer);
      const sampleRate = estimateWavSampleRate(arrayBuffer) || 48000;
      const features = extractSignalFeatures(samples, sampleRate);
      return {
        fileName: file.name,
        sampleRate,
        samples,
        features,
      };
    })
  )
    .then((entries) => {
      const clusterCount = Math.min(3, Math.max(2, entries.length));
      const assignments = kMeans(entries.map((entry) => entry.features), clusterCount);
      clusteredFiles = entries.map((entry, index) => ({
        ...entry,
        clusterIndex: assignments[index],
      }));

      renderClusterReview();
      setClusterSummary(
        `Grouped ${entries.length} unlabeled WAV files into ${clusterCount} clusters. Review each cluster and assign a label.`
      );
    })
    .catch((error) => {
      setClusterSummary(error.message || "Could not analyze the WAV folder.");
    });
}

function renderClusterReview() {
  if (!clusteredFiles.length) {
    elements.clusterReviewList.innerHTML = "";
    return;
  }

  const groups = {};
  clusteredFiles.forEach((entry) => {
    if (!groups[entry.clusterIndex]) {
      groups[entry.clusterIndex] = [];
    }
    groups[entry.clusterIndex].push(entry);
  });

  elements.clusterReviewList.innerHTML = Object.entries(groups)
    .map(([clusterIndex, entries]) => {
      const clusterName = `Cluster ${Number(clusterIndex) + 1}`;
      return `
        <div class="cluster-card">
          <h3>${clusterName}</h3>
          <ul>
            ${entries.map((entry) => `<li>${entry.fileName}</li>`).join("")}
          </ul>
          <label>
            <span>Assign label</span>
            <select data-cluster-index="${clusterIndex}">
              <option value="uncertain">uncertain</option>
              <option value="noise">noise</option>
              <option value="speech_like">speech_like</option>
              <option value="ecg_like">ecg_like</option>
            </select>
          </label>
        </div>
      `;
    })
    .join("");
}

function applyClusterLabelsToTrainingQueue() {
  if (!clusteredFiles.length) {
    setClusterSummary("Analyze a WAV folder first before assigning cluster labels.");
    return;
  }

  const selects = Array.from(document.querySelectorAll("[data-cluster-index]"));
  const assignmentMap = {};
  selects.forEach((select) => {
    assignmentMap[select.dataset.clusterIndex] = select.value;
  });

  const queued = clusteredFiles
    .filter((entry) => assignmentMap[entry.clusterIndex] && assignmentMap[entry.clusterIndex] !== "uncertain")
    .map((entry) => ({
      label: assignmentMap[entry.clusterIndex],
      sampleRate: entry.sampleRate,
      samples: entry.samples,
    }));

  if (!queued.length) {
    setClusterSummary("No cluster labels were assigned. Choose a label for at least one cluster before training.");
    return;
  }

  trainingExamples.push(...queued);
  setTrainingSummary(`Added ${queued.length} reviewed files from the unlabeled WAV clusters to the training queue.`);
  setClusterSummary(`Applied labels to ${queued.length} WAV files. Click “Train model” to retrain the classifier.`);
}

async function saveReviewedLabels() {
  const selects = Array.from(document.querySelectorAll("[data-cluster-index]"));
  const assignments = selects
    .filter((select) => select.value && select.value !== "uncertain")
    .map((select) => {
      const clusterIndex = Number(select.dataset.clusterIndex);
      const group = clusteredFiles.filter((entry) => entry.clusterIndex === clusterIndex);
      return group.map((entry) => ({
        fileName: entry.fileName,
        label: select.value,
      }));
    })
    .flat();

  if (!assignments.length) {
    setClusterSummary("No cluster labels were selected. Choose labels before saving.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/save-reviewed-labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assignments }),
    });

    const payload = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || "Failed to save reviewed labels.");
    }

    setClusterSummary(`Saved ${payload.savedAssignments} reviewed assignments to the training plan.`);
  } catch (error) {
    setClusterSummary(error.message || "Failed to save reviewed labels.");
  }
}

function extractSignalFeatures(samples, sampleRate) {
  const safeSamples = Array.from(samples || []).filter((value) => Number.isFinite(value));
  if (!safeSamples.length) {
    return [0, 0, 0, 0, 0, 0, 0, 0];
  }

  const rms = Math.sqrt(safeSamples.reduce((sum, value) => sum + value * value, 0) / safeSamples.length);
  const peak = safeSamples.reduce((maxValue, value) => Math.max(maxValue, Math.abs(value)), 0);
  const zcr = safeSamples.slice(1).reduce((count, sample, index) => {
    const previous = safeSamples[index];
    return count + (Math.sign(previous) !== Math.sign(sample) ? 1 : 0);
  }, 0) / Math.max(safeSamples.length - 1, 1);

  const lowBand = computeDftEnergy(safeSamples, sampleRate, 0, 40);
  const speechBand = computeDftEnergy(safeSamples, sampleRate, 120, 3400);
  const dominant = estimateDominantFrequency(safeSamples, sampleRate);
  const flatness = estimateSpectralFlatness(safeSamples, sampleRate);

  return [
    rms,
    peak,
    20 * Math.log10(Math.max(rms, 1e-6)),
    zcr,
    speechBand,
    lowBand,
    flatness,
    dominant,
  ];
}

function computeDftEnergy(samples, sampleRate, minHz, maxHz) {
  if (!samples.length) {
    return 0;
  }

  const duration = samples.length / sampleRate;
  const binWindow = Math.max(1, Math.floor((sampleRate / 2) / 64));
  let total = 0;
  for (let bin = 0; bin < 64; bin += 1) {
    const frequency = bin * binWindow;
    if (frequency < minHz || frequency > maxHz) {
      continue;
    }

    let real = 0;
    let imag = 0;
    for (let index = 0; index < samples.length; index += 1) {
      const angle = -2 * Math.PI * frequency * (index / sampleRate);
      real += samples[index] * Math.cos(angle);
      imag += samples[index] * Math.sin(angle);
    }

    const magnitude = Math.sqrt(real * real + imag * imag);
    total += magnitude * magnitude;
  }

  return total / Math.max(duration, 1e-6);
}

function estimateDominantFrequency(samples, sampleRate) {
  if (!samples.length) {
    return 0;
  }

  let bestFrequency = 0;
  let bestMagnitude = -Infinity;
  const maxFrequency = Math.min(sampleRate / 2, 5000);
  const step = Math.max(1, Math.floor(maxFrequency / 64));

  for (let frequency = 1; frequency <= maxFrequency; frequency += step) {
    let real = 0;
    let imag = 0;
    for (let index = 0; index < samples.length; index += 1) {
      const angle = -2 * Math.PI * frequency * (index / sampleRate);
      real += samples[index] * Math.cos(angle);
      imag += samples[index] * Math.sin(angle);
    }
    const magnitude = Math.sqrt(real * real + imag * imag);
    if (magnitude > bestMagnitude) {
      bestMagnitude = magnitude;
      bestFrequency = frequency;
    }
  }
  return bestFrequency;
}

function estimateSpectralFlatness(samples, sampleRate) {
  if (!samples.length) {
    return 0;
  }

  const magnitudes = [];
  const maxFrequency = Math.min(sampleRate / 2, 5000);
  const step = Math.max(1, Math.floor(maxFrequency / 32));

  for (let frequency = 1; frequency <= maxFrequency; frequency += step) {
    let real = 0;
    let imag = 0;
    for (let index = 0; index < samples.length; index += 1) {
      const angle = -2 * Math.PI * frequency * (index / sampleRate);
      real += samples[index] * Math.cos(angle);
      imag += samples[index] * Math.sin(angle);
    }
    magnitudes.push(Math.sqrt(real * real + imag * imag));
  }

  const positive = magnitudes.filter((value) => value > 1e-9);
  if (!positive.length) {
    return 0;
  }

  const geomean = Math.exp(positive.reduce((sum, value) => sum + Math.log(value), 0) / positive.length);
  const arithmean = positive.reduce((sum, value) => sum + value, 0) / positive.length;
  return geomean / (arithmean + 1e-9);
}

function kMeans(data, clusters) {
  if (!data.length) {
    return [];
  }

  const clusterCount = Math.min(clusters, data.length);
  const centroids = data.slice(0, clusterCount).map((point) => [...point]);
  const assignments = new Array(data.length).fill(0);

  for (let iteration = 0; iteration < 20; iteration += 1) {
    for (let pointIndex = 0; pointIndex < data.length; pointIndex += 1) {
      const point = data[pointIndex];
      let bestCluster = 0;
      let bestDistance = Infinity;
      for (let clusterIndex = 0; clusterIndex < clusterCount; clusterIndex += 1) {
        const distance = euclideanDistance(point, centroids[clusterIndex]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = clusterIndex;
        }
      }
      assignments[pointIndex] = bestCluster;
    }

    for (let clusterIndex = 0; clusterIndex < clusterCount; clusterIndex += 1) {
      const members = data
        .map((point, index) => (assignments[index] === clusterIndex ? point : null))
        .filter(Boolean);

      if (!members.length) {
        centroids[clusterIndex] = [...data[clusterIndex]];
        continue;
      }

      centroids[clusterIndex] = members[0].map((_, dimIndex) => {
        const total = members.reduce((sum, member) => sum + member[dimIndex], 0);
        return total / members.length;
      });
    }
  }

  return assignments;
}

function euclideanDistance(left, right) {
  return left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0) ** 0.5;
}

function estimateWavSampleRate(buffer) {
  const view = new DataView(buffer);
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const chunkId = String.fromCharCode(...new Uint8Array(buffer.slice(offset, offset + 4)));
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkId === "fmt ") {
      return view.getUint32(chunkStart + 12, true);
    }
    offset += 8 + chunkSize;
  }
  return null;
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
