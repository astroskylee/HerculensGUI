const pageCopy = {
  "image-preprocess": {
    title: "Image preprocess",
    subtitle: "Prepare science images, masks, noise maps, and display products.",
  },
  "psf-fit": {
    title: "PSF fit",
    subtitle: "Fit or load a PSF kernel for the lens model.",
  },
  "lens-model": {
    title: "Lens model",
    subtitle: "Run parametric and pixelated source modeling, then inspect the result.",
  },
};

const pageTitle = document.querySelector("#page-title");
const pageSubtitle = document.querySelector("#page-subtitle");
const backendStatus = document.querySelector("#backend-status");
const backendText = document.querySelector("#backend-text");
const projectRoot = document.querySelector("#project-root");
const runLog = document.querySelector("#run-log");
const themeToggle = document.querySelector("#theme-toggle");
const themeIcon = document.querySelector("#theme-icon");
const themeLabel = document.querySelector("#theme-label");
const imageFileInput = document.querySelector("#image-file");
const selectedImageName = document.querySelector("#selected-image-name");
const loadImagePreview = document.querySelector("#load-image-preview");
const resetImageView = document.querySelector("#reset-image-view");
const imagePreviewCanvas = document.querySelector("#image-preview-canvas");
const imagePreviewEmpty = document.querySelector("#image-preview-empty");
const imageZoomStage = document.querySelector('[data-preview="image-preprocess"]');
const folderPath = document.querySelector("#folder-path");
const openFolder = document.querySelector("#open-folder");
const parentFolder = document.querySelector("#parent-folder");
const folderList = document.querySelector("#folder-list");
const mtfShadows = document.querySelector("#mtf-shadows");
const mtfMidtones = document.querySelector("#mtf-midtones");
const mtfHighlights = document.querySelector("#mtf-highlights");
const mtfShadowsValue = document.querySelector("#mtf-shadows-value");
const mtfMidtonesValue = document.querySelector("#mtf-midtones-value");
const mtfHighlightsValue = document.querySelector("#mtf-highlights-value");
const mtfCurve = document.querySelector("#mtf-curve");
const displayBrightness = document.querySelector("#display-brightness");
const displayBrightnessValue = document.querySelector("#display-brightness-value");
const cutoutSize = document.querySelector("#cutout-size");
const saveCutout = document.querySelector("#save-cutout");
const cutoutCenterLabel = document.querySelector("#cutout-center-label");
const cutoutMarker = document.querySelector("#cutout-marker");

const imageView = {
  scale: 1,
  x: 0,
  y: 0,
  dragging: false,
  lastX: 0,
  lastY: 0,
  original: null,
  sourcePath: "",
  cutoutCenter: null,
  pointerMoved: false,
};

function appendLog(message) {
  const timestamp = new Date().toLocaleTimeString();
  runLog.textContent += `\n[${timestamp}] ${message}`;
  runLog.scrollTop = runLog.scrollHeight;
}

function setBackendState(state, text) {
  backendStatus.classList.remove("idle", "ready", "error");
  backendStatus.classList.add(state);
  backendText.textContent = text;
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  const isDark = theme === "dark";
  themeIcon.textContent = isDark ? "L" : "D";
  themeLabel.textContent = isDark ? "Light" : "Dark";
  localStorage.setItem("herculens-gui-theme", theme);
}

function activePageId() {
  return document.querySelector(".page.active")?.id ?? "image-preprocess";
}

function setPage(pageId) {
  document.querySelectorAll(".nav-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === pageId);
  });

  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === pageId);
  });

  pageTitle.textContent = pageCopy[pageId].title;
  pageSubtitle.textContent = pageCopy[pageId].subtitle;
}

function collectFormPayload(form) {
  const payload = {
    project_root: projectRoot.value.trim(),
  };

  const fields = new FormData(form);
  for (const [key, value] of fields.entries()) {
    payload[key] = value;
  }

  form.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    payload[checkbox.name] = checkbox.checked;
  });

  form.querySelectorAll('input[type="number"]').forEach((numberInput) => {
    if (numberInput.name && payload[numberInput.name] !== "") {
      payload[numberInput.name] = Number(numberInput.value);
    }
  });

  return payload;
}

function endpointUrl(endpoint) {
  return endpoint;
}

async function callBackend(endpoint, payload = null) {
  const options = payload
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    : { method: "GET" };

  const response = await fetch(endpointUrl(endpoint), options);
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { message: text };
  }

  if (!response.ok) {
    throw new Error(data.detail || data.message || `HTTP ${response.status}`);
  }

  return data;
}

async function uploadImageForPreview(file) {
  const formData = new FormData();
  formData.append("image_file", file);

  const response = await fetch(endpointUrl("/api/image-preprocess/run"), {
    method: "POST",
    body: formData,
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { message: text };
  }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `HTTP ${response.status}`);
  }
  return data;
}

async function loadFolder(path = "") {
  const query = path ? `?dir=${encodeURIComponent(path)}` : "";
  const data = await callBackend(`/api/files${query}`);
  folderPath.value = data.cwd;
  folderList.innerHTML = "";

  if (!data.entries.length) {
    folderList.innerHTML = '<span class="empty-list">No previewable images in this folder</span>';
    return;
  }

  for (const entry of data.entries) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "folder-item";
    item.dataset.path = entry.path;
    item.dataset.kind = entry.kind;
    item.innerHTML = `<span class="name">${entry.name}</span><span class="kind">${entry.kind}</span>`;
    item.addEventListener("click", async () => {
      if (entry.kind === "folder") {
        await loadFolder(entry.path);
      } else {
        await loadImageFromPath(entry.path);
      }
    });
    folderList.appendChild(item);
  }
}

async function loadImageFromPath(path) {
  appendLog(`Image preprocess: loading ${path}`);
  const data = await callBackend("/api/image-preprocess/run", { path });
  imageView.sourcePath = data.source_path || path;
  setImagePreview(data.preview_url);
  selectedImageName.textContent = path;
  appendLog(data.message || "Image preprocess: backend preview loaded.");
}

function updatePreview(pageId, data) {
  const preview = document.querySelector(`[data-preview="${pageId}"]`);
  if (!preview || !data) return;

  const imageUrl = data.preview_url || data.image_url || data.figure_url;
  if (imageUrl) {
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = `${pageCopy[pageId].title} preview`;
    preview.appendChild(img);
    return;
  }

  if (data.message) {
    preview.textContent = data.message;
  }
}

function applyImageTransform() {
  imagePreviewCanvas.style.transform = `translate(calc(-50% + ${imageView.x}px), calc(-50% + ${imageView.y}px)) scale(${imageView.scale})`;
  updateCutoutMarker();
}

function resetImageTransform() {
  imageView.scale = 1;
  imageView.x = 0;
  imageView.y = 0;
  applyImageTransform();
}

function setImagePreview(src) {
  const img = new Image();
  img.onload = () => {
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = img.naturalWidth;
    sourceCanvas.height = img.naturalHeight;
    const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    sourceCtx.drawImage(img, 0, 0);
    imageView.original = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);

    imagePreviewCanvas.width = sourceCanvas.width;
    imagePreviewCanvas.height = sourceCanvas.height;
    imagePreviewCanvas.style.display = "block";
    imagePreviewEmpty.style.display = "none";
    clearCutoutCenter();
    renderMtfPreview();
    resetImageTransform();
  };
  img.onerror = () => {
    appendLog("Image preprocess: failed to load preview image.");
  };
  img.src = src;
}

function canvasPixelFromEvent(event) {
  const rect = imagePreviewCanvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const x = ((event.clientX - rect.left) / rect.width) * imagePreviewCanvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * imagePreviewCanvas.height;
  if (x < 0 || y < 0 || x >= imagePreviewCanvas.width || y >= imagePreviewCanvas.height) {
    return null;
  }
  return { x, y };
}

function updateCutoutMarker() {
  if (!imageView.cutoutCenter) {
    cutoutMarker.style.display = "none";
    cutoutCenterLabel.textContent = "No center";
    return;
  }
  const rect = imagePreviewCanvas.getBoundingClientRect();
  const stageRect = imageZoomStage.getBoundingClientRect();
  const left = rect.left - stageRect.left + (imageView.cutoutCenter.x / imagePreviewCanvas.width) * rect.width;
  const top = rect.top - stageRect.top + (imageView.cutoutCenter.y / imagePreviewCanvas.height) * rect.height;
  cutoutMarker.style.left = `${left}px`;
  cutoutMarker.style.top = `${top}px`;
  cutoutMarker.style.display = "block";
  cutoutCenterLabel.textContent = `x=${Math.round(imageView.cutoutCenter.x)}, y=${Math.round(imageView.cutoutCenter.y)}`;
}

function setCutoutCenterFromEvent(event) {
  const point = canvasPixelFromEvent(event);
  if (!point) return;
  imageView.cutoutCenter = point;
  updateCutoutMarker();
  appendLog(`Cutout center set: x=${Math.round(point.x)}, y=${Math.round(point.y)}.`);
}

function clearCutoutCenter() {
  imageView.cutoutCenter = null;
  updateCutoutMarker();
}

function fileCanPreviewInBrowser(file) {
  return file.type.startsWith("image/") && !/tiff?$/i.test(file.name);
}

function mtf(m, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (Math.abs(m - 0.5) < 1e-12) return x;
  if (m <= 0) return 1;
  if (m >= 1) return 0;
  return ((m - 1) * x) / (((2 * m - 1) * x) - m);
}

function histogramTransformValue(value, shadows, midtones, highlights) {
  const width = Math.max(highlights - shadows, 1e-6);
  const clipped = Math.min(1, Math.max(0, (value - shadows) / width));
  return Math.min(1, Math.max(0, mtf(midtones, clipped)));
}

function mtfParameters() {
  let shadows = Number(mtfShadows.value);
  let midtones = Number(mtfMidtones.value);
  let highlights = Number(mtfHighlights.value);
  if (highlights <= shadows + 0.005) {
    highlights = Math.min(1, shadows + 0.005);
    mtfHighlights.value = String(highlights);
  }
  return { shadows, midtones, highlights };
}

function updateMtfLabels() {
  const { shadows, midtones, highlights } = mtfParameters();
  mtfShadowsValue.textContent = shadows.toFixed(3);
  mtfMidtonesValue.textContent = midtones.toFixed(3);
  mtfHighlightsValue.textContent = highlights.toFixed(3);
  displayBrightnessValue.textContent = `${Number(displayBrightness.value).toFixed(2)}x`;
}

function drawMtfCurve() {
  const ctx = mtfCurve.getContext("2d");
  const width = mtfCurve.width;
  const height = mtfCurve.height;
  const { shadows, midtones, highlights } = mtfParameters();

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--surface").trim();
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--border").trim();
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = (width * i) / 4;
    const y = (height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--accent").trim();
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let px = 0; px < width; px += 1) {
    const x = px / (width - 1);
    const y = histogramTransformValue(x, shadows, midtones, highlights);
    const py = height - y * height;
    if (px === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function renderMtfPreview() {
  updateMtfLabels();
  drawMtfCurve();
  if (!imageView.original) return;

  const { shadows, midtones, highlights } = mtfParameters();
  const brightness = Number(displayBrightness.value);
  const output = new ImageData(
    new Uint8ClampedArray(imageView.original.data),
    imageView.original.width,
    imageView.original.height,
  );

  for (let i = 0; i < output.data.length; i += 4) {
    output.data[i] = Math.round(255 * Math.min(1, brightness * histogramTransformValue(output.data[i] / 255, shadows, midtones, highlights)));
    output.data[i + 1] = Math.round(255 * Math.min(1, brightness * histogramTransformValue(output.data[i + 1] / 255, shadows, midtones, highlights)));
    output.data[i + 2] = Math.round(255 * Math.min(1, brightness * histogramTransformValue(output.data[i + 2] / 255, shadows, midtones, highlights)));
  }

  const ctx = imagePreviewCanvas.getContext("2d");
  ctx.putImageData(output, 0, 0);
}

function resetMtf() {
  mtfShadows.value = "0";
  mtfMidtones.value = "0.5";
  mtfHighlights.value = "1";
  displayBrightness.value = "1";
  renderMtfPreview();
}

document.querySelectorAll(".nav-tab").forEach((button) => {
  button.addEventListener("click", () => setPage(button.dataset.page));
});

document.querySelector("#clear-log").addEventListener("click", () => {
  runLog.textContent = "Ready.";
});

imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files[0];
  selectedImageName.textContent = file ? `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)` : "No file selected";
});

loadImagePreview.addEventListener("click", async () => {
  const file = imageFileInput.files[0];
  if (!file) {
    appendLog("Image preprocess: please choose an image file first.");
    return;
  }

  loadImagePreview.disabled = true;
  appendLog(`Image preprocess: loading ${file.name}`);
  try {
    const data = await uploadImageForPreview(file);
    imageView.sourcePath = data.uploaded_path || "";
    setImagePreview(data.preview_url);
    appendLog(data.message || "Image preprocess: backend preview loaded.");
  } catch (error) {
    appendLog(`Image preprocess: ${error.message}`);
  } finally {
    loadImagePreview.disabled = false;
  }
});

resetImageView.addEventListener("click", resetImageTransform);

saveCutout.addEventListener("click", async () => {
  if (!imageView.sourcePath) {
    appendLog("Cutout: load an image through the backend first.");
    return;
  }
  if (!imageView.cutoutCenter) {
    appendLog("Cutout: click the image to set a center first.");
    return;
  }
  saveCutout.disabled = true;
  try {
    const data = await callBackend("/api/image-preprocess/cutout", {
      path: imageView.sourcePath,
      x: Math.round(imageView.cutoutCenter.x),
      y: Math.round(imageView.cutoutCenter.y),
      size: Number(cutoutSize.value),
    });
    appendLog(`${data.message} -> ${data.fits_path}`);
  } catch (error) {
    appendLog(`Cutout: ${error.message}`);
  } finally {
    saveCutout.disabled = false;
  }
});

[mtfShadows, mtfMidtones, mtfHighlights, displayBrightness].forEach((slider) => {
  slider.addEventListener("input", renderMtfPreview);
});

document.querySelector("#reset-mtf").addEventListener("click", resetMtf);

imageZoomStage.addEventListener("wheel", (event) => {
  if (!imageView.original) return;
  event.preventDefault();
  const zoomFactor = event.deltaY < 0 ? 1.12 : 0.89;
  imageView.scale = Math.min(20, Math.max(0.1, imageView.scale * zoomFactor));
  applyImageTransform();
});

imageZoomStage.addEventListener("pointerdown", (event) => {
  if (!imageView.original) return;
  imageView.dragging = true;
  imageView.lastX = event.clientX;
  imageView.lastY = event.clientY;
  imageView.pointerMoved = false;
  imageZoomStage.classList.add("dragging");
  imageZoomStage.setPointerCapture(event.pointerId);
});

imageZoomStage.addEventListener("pointermove", (event) => {
  if (!imageView.dragging) return;
  const dx = event.clientX - imageView.lastX;
  const dy = event.clientY - imageView.lastY;
  if (Math.abs(dx) + Math.abs(dy) > 2) {
    imageView.pointerMoved = true;
  }
  imageView.x += dx;
  imageView.y += dy;
  imageView.lastX = event.clientX;
  imageView.lastY = event.clientY;
  applyImageTransform();
});

imageZoomStage.addEventListener("pointerup", (event) => {
  if (!imageView.pointerMoved) {
    setCutoutCenterFromEvent(event);
  }
  imageView.dragging = false;
  imageZoomStage.classList.remove("dragging");
  imageZoomStage.releasePointerCapture(event.pointerId);
});

imageZoomStage.addEventListener("pointercancel", () => {
  imageView.dragging = false;
  imageZoomStage.classList.remove("dragging");
});

themeToggle.addEventListener("click", () => {
  const nextTheme = document.body.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  appendLog(`Theme switched to ${nextTheme}.`);
});

document.querySelector("#check-backend").addEventListener("click", async () => {
  appendLog("Checking backend via same-origin API.");
  try {
    const data = await callBackend("/api/health");
    setBackendState("ready", data.status || "Ready");
    if (!folderPath.value && data.cwd) {
      folderPath.value = data.cwd;
    }
    appendLog("Backend check succeeded.");
  } catch (error) {
    setBackendState("error", "Unavailable");
    appendLog(`Backend check failed: ${error.message}`);
  }
});

openFolder.addEventListener("click", async () => {
  try {
    await loadFolder(folderPath.value.trim());
    appendLog(`Opened folder: ${folderPath.value}`);
  } catch (error) {
    appendLog(`Folder browser: ${error.message}`);
  }
});

parentFolder.addEventListener("click", async () => {
  try {
    const current = folderPath.value.trim();
    const data = await callBackend(`/api/files?dir=${encodeURIComponent(current)}`);
    await loadFolder(data.parent);
    appendLog(`Opened folder: ${data.parent}`);
  } catch (error) {
    appendLog(`Folder browser: ${error.message}`);
  }
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", async () => {
    const pageId = activePageId();
    const form = document.querySelector(`[data-form="${pageId}"]`);
    const endpoint = button.dataset.endpoint;
    const action = button.dataset.action;
    const payload = action === "run" ? collectFormPayload(form) : null;

    appendLog(`${pageCopy[pageId].title}: calling ${endpoint}`);
    button.disabled = true;

    try {
      const data = await callBackend(endpoint, payload);
      updatePreview(pageId, data);
      appendLog(data.message || `${pageCopy[pageId].title}: request completed.`);
    } catch (error) {
      appendLog(`${pageCopy[pageId].title}: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  });
});

applyTheme(localStorage.getItem("herculens-gui-theme") || "light");
renderMtfPreview();

loadFolder().catch((error) => {
  appendLog(`Folder browser: ${error.message}`);
});
