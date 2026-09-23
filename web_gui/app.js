const pageCopy = {
  "euclid-cutout": {
    title: "Get Euclid cutout",
    subtitle: "Download a Euclid cutout and inspect photometry.",
  },
  "image-preprocess": {
    title: "Image preprocess",
    subtitle: "Prepare science images, masks, noise maps, and display products.",
  },
  "psf-fit": {
    title: "PSF fit",
    subtitle: "Fit or load a PSF kernel for the lens model.",
  },
  mask: {
    title: "Mask",
    subtitle: "Draw and save a cutout mask.",
  },
  "lens-model": {
    title: "Lens model",
    subtitle: "Run parametric and pixelated source modeling, then inspect the result.",
  },
};

const PHOTOZ_BAND_LIST = ["VIS", "NIR_Y", "NIR_J", "NIR_H", "g", "r", "i", "z"];
const PHOTOZ_SUPPORTED_BANDS = new Set(PHOTOZ_BAND_LIST);
const WHEEL_ZOOM_STEP = 1.04;
const MTF_DATA_DISPLAY_TARGET = 0.125;
const MTF_LENS_LIGHT_DISPLAY_TARGET = 0.25;
const DEFAULT_MTF_VALUES = { shadows: 0, midtones: 0.5, highlights: 1 };
const DEFAULT_MASK_MTF_VALUES = { shadows: 0, midtones: 0.125, highlights: 1 };
const MASK_LENS_LIGHT_SIGMA_MAX_FRACTION = 0.4;
const DEFAULT_SIS_THETA_HIGH = 1;

const pageTitle = document.querySelector("#page-title");
const pageSubtitle = document.querySelector("#page-subtitle");
const backendStatus = document.querySelector("#backend-status");
const backendText = document.querySelector("#backend-text");
const activityStatus = document.querySelector("#activity-status");
const checkBackendButton = document.querySelector("#check-backend");
const runLog = document.querySelector("#run-log");
const themeToggle = document.querySelector("#theme-toggle");
const themeIcon = document.querySelector("#theme-icon");
const themeLabel = document.querySelector("#theme-label");
const settingsOpen = document.querySelector("#settings-open");
const settingsModal = document.querySelector("#settings-modal");
const settingsClose = document.querySelector("#settings-close");
const settingsForm = document.querySelector("#settings-form");
const settingsStatus = document.querySelector("#settings-status");
const settingsPath = document.querySelector("#settings-path");
const settingsHerculensPython = document.querySelector("#settings-herculens-python");
const settingsEuclidPython = document.querySelector("#settings-euclid-python");
const settingsPhotozPython = document.querySelector("#settings-photoz-python");
const settingsPhosphorosRoot = document.querySelector("#settings-phosphoros-root");
const startNewModel = document.querySelector("#start-new-model");
const chooseProjectButton = document.querySelector("#choose-project");
const projectChooserModal = document.querySelector("#project-chooser-modal");
const projectChooserClose = document.querySelector("#project-chooser-close");
const projectChooserStatus = document.querySelector("#project-chooser-status");
const projectChooserGrid = document.querySelector("#project-chooser-grid");
const projectChooserViewInputs = document.querySelectorAll('input[name="project_chooser_view"]');
const projectChooserSearch = document.querySelector("#project-chooser-search");
const projectChooserScore = document.querySelector("#project-chooser-score");
const projectChooserSort = document.querySelector("#project-chooser-sort");
const projectSelect = document.querySelector("#project-select");
const projectName = document.querySelector("#project-name");
const renameProjectButton = document.querySelector("#rename-project");
const currentProjectState = document.querySelector("#current-project-state");
const euclidRa = document.querySelector("#euclid-ra");
const euclidDec = document.querySelector("#euclid-dec");
const euclidCutoutSize = document.querySelector("#euclid-cutout-size");
const euclidUsername = document.querySelector("#euclid-username");
const euclidPassword = document.querySelector("#euclid-password");
const euclidGenerateProjectName = document.querySelector("#euclid-generate-project-name");
const euclidRun = document.querySelector("#euclid-run");
const legacyRun = document.querySelector("#legacy-run");
const euclidStage = document.querySelector(".euclid-stage");
const euclidPlotFrame = document.querySelector("#euclid-plot-frame");
const euclidPreviewCanvas = document.querySelector("#euclid-preview-canvas");
const euclidPreviewEmpty = document.querySelector("#euclid-preview-empty");
const euclidPreviewMeta = document.querySelector("#euclid-preview-meta");
const euclidZeropoint = document.querySelector("#euclid-zeropoint");
const cutoutImageSelect = document.querySelector("#cutout-image-select");
const euclidDisplayMode = document.querySelector("#euclid-display-mode");
const euclidSbLower = document.querySelector("#euclid-sb-lower");
const euclidSbUpper = document.querySelector("#euclid-sb-upper");
const euclidColorbar = document.querySelector("#euclid-colorbar");
const euclidColorbarTicks = document.querySelector("#euclid-colorbar-ticks");
const euclidAxisX = document.querySelector("#euclid-axis-x");
const euclidAxisY = document.querySelector("#euclid-axis-y");
const euclidApertureOverlay = document.querySelector("#euclid-aperture-overlay");
const euclidAnnulusOverlay = document.querySelector("#euclid-annulus-overlay");
const euclidMtfShadows = document.querySelector("#euclid-mtf-shadows");
const euclidMtfMidtones = document.querySelector("#euclid-mtf-midtones");
const euclidMtfHighlights = document.querySelector("#euclid-mtf-highlights");
const euclidMtfShadowsValue = document.querySelector("#euclid-mtf-shadows-value");
const euclidMtfMidtonesValue = document.querySelector("#euclid-mtf-midtones-value");
const euclidMtfHighlightsValue = document.querySelector("#euclid-mtf-highlights-value");
const euclidResetMtf = document.querySelector("#euclid-reset-mtf");
const euclidPhotometryToggle = document.querySelector("#euclid-photometry-toggle");
const euclidPhotozToggle = document.querySelector("#euclid-photoz-toggle");
const euclidPhotozBandInputs = Array.from(document.querySelectorAll("[data-photoz-band]"));
const euclidMagSystem = document.querySelector("#euclid-mag-system");
const euclidApertureRadius = document.querySelector("#euclid-aperture-radius");
const euclidAnnulusWidth = document.querySelector("#euclid-annulus-width");
const euclidPhotometryReadout = document.querySelector("#euclid-photometry-readout");
const euclidPreviewScale = document.querySelector("#euclid-preview-scale");
const euclidPreviewScaleValue = document.querySelector("#euclid-preview-scale-value");
const imageFileInput = document.querySelector("#image-file");
const inputImagePath = document.querySelector("#input-image-path");
const selectedImageName = document.querySelector("#selected-image-name");
const rmsFileInput = document.querySelector("#rms-file");
const inputRmsPath = document.querySelector("#input-rms-path");
const selectedRmsName = document.querySelector("#selected-rms-name");
const loadRmsFile = document.querySelector("#load-rms-file");
const loadImagePreview = document.querySelector("#load-image-preview");
const resetImageView = document.querySelector("#reset-image-view");
const imageInputGroup = document.querySelector("#image-input-group");
const imageCutoutGroup = document.querySelector("#image-cutout-group");
const imagePreviewCanvas = document.querySelector("#image-preview-canvas");
const imagePreviewEmpty = document.querySelector("#image-preview-empty");
const imageZoomStage = document.querySelector('[data-preview="image-preprocess"]');
const folderPath = document.querySelector("#folder-path");
const setProjectFolder = document.querySelector("#set-project-folder");
const openFolder = document.querySelector("#open-folder");
const folderList = document.querySelector("#folder-list");
const mtfShadows = document.querySelector("#mtf-shadows");
const mtfMidtones = document.querySelector("#mtf-midtones");
const mtfHighlights = document.querySelector("#mtf-highlights");
const mtfShadowsValue = document.querySelector("#mtf-shadows-value");
const mtfMidtonesValue = document.querySelector("#mtf-midtones-value");
const mtfHighlightsValue = document.querySelector("#mtf-highlights-value");
const mtfCurve = document.querySelector("#mtf-curve");
const cutoutSize = document.querySelector("#cutout-size");
const saveCutout = document.querySelector("#save-cutout");
const cutoutCenterLabel = document.querySelector("#cutout-center-label");
const cutoutMarker = document.querySelector("#cutout-marker");
const bgBoxSize = document.querySelector("#bg-box-size");
const bgBoxLabel = document.querySelector("#bg-box-label");
const bgBoxMarker = document.querySelector("#bg-box-marker");
const clearBgBox = document.querySelector("#clear-bg-box");
const cutoutPreviewPanel = document.querySelector("#cutout-preview-panel");
const cutoutPreviewCanvas = document.querySelector("#cutout-preview-canvas");
const cutoutPreviewLabel = document.querySelector("#cutout-preview-label");
const psfOverlay = document.querySelector("#psf-overlay");
const imagePanX = document.querySelector("#image-pan-x");
const imagePanY = document.querySelector("#image-pan-y");
const psfModeInputs = document.querySelectorAll('input[name="psf_mode"]');
const psfModePanels = document.querySelectorAll("[data-psf-mode-panel]");
const psfInputPath = document.querySelector("#psf-input-path");
const psfInputFile = document.querySelector("#psf-input-file");
const psfLoadInput = document.querySelector("#psf-load-input");
const psfDefaultEuclid = document.querySelector("#psf-default-euclid");
const psfDefaultEuclidY = document.querySelector("#psf-default-euclid-y");
const psfDefaultEuclidJ = document.querySelector("#psf-default-euclid-j");
const psfDefaultEuclidH = document.querySelector("#psf-default-euclid-h");
const psfSciencePath = document.querySelector("#psf-science-path");
const psfKernelSize = document.querySelector("#psf-kernel-size");
const psfFwhm = document.querySelector("#psf-fwhm");
const psfThresholdSigma = document.querySelector("#psf-threshold-sigma");
const psfFindStars = document.querySelector("#psf-find-stars");
const psfRunFit = document.querySelector("#psf-run-fit");
const psfStopFit = document.querySelector("#psf-stop-fit");
const psfClearOutput = document.querySelector("#psf-clear-output");
const psfSelectedIds = document.querySelector("#psf-selected-ids");
const psfSviSteps = document.querySelector("#psf-svi-steps");
const psfSsFactor = document.querySelector("#psf-ss-factor");
const psfPreviewViewInputs = document.querySelectorAll('input[name="psf_preview_view"]');
const psfPreviewLayout = document.querySelector(".psf-preview-layout");
const psfMainPreview = document.querySelector("#psf-main-preview");
const psfFullImagePreview = document.querySelector("#psf-full-image-preview");
const psfFullImageFrame = document.querySelector("#psf-full-image-frame");
const psfFullImageImg = document.querySelector("#psf-full-image-img");
const psfFullImageOverlay = document.querySelector("#psf-full-image-overlay");
const psfFullImageEmpty = document.querySelector("#psf-full-image-empty");
const psfCandidatesPanel = document.querySelector(".psf-candidates-panel");
const psfCandidateList = document.querySelector("#psf-candidate-list");
const psfCandidateCount = document.querySelector("#psf-candidate-count");
const maskToolInputs = document.querySelectorAll('input[name="mask_tool"]');
const maskModeInputs = document.querySelectorAll('input[name="mask_mode"]');
const maskType = document.querySelector("#mask-type");
const maskBrushRadius = document.querySelector("#mask-brush-radius");
const maskBrushRadiusValue = document.querySelector("#mask-brush-radius-value");
const maskPreviewGrid = document.querySelector(".mask-preview-grid");
const maskPreviewSource = document.querySelector("#mask-preview-source");
const maskPreviewColormap = document.querySelector("#mask-preview-colormap");
const maskLoadImage = document.querySelector("#mask-load-image");
const maskClear = document.querySelector("#mask-clear");
const maskSave = document.querySelector("#mask-save");
const maskLensLightPanel = document.querySelector(".mask-lens-light-panel");
const maskLensLightSemilinear = document.querySelector("#mask-lens-light-semilinear");
const maskLensLightSubtraction = document.querySelector("#mask-lens-light-subtraction");
const maskLensLightSubtractionGpu = document.querySelector("#mask-lens-light-subtraction-gpu");
const maskLensLightStop = document.querySelector("#mask-lens-light-stop");
const maskLensLightNGauss = document.querySelector("#mask-lens-light-n-gauss");
const maskLensLightSigmaMin = document.querySelector("#mask-lens-light-sigma-min");
const maskLensLightSigmaMax = document.querySelector("#mask-lens-light-sigma-max");
const maskLensLightCenterMaxOffset = document.querySelector("#mask-lens-light-center-max-offset");
const maskLensLightSviSteps = document.querySelector("#mask-lens-light-svi-steps");
const maskLensLightUnconstrainedSviSteps = document.querySelector("#mask-lens-light-unconstrained-svi-steps");
const maskLensLightResultChoice = document.querySelector("#mask-lens-light-result-choice");
const maskLensLightBgCorner = document.querySelector("#mask-lens-light-bg-corner");
const maskSubtractionStatus = document.querySelector("#mask-subtraction-status");
const maskSubtractionStage = document.querySelector(".mask-subtraction-stage");
const maskSubtractionPreview = document.querySelector("#mask-subtraction-preview");
const maskSubtractionEmpty = document.querySelector("#mask-subtraction-empty");
const maskCutoutMtfShadows = document.querySelector("#mask-cutout-mtf-shadows");
const maskCutoutMtfMidtones = document.querySelector("#mask-cutout-mtf-midtones");
const maskCutoutMtfHighlights = document.querySelector("#mask-cutout-mtf-highlights");
const maskCutoutMtfShadowsValue = document.querySelector("#mask-cutout-mtf-shadows-value");
const maskCutoutMtfMidtonesValue = document.querySelector("#mask-cutout-mtf-midtones-value");
const maskCutoutMtfHighlightsValue = document.querySelector("#mask-cutout-mtf-highlights-value");
const maskCutoutMtfAuto = document.querySelector("#mask-cutout-mtf-auto");
const maskCutoutMtfApply = document.querySelector("#mask-cutout-mtf-apply");
const maskSubtractionMtfShadows = document.querySelector("#mask-subtraction-mtf-shadows");
const maskSubtractionMtfMidtones = document.querySelector("#mask-subtraction-mtf-midtones");
const maskSubtractionMtfHighlights = document.querySelector("#mask-subtraction-mtf-highlights");
const maskSubtractionMtfShadowsValue = document.querySelector("#mask-subtraction-mtf-shadows-value");
const maskSubtractionMtfMidtonesValue = document.querySelector("#mask-subtraction-mtf-midtones-value");
const maskSubtractionMtfHighlightsValue = document.querySelector("#mask-subtraction-mtf-highlights-value");
const maskSubtractionMtfAuto = document.querySelector("#mask-subtraction-mtf-auto");
const maskSubtractionMtfApply = document.querySelector("#mask-subtraction-mtf-apply");
const maskStatus = document.querySelector("#mask-status");
const maskConjugatePoint = document.querySelector("#mask-conjugate-point");
const maskConjugatePlane = document.querySelector("#mask-conjugate-plane");
const maskConjugateSource = document.querySelector("#mask-conjugate-source");
const maskConjugateClickMode = document.querySelector("#mask-conjugate-click-mode");
const maskClearConjugate = document.querySelector("#mask-clear-conjugate");
const maskConjugateStatus = document.querySelector("#mask-conjugate-status");
const maskConjugateLayer = document.querySelector("#mask-conjugate-layer");
const maskImageCanvas = document.querySelector("#mask-image-canvas");
const maskOverlayCanvas = document.querySelector("#mask-overlay-canvas");
const maskEmpty = document.querySelector("#mask-empty");
const lensProgress = document.querySelector("#lens-progress");
const lensProgressLabel = document.querySelector("#lens-progress-label");
const lensProgressPercent = document.querySelector("#lens-progress-percent");
const lensParametricProgressLabel = document.querySelector("#lens-parametric-progress-label");
const lensParametricProgressPercent = document.querySelector("#lens-parametric-progress-percent");
const lensParametricProgressBar = document.querySelector("#lens-parametric-progress-bar");
const lensPixelatedProgressLabel = document.querySelector("#lens-pixelated-progress-label");
const lensPixelatedProgressPercent = document.querySelector("#lens-pixelated-progress-percent");
const lensPixelatedProgressBar = document.querySelector("#lens-pixelated-progress-bar");
const lensGenerateCode = document.querySelector("#lens-generate-code");
const lensClearCode = document.querySelector("#lens-clear-code");
const lensRunModel = document.querySelector("#lens-run-model");
const lensRunGpuModel = document.querySelector("#lens-run-gpu-model");
const lensRunAllGpuModel = document.querySelector("#lens-run-all-gpu-model");
const lensStopRun = document.querySelector("#lens-stop-run");
const lensGeneratedCode = document.querySelector("#lens-generated-code");
const lensEditCode = document.querySelector("#lens-edit-code");
const lensScriptEditor = document.querySelector("#lens-script-editor");
const lensScriptCutoutCanvas = document.querySelector("#lens-script-cutout-canvas");
const lensScriptCutoutOverlay = document.querySelector("#lens-script-cutout-overlay");
const lensScriptCutoutEmpty = document.querySelector("#lens-script-cutout-empty");
const lensScriptSubtractedCanvas = document.querySelector("#lens-script-subtracted-canvas");
const lensScriptSubtractedEmpty = document.querySelector("#lens-script-subtracted-empty");
const lensScriptCutoutStatus = document.querySelector("#lens-script-cutout-status");
const lensChainSelect = document.querySelector("#lens-chain-select");
const lensChainSelectLabel = document.querySelector("#lens-chain-select-label");
const lensResultRating = document.querySelector("#lens-result-rating");
const lensMassSummary = document.querySelector("#lens-mass-summary");
const lensSigmaMaxAutoValue = document.querySelector("#lens-sigma-max-auto-value");
const lensModelForm = document.querySelector('[data-form="lens-model"]');
const lensPriorPanel = document.querySelector(".lens-prior-panel");
const lensCodePanel = document.querySelector(".lens-code-panel");
const lensExtraMassProfile = document.querySelector("#lens-extra-mass-profile");
const lensExtraMassAdd = document.querySelector("#lens-extra-mass-add");
const lensExtraMassList = document.querySelector("#lens-extra-mass-list");
const lensPlaneMassComponentsInput = document.querySelector("#lens-plane-mass-components");
const lensLightExternalMode = lensModelForm?.querySelector('[name="lens_light_external_mode"]');
const lensLightExternalPath = lensModelForm?.querySelector('[name="lens_light_external_path"]');
const lensLightGaussianCount = lensModelForm?.querySelector('[name="n_gauss_lens"]');
const lensLightCenterMaxOffset = lensModelForm?.querySelector('[name="lens_light_center_max_offset"]');
const sourcePositiveInput = lensModelForm?.querySelector('[name="source_positive"]');
const sourceNonlinearBrightnessInput = lensModelForm?.querySelector('[name="source_nonlinear_brightness"]');
const sourceLogBrightnessInput = lensModelForm?.querySelector('[name="source_log_brightness"]');
const lensLightHandoffStatus = document.querySelector("#lens-light-handoff-status");
const dsplEnabled = document.querySelector("#dspl-enabled");
const dsplPriorBlocks = document.querySelectorAll(".dspl-prior-block");

if (maskPreviewGrid && maskLensLightPanel) {
  maskPreviewGrid.append(maskLensLightPanel);
}

const imageView = {
  scale: 1,
  x: 0,
  y: 0,
  dragging: false,
  lastX: 0,
  lastY: 0,
  original: null,
  rawPayload: null,
  previewSrc: "",
  sourcePath: "",
  originalFilePath: "",
  rmsPath: "",
  dataCutoutPath: "",
  cutoutCenter: null,
  bgBoxCenter: null,
  centroid: null,
  shape: null,
  sourceShape: null,
  displayStride: 1,
  pixelScaleArcsec: null,
  pointerMoved: false,
  panDragAxis: null,
  needsLayoutCenter: false,
};

const euclidState = {
  data: null,
  width: 0,
  height: 0,
  vmin: 0,
  vmax: 1,
  zeropoint: null,
  zeropoints: { ab: null, vega: null },
  zeropointSource: {},
  fluxUnit: "",
  pixelScaleArcsec: null,
  band: "",
  source: "",
  displayMode: "flux",
  imageOptions: {},
  currentImageKey: "",
  path: "",
  rmsMapPath: "",
  rmsBundlePath: "",
  jobId: "",
  photometryEnabled: false,
  photozEnabled: false,
  photozBands: new Set(PHOTOZ_BAND_LIST),
  lastPoint: null,
};

const cutoutState = {
  fitsPath: "",
  previewUrl: "",
  shape: null,
  bounds: null,
};

const psfState = {
  mode: "input",
  inputPath: "",
  detectedStars: [],
  selectedIds: new Set(),
  previewView: "product",
  fullImageSourcePath: "",
  fullImagePreviewSrc: "",
  fullImageShape: null,
  fullImageDisplayStride: 1,
  fullImageRenderSeq: 0,
  mainPreviewData: null,
  jobId: "",
  jobStatus: null,
  jobRunning: false,
  pollToken: 0,
};

const lensState = {
  previewData: null,
  chainPreviews: [],
  selectedChain: "",
  panelAutoMtfKey: "",
  progress: null,
  jobId: "",
  jobRunning: false,
  generatedCode: "",
  modelConfig: null,
  scriptEditMode: false,
  scriptEdited: false,
  scriptStale: false,
  scriptStaleReason: "",
  lensLightPriorAuto: true,
  rating: 0,
};

const lensScriptCutoutState = {
  image: null,
  previewSrc: "",
  matplotlibPreviewSrc: "",
  previewToken: "",
  subtractedImage: null,
  subtractedMatplotlibPreviewSrc: "",
  subtractedPreviewToken: "",
  subtractedStaleMessage: "",
  refreshTimer: null,
  masks: {},
  conjugatePointsSource1: [],
  conjugatePointsSource2: [],
};

const lensPlaneMassState = {
  components: [],
  nextId: 1,
  activePickerId: "",
  locating: false,
};
let lensCodeHeightObserver = null;

const projectThumbnailState = {
  src: "",
  image: null,
  loading: null,
  failed: false,
  dirty: false,
  version: "",
};

const lensPixelGridManual = {
  source: false,
  source2: false,
};

const maskState = {
  width: 0,
  height: 0,
  data: null,
  tool: "line",
  mode: "add",
  type: "mask_1",
  drawing: false,
  lastPoint: null,
  polygonPoints: [],
  conjugateMode: false,
  conjugatePlane: "source1",
  conjugateMeasureSource: "mask",
  conjugateClickMode: "brightest",
  conjugatePoints: [],
  conjugatePointsSource2: [],
  bounds: null,
  sourcePath: "",
  sourceImage: null,
  previewSrc: "",
  previewSource: "image",
  previewColormap: "twilight",
  savedPath: "",
  useMatplotlibPreview: true,
  autosaveTimer: null,
  autosaveInFlight: false,
  autosaveQueued: false,
  matplotlibPreviewTimer: null,
  matplotlibPreviewToken: 0,
  lensLightPreviewTimer: null,
  lensLightPreviewSeq: 0,
  lensLightPreviewToken: "",
  loadToken: 0,
  lensLightJobId: "",
  lensLightPreviewUrl: "",
  lensLightJobStatus: null,
  lensLightJobRunning: false,
  lensLightImage: null,
  lensLightResultChoice: "unconstrained",
};

const projectState = {
  id: "",
  name: "",
  folder: "",
  comment: "",
  restoring: false,
  saveTimer: null,
  saveRetryTimer: null,
  saveRetryAttempt: 0,
  thumbnailTimer: null,
  cutoutPreviewTimer: null,
  thumbnailRefreshAfterRestore: false,
};

let projectContextMenu = null;
let projectChooserView = localStorage.getItem("herculens-project-view") === "list" ? "list" : "thumbnail";
let projectChooserProjects = [];
let projectChooserQuery = "";
let projectChooserMinimumScore = "all";
let projectChooserSortMode = localStorage.getItem("herculens-project-sort") || "score";
let lensLightHandoffKey = "";

function setActivity(message = "Ready.", busy = false) {
  activityStatus.textContent = message;
  activityStatus.classList.toggle("busy", busy);
  activityStatus.setAttribute("aria-busy", String(Boolean(busy)));
  const normalized = String(message || "").toLowerCase();
  const tone = busy
    ? "busy"
    : /failed|error|unavailable|missing|cannot/.test(normalized)
      ? "error"
      : /ready|loaded|saved|complete|generated|started/.test(normalized)
        ? "success"
        : "neutral";
  activityStatus.dataset.tone = tone;
  updateWorkflowStatus();
}

function appendLog(message) {
  const timestamp = new Date().toLocaleTimeString();
  runLog.textContent += `\n[${timestamp}] ${message}`;
  runLog.scrollTop = runLog.scrollHeight;
}

function progressBarText(fraction) {
  const value = Math.min(1, Math.max(0, Number(fraction ?? 0)));
  const width = 24;
  const filled = Math.round(width * value);
  const empty = width - filled;
  const percent = Math.round(100 * value);
  return `[${"#".repeat(filled)}${"-".repeat(empty)}] ${percent}%`;
}

function updateLogProgress(key, progress, state = "running") {
  const timestamp = new Date().toLocaleTimeString();
  const bar = progressBarText(progress?.fraction ?? 0);
  const message = progress?.message || state;
  const nextLine = `[${timestamp}] ${key}: ${bar} ${message}`;
  const lines = runLog.textContent.split("\n");
  const prefix = `${key}: [`;
  let lineIndex = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i].includes(prefix)) {
      lineIndex = i;
      break;
    }
  }
  if (lineIndex >= 0) {
    lines[lineIndex] = nextLine;
  } else {
    lines.push(nextLine);
  }
  runLog.textContent = lines.join("\n");
  runLog.scrollTop = runLog.scrollHeight;
}

function updateLensLogProgress(jobId, progress, state = "running") {
  const stages = progress?.stages || {};
  if (!stages.parametric && !stages.pixelated) {
    updateLogProgress(`Lens job ${jobId}`, progress, state);
    return;
  }

  updateLogProgress(
    `Lens job ${jobId} parametric`,
    stages.parametric || { fraction: 0, message: "Parametric SVI pending." },
    state,
  );
  updateLogProgress(
    `Lens job ${jobId} pixelated`,
    stages.pixelated || { fraction: 0, message: "Pixelated SVI pending." },
    state,
  );
  // Only present when the exact two-image solver stage is enabled.
  if (stages.solver) {
    updateLogProgress(`Lens job ${jobId} solver`, stages.solver, state);
  }
}

function updatePsfLogProgress(jobId, data) {
  const latestLoss = Number(data.progress?.latest_loss);
  const lossText = Number.isFinite(latestLoss) ? `, loss ${latestLoss.toPrecision(5)}` : "";
  const progress = {
    ...(data.progress || {}),
    fraction: data.progress?.fraction ?? (data.state === "completed" ? 1 : 0),
    message: `${data.message || `PSF job ${data.state || "running"}`}${lossText}`,
  };
  updateLogProgress(`PSF job ${jobId}`, progress, data.state || "running");
}

function setBackendState(state, text) {
  backendStatus.classList.remove("idle", "ready", "error");
  backendStatus.classList.add(state);
  backendText.textContent = text;
  if (checkBackendButton) {
    checkBackendButton.dataset.state = state;
    checkBackendButton.title = `Backend: ${text}`;
  }
}

function updateWorkflowStatus() {
  const euclidReady = Boolean(
    euclidState.path || euclidState.currentImageKey || Object.keys(euclidState.imageOptions || {}).length,
  );
  const imageReady = Boolean(cutoutState.fitsPath || imageView.dataCutoutPath);
  const psfReady = Boolean(psfState.inputPath || psfInputPath?.value.trim() || psfState.mainPreviewData);
  const maskReady = Boolean(maskState.savedPath);
  const lensResultReady = Boolean(lensState.chainPreviews.length || lensState.previewData);
  const lensScriptReady = Boolean(lensState.generatedCode.trim());
  const statusByPage = {
    "euclid-cutout": euclidReady
      ? { state: "complete", label: "Cutout available" }
      : { state: "optional", label: "Optional" },
    "image-preprocess": imageReady
      ? { state: "complete", label: "Cutout ready" }
      : imageView.original
        ? { state: "ready", label: "Select cutout" }
        : { state: "pending", label: "Needs image" },
    "psf-fit": psfReady
      ? { state: "complete", label: "PSF ready" }
      : imageReady
        ? { state: "ready", label: "Ready to configure" }
        : { state: "pending", label: "Needs cutout" },
    mask: maskReady
      ? { state: "complete", label: "Mask saved" }
      : imageReady
        ? { state: "ready", label: "Ready to draw" }
        : { state: "pending", label: "Needs cutout" },
    "lens-model": lensResultReady
      ? { state: "complete", label: "Result available" }
      : lensState.scriptStale
        ? { state: "ready", label: "Regenerate script" }
      : lensScriptReady
        ? { state: "ready", label: "Script generated" }
        : { state: "pending", label: maskReady ? "Ready to configure" : "Not generated" },
  };

  document.querySelectorAll(".nav-tab").forEach((button) => {
    const status = statusByPage[button.dataset.page] || { state: "pending", label: "Pending" };
    button.dataset.stageState = status.state;
    button.querySelector(".tab-state").textContent = status.label;
    button.setAttribute("aria-current", button.classList.contains("active") ? "step" : "false");
    button.title = `${pageCopy[button.dataset.page]?.title || button.dataset.page}: ${status.label}`;
  });

  if (currentProjectState) {
    currentProjectState.textContent = projectState.folder
      ? lensResultReady
        ? "Result available"
        : "Saved workspace"
      : "Unsaved workspace";
  }
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  const isDark = theme === "dark";
  themeIcon.textContent = isDark ? "☀" : "◐";
  themeLabel.textContent = isDark ? "Light" : "Dark";
  localStorage.setItem("herculens-gui-theme", theme);
}

function applySettings(settings = {}, settingsFile = "") {
  settingsHerculensPython.value = settings.herculens_python || "";
  settingsEuclidPython.value = settings.euclid_python || "";
  settingsPhotozPython.value = settings.photoz_python || "";
  settingsPhosphorosRoot.value = settings.phosphoros_root || "";
  settingsPath.textContent = settingsFile || "Local runtime settings";
}

function collectSettingsPayload() {
  return {
    herculens_python: settingsHerculensPython.value.trim(),
    euclid_python: settingsEuclidPython.value.trim(),
    photoz_python: settingsPhotozPython.value.trim(),
    phosphoros_root: settingsPhosphorosRoot.value.trim(),
  };
}

async function loadSettings() {
  const data = await callBackend("/api/settings");
  applySettings(data.settings || {}, data.settings_path || "");
  return data;
}

async function openSettingsPanel() {
  settingsModal.hidden = false;
  settingsStatus.textContent = "Loading settings...";
  try {
    const data = await loadSettings();
    settingsStatus.textContent = `Backend Python: ${data.backend_python || ""}`;
  } catch (error) {
    settingsStatus.textContent = `Settings load failed: ${error.message}`;
  }
}

function closeSettingsPanel() {
  settingsModal.hidden = true;
}

async function saveSettings(event) {
  event.preventDefault();
  settingsStatus.textContent = "Saving settings...";
  try {
    const data = await callBackend("/api/settings/save", collectSettingsPayload());
    applySettings(data.settings || {}, data.settings_path || "");
    settingsStatus.textContent = data.message || "Settings saved.";
    appendLog("Settings saved.");
  } catch (error) {
    settingsStatus.textContent = `Settings save failed: ${error.message}`;
    appendLog(`Settings: ${error.message}`);
  }
}

async function checkRuntimePython(key) {
  const pathByKey = {
    herculens_python: settingsHerculensPython.value.trim(),
    euclid_python: settingsEuclidPython.value.trim(),
    photoz_python: settingsPhotozPython.value.trim(),
  };
  settingsStatus.textContent = `Checking ${key}...`;
  try {
    const data = await callBackend("/api/settings/check", { key, path: pathByKey[key] || "" });
    settingsStatus.textContent = `${data.path}\n${data.stdout || "ok"}`;
    appendLog(`Settings check ${key}: ok.`);
  } catch (error) {
    settingsStatus.textContent = `Check failed: ${error.message}`;
    appendLog(`Settings check ${key}: ${error.message}`);
  }
}

function activePageId() {
  return document.querySelector(".page.active")?.id ?? "euclid-cutout";
}

function dirname(path) {
  const text = String(path || "").trim().replace(/\\/g, "/");
  if (!text || !text.includes("/")) return "";
  return text.replace(/\/+$/, "").replace(/\/[^/]*$/, "") || "/";
}

function currentProjectFolderPath(options = {}) {
  const inputFolder = folderPath.value.trim();
  if (options.preferInput || activePageId() === "image-preprocess") {
    return inputFolder || projectState.folder || "";
  }
  return projectState.folder || inputFolder || "";
}

function currentProjectFolderName() {
  const folder = currentProjectFolderPath().replace(/\\/g, "/").replace(/\/+$/, "");
  return folder ? folder.split("/").pop() : "";
}

function normalizedProjectRating(value) {
  const rating = Number(value);
  if (!Number.isFinite(rating)) return 0;
  return Math.max(0, Math.min(5, Math.round(rating)));
}

function currentLensRating() {
  return normalizedProjectRating(lensResultRating ? lensResultRating.value : lensState.rating);
}

function normalizeLensResultUrl(url) {
  const text = String(url || "").trim();
  if (!text || text.startsWith("http://") || text.startsWith("https://") || text.startsWith("data:")) return text;
  if (text.startsWith("/runs/")) return text;
  const folderName = currentProjectFolderName();
  if (!folderName) return text;
  if (text.startsWith("/lens_model_result/")) {
    return `/runs/${encodeURIComponent(folderName)}${text}`;
  }
  if (text.startsWith("lens_model_result/")) {
    return `/runs/${encodeURIComponent(folderName)}/${text}`;
  }
  return text;
}

function normalizeProjectAssetUrl(url) {
  const text = String(url || "").trim();
  if (!text || /^(https?:|data:|blob:)/i.test(text)) return text;
  const normalized = text.replace(/\\/g, "/");
  if (normalized.startsWith("/runs/")) return normalized;
  const runsIndex = normalized.indexOf("/runs/");
  if (runsIndex >= 0) return normalized.slice(runsIndex);
  return normalized;
}

function normalizeLensPreviewData(data = {}) {
  if (!data || typeof data !== "object") return data;
  const normalized = { ...data };
  ["preview_url", "image_url", "figure_url", "losses_url"].forEach((key) => {
    if (normalized[key]) normalized[key] = normalizeLensResultUrl(normalized[key]);
  });
  if (Array.isArray(normalized.chain_previews)) {
    normalized.chain_previews = normalized.chain_previews.map((item) => {
      if (!item || typeof item !== "object") return item;
      const fixed = { ...item };
      ["preview_url", "parametric_url", "pixelated_url", "semilinear_url"].forEach((key) => {
        if (fixed[key]) fixed[key] = normalizeLensResultUrl(fixed[key]);
      });
      return fixed;
    });
  }
  return normalized;
}

function currentProjectLensPreviewData() {
  const previewData = normalizeLensPreviewData(lensState.previewData);
  return previewData && projectReferenceBelongsHere(previewData) ? previewData : null;
}

function clearLensModelPreview(message = "No model result loaded for this project.") {
  lensState.previewData = null;
  lensState.chainPreviews = [];
  lensState.selectedChain = "";
  lensState.panelAutoMtfKey = "";
  const preview = document.querySelector('[data-preview="lens-model"]');
  if (preview) {
    preview.innerHTML = `<span class="empty-list">${message}</span>`;
  }
  renderLensChainSelector();
  renderLensMassSummary(null);
}

function resetLensRuntimeState() {
  clearLensModelPreview();
  setLensPlaneMassComponents([], { markStale: false, save: false });
  const applyMaskInput = lensModelForm?.querySelector('input[name="apply_lensed_arc_mask"]');
  if (applyMaskInput) applyMaskInput.checked = true;
  const sisThetaHighInput = lensModelForm?.querySelector('input[name="sis_theta_high"]');
  if (sisThetaHighInput) sisThetaHighInput.value = String(DEFAULT_SIS_THETA_HIGH);
  lensState.progress = null;
  lensState.jobId = "";
  lensState.jobRunning = false;
  lensProgress.hidden = true;
  lensProgressLabel.textContent = "";
  lensProgressPercent.textContent = "";
  lensParametricProgressLabel.textContent = "Parametric SVI pending.";
  lensParametricProgressPercent.textContent = "0%";
  lensParametricProgressBar.style.width = "0%";
  lensPixelatedProgressLabel.textContent = "Pixelated SVI pending.";
  lensPixelatedProgressPercent.textContent = "0%";
  lensPixelatedProgressBar.style.width = "0%";
  setLensRunState(false);
}

function collectTextValues(value, output = []) {
  if (typeof value === "string") {
    output.push(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => collectTextValues(item, output));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectTextValues(item, output));
  }
  return output;
}

function projectReferenceBelongsHere(value) {
  const projectFolder = currentProjectFolderPath().replace(/\\/g, "/").replace(/\/+$/, "");
  const projectFolderName = currentProjectFolderName();
  const refs = collectTextValues(value).filter((text) => {
    const normalized = text.replace(/\\/g, "/");
    return normalized.includes("/runs/") || (projectFolder && normalized.includes(projectFolder));
  });
  if (!refs.length || !projectFolderName) return false;
  return refs.every((text) => {
    const normalized = text.replace(/\\/g, "/");
    return normalized.includes(`${projectFolder}/`) || normalized.endsWith(projectFolder) || normalized.includes(`/runs/${projectFolderName}/`);
  });
}

function projectReferenceHasForeignRun(value) {
  const projectFolderName = currentProjectFolderName();
  if (!projectFolderName) return false;
  return collectTextValues(value).some((text) => {
    const normalized = text.replace(/\\/g, "/");
    const matches = normalized.matchAll(/\/runs\/([^/]+)/g);
    for (const match of matches) {
      if (decodeURIComponent(match[1]) !== projectFolderName) return true;
    }
    return false;
  });
}

function projectReferenceIsCurrentOrExternal(value) {
  return !projectReferenceHasForeignRun(value);
}

function filterConjugatePointsForProject(points, lensLightValid) {
  const list = Array.isArray(points) ? points : [];
  return list.filter((point) => {
    if (projectReferenceHasForeignRun(point)) return false;
    const source = String(point?.measurement_source || "").toLowerCase();
    if (!lensLightValid && (source === "lens_light_subtracted" || source === "subtracted")) return false;
    return true;
  });
}

function resetMaskRuntimeState() {
  if (maskState.autosaveTimer) {
    window.clearTimeout(maskState.autosaveTimer);
  }
  maskState.loadToken += 1;
  maskState.width = 0;
  maskState.height = 0;
  maskState.data = null;
  maskState.drawing = false;
  maskState.lastPoint = null;
  maskState.polygonPoints = [];
  maskState.conjugateMode = false;
  maskState.conjugatePlane = "source1";
  maskState.conjugateMeasureSource = "mask";
  maskState.conjugateClickMode = "brightest";
  maskState.conjugatePoints = [];
  maskState.conjugatePointsSource2 = [];
  maskState.bounds = null;
  maskState.sourcePath = "";
  maskState.sourceImage = null;
  maskState.previewSrc = "";
  maskState.previewSource = "image";
  maskState.previewToken = "";
  maskState.savedPath = "";
  maskState.autosaveTimer = null;
  maskState.autosaveInFlight = false;
  maskState.autosaveQueued = false;
  maskState.lensLightJobId = "";
  maskState.lensLightPreviewUrl = "";
  maskState.lensLightJobStatus = null;
  maskState.lensLightImage = null;
  maskLensLightSigmaMax.value = "auto";
  maskLensLightSigmaMax.dataset.dynamicDefault = "true";
  maskLensLightCenterMaxOffset.value = "0.4";
  lensScriptCutoutState.matplotlibPreviewSrc = "";
  if (lensScriptCutoutState.refreshTimer) {
    window.clearTimeout(lensScriptCutoutState.refreshTimer);
    lensScriptCutoutState.refreshTimer = null;
  }
  lensScriptCutoutState.conjugatePointsSource1 = [];
  lensScriptCutoutState.conjugatePointsSource2 = [];
  maskConjugatePoint.classList.remove("active");
  maskConjugatePoint.textContent = "Conjugate point";
  if (maskConjugatePlane) maskConjugatePlane.value = "source1";
  if (maskConjugateSource) maskConjugateSource.value = "mask";
  if (maskPreviewSource) maskPreviewSource.value = "image";
  if (maskConjugateClickMode) maskConjugateClickMode.value = "brightest";
  updateConjugateStatus();
  renderMaskOverlay();
  setMaskSubtractionPreview("", null);
}

function setPage(pageId) {
  document.body.dataset.activePage = pageId;
  document.querySelectorAll(".nav-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === pageId);
  });

  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === pageId);
  });

  pageTitle.textContent = pageCopy[pageId].title;
  pageSubtitle.textContent = pageCopy[pageId].subtitle;
  if (pageId === "image-preprocess" && projectState.folder) {
    folderPath.value = projectState.folder;
    if (imageView.needsLayoutCenter) {
      imageView.needsLayoutCenter = false;
      centerImagePreviewAfterLayout();
    }
    scheduleProjectSave();
  }
  if (pageId === "mask") {
    const targetMaskSource = currentMaskSourcePath();
    if (projectState.folder && (!maskState.data || (targetMaskSource && maskState.sourcePath !== targetMaskSource))) {
      void loadProjectMask(maskState.type, { silent: true });
    } else if (maskState.data) {
      requestAnimationFrame(() => {
        fitMaskCanvasToStage();
        refreshVisibleMaskPreview();
      });
    }
  }
  if (pageId === "lens-model") {
    syncLensLightPriorFromMask(maskState.lensLightJobStatus, { silent: true });
    void loadLensScriptCutoutPreview({ silent: true });
    requestAnimationFrame(syncLensCodePreviewHeight);
    if (!projectState.restoring) {
      void restoreLensModelResultFromProject({ silent: true });
    }
  }
  updateWorkflowStatus();
  requestAnimationFrame(renderProjectCanvasAxes);
}

function collectFormPayload(form) {
  const projectFolderValue = currentProjectFolderPath();
  const projectDisplayName = canonicalProjectName({
    project_id: projectState.id,
    project_name: projectState.name,
    project_folder: projectFolderValue || projectState.folder,
  });
  const payload = {
    data_folder: projectFolderValue,
    project_id: projectState.id || undefined,
    project_name: projectDisplayName || undefined,
    project_folder: projectFolderValue || undefined,
  };

  if (form?.dataset?.form === "lens-model") {
    syncLensPlaneMassComponentsInput();
  }
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

  if (form?.dataset?.form === "lens-model") {
    const rmsMode = payload.rms_mode === "scalar" ? "scalar" : "map";
    payload.use_scalar_rms_loguniform = rmsMode === "scalar";
    payload.use_existing_rms_map = rmsMode === "map";
    payload.automatic_background_rms = true;
    payload.apply_lensed_arc_mask = payload.apply_lensed_arc_mask !== false;
    payload.dspl_lensed_arcs_unmasked = !payload.apply_lensed_arc_mask;
    expandSymmetricPriorLimit(payload, "center_lim", "center_low", "center_high");
    expandSymmetricPriorLimit(payload, "e_lim", "e_low", "e_high");
    payload.shear_strength_low = 0;
    payload.shear_strength_high = Math.max(0, lensMassNumber(payload.shear_lim, 0.2));
    const sourceNpixProvided =
      payload.source_pixel_grid_shape !== "" &&
      payload.source_pixel_grid_shape !== null &&
      payload.source_pixel_grid_shape !== undefined &&
      Number.isFinite(Number(payload.source_pixel_grid_shape));
    const source2NpixProvided =
      payload.source2_pixel_grid_shape !== "" &&
      payload.source2_pixel_grid_shape !== null &&
      payload.source2_pixel_grid_shape !== undefined &&
      Number.isFinite(Number(payload.source2_pixel_grid_shape));
    const manualSourcePixelGrid = lensPixelGridManual.source || sourceNpixProvided;
    const manualSource2PixelGrid = lensPixelGridManual.source2 || source2NpixProvided;
    payload.use_best_pixel_size = !(manualSourcePixelGrid || manualSource2PixelGrid);
    payload.source_pixel_grid_manual = manualSourcePixelGrid;
    payload.source2_pixel_grid_manual = manualSource2PixelGrid;
    payload.display_mtf = {
      shadows: Number(mtfShadows.value),
      midtones: Number(mtfMidtones.value),
      highlights: Number(mtfHighlights.value),
    };
  }

  payload.conjugate_points_source1 = arcsecConjugatePoints(maskState.conjugatePoints);
  payload.conjugate_points_source2 = arcsecConjugatePoints(maskState.conjugatePointsSource2);

  if (payload.point_source_profile === "IMAGE_POSITIONS") {
    payload.point_source_measured_fluxes = maskState.conjugatePoints.map((point) => {
      const flux = Number(point.measured_flux);
      return Number.isFinite(flux) && flux > 0 ? flux : null;
    });
  }

  return payload;
}

function expandSymmetricPriorLimit(payload, limKey, lowKey, highKey) {
  if (payload[limKey] === "" || payload[limKey] === null || payload[limKey] === undefined) return;
  const lim = Math.abs(Number(payload[limKey]));
  if (!Number.isFinite(lim)) return;
  payload[lowKey] = -lim;
  payload[highKey] = lim;
}

function arcsecConjugatePoints(points) {
  return (points || [])
    .filter((point) => point.arcsec && Number.isFinite(point.arcsec.x) && Number.isFinite(point.arcsec.y))
    .map((point) => [point.arcsec.x, point.arcsec.y]);
}

function setImagePreviewMetadata(data = {}) {
  const shape = Array.isArray(data.image_shape) ? data.image_shape.map(Number) : null;
  imageView.shape = shape && shape.length >= 2 && shape.every(Number.isFinite) ? shape.slice(0, 2) : imageView.shape;
  const payloadShape = data.image_payload?.source_shape || data.source_shape;
  const sourceShape = Array.isArray(payloadShape) ? payloadShape.map(Number) : null;
  if (sourceShape && sourceShape.length >= 2 && sourceShape.every(Number.isFinite)) {
    imageView.sourceShape = sourceShape.slice(0, 2);
  }
  const displayStride = Number(data.image_payload?.display_stride ?? data.display_stride);
  if (Number.isFinite(displayStride) && displayStride >= 1) {
    imageView.displayStride = Math.max(1, displayStride);
  }
  const pixelScale = Number(data.image_payload?.pixel_scale_arcsec ?? data.pixel_scale_arcsec);
  imageView.pixelScaleArcsec = Number.isFinite(pixelScale) && pixelScale > 0 ? pixelScale : imageView.pixelScaleArcsec;
  updateLensSigmaMaxAutoHint();
}

function applyRegisteredImageArtifacts(data = {}, options = {}) {
  const originalPath = data.original_path || data.psf_source_path || "";
  const cutoutPath = data.data_cutout_path || "";
  if (originalPath) {
    imageView.originalFilePath = originalPath;
    imageView.sourcePath = originalPath;
    inputImagePath.value = originalPath;
    selectedImageName.textContent = originalPath;
    psfSciencePath.value = originalPath;
  } else if (data.psf_source_path) {
    psfSciencePath.value = data.psf_source_path;
  }
  if (cutoutPath) {
    imageView.dataCutoutPath = cutoutPath;
    cutoutState.fitsPath = cutoutPath;
    cutoutState.previewUrl = data.data_cutout_preview_url || cutoutState.previewUrl;
    cutoutState.shape = data.data_cutout_shape || data.image_shape || imageView.shape || cutoutState.shape;
    cutoutState.bounds = data.data_cutout_bounds || (
      cutoutState.shape?.length >= 2
        ? { x0: 0, y0: 0, x1: cutoutState.shape[1], y1: cutoutState.shape[0], width: cutoutState.shape[1], height: cutoutState.shape[0] }
        : cutoutState.bounds
    );
    if (options.showCutoutPanel) {
      cutoutPreviewLabel.textContent = PathName(cutoutPath);
      cutoutPreviewPanel.hidden = false;
    }
    updateLensSigmaMaxAutoHint();
  }
  if (data.rms_map_path) {
    imageView.rmsPath = data.rms_map_path;
    inputRmsPath.value = data.rms_map_path;
    selectedRmsName.textContent = data.rms_map_path;
  }
}

function lensSigmaMaxInput() {
  return document.querySelector('[data-form="lens-model"] [name="lens_sigma_max"]');
}

function lensSigmaMaxAutoArcsec() {
  const shape = cutoutState.shape || imageView.shape || (imageView.original ? [imageView.original.height, imageView.original.width] : null);
  const pixelScale = Number(imageView.pixelScaleArcsec);
  if (!shape || shape.length < 2 || !Number.isFinite(pixelScale) || pixelScale <= 0) return null;
  const minSide = Math.min(Number(shape[0]), Number(shape[1]));
  if (!Number.isFinite(minSide) || minSide <= 0) return null;
  return 0.5 * minSide * pixelScale;
}

function updateLensSigmaMaxAutoHint() {
  if (!lensSigmaMaxAutoValue) return;
  const inputValue = lensSigmaMaxInput()?.value.trim().toLowerCase() || "auto";
  const autoValue = lensSigmaMaxAutoArcsec();
  const isAuto = ["", "auto", "half", "half_image", "half image"].includes(inputValue);
  if (!isAuto) {
    lensSigmaMaxAutoValue.textContent = "manual";
    return;
  }
  lensSigmaMaxAutoValue.textContent = autoValue === null ? "auto = --" : `auto = ${autoValue.toFixed(4)} arcsec`;
}

function maskLensLightSigmaMaxDefaultArcsec() {
  const shape =
    maskState.width > 0 && maskState.height > 0
      ? [maskState.height, maskState.width]
      : cutoutState.shape || imageView.shape || (imageView.original ? [imageView.original.height, imageView.original.width] : null);
  const pixelScale = currentImagePixelScaleArcsec();
  if (!shape || shape.length < 2 || pixelScale === null) return null;
  const minSide = Math.min(Number(shape[0]), Number(shape[1]));
  if (!Number.isFinite(minSide) || minSide <= 0) return null;
  return MASK_LENS_LIGHT_SIGMA_MAX_FRACTION * minSide * pixelScale;
}

function updateMaskLensLightSigmaMaxDefault(options = {}) {
  const currentValue = maskLensLightSigmaMax.value.trim().toLowerCase();
  const isAuto = ["", "auto", "half", "half_image", "half image"].includes(currentValue);
  const usesDynamicDefault = maskLensLightSigmaMax.dataset.dynamicDefault === "true";
  if (!options.force && !isAuto && !usesDynamicDefault) return;
  const sigmaMax = maskLensLightSigmaMaxDefaultArcsec();
  if (sigmaMax === null) {
    maskLensLightSigmaMax.value = "auto";
  } else {
    maskLensLightSigmaMax.value = String(Number(sigmaMax.toFixed(4)));
  }
  maskLensLightSigmaMax.dataset.dynamicDefault = "true";
}

function applyFormPayload(form, payload) {
  if (!form || !payload) return;
  if (form.dataset.form === "lens-model" && payload.rms_mode === undefined) {
    payload = {
      ...payload,
      rms_mode: payload.use_scalar_rms_loguniform || payload.data?.use_scalar_rms_loguniform ? "scalar" : "map",
    };
  }
  if (form.dataset.form === "lens-model" && payload.apply_lensed_arc_mask === undefined) {
    const savedApplyMask = payload.display?.apply_lensed_arc_mask;
    const savedUnmasked = payload.dspl_lensed_arcs_unmasked ?? payload.display?.lensed_arcs_unmasked;
    payload = {
      ...payload,
      apply_lensed_arc_mask:
        savedApplyMask !== undefined ? Boolean(savedApplyMask) : savedUnmasked !== undefined ? !Boolean(savedUnmasked) : true,
    };
  }
  if (payload.data_folder !== undefined) {
    folderPath.value = String(payload.data_folder ?? "");
  }
  for (const [key, value] of Object.entries(payload)) {
    const controls = Array.from(form.querySelectorAll("[name]")).filter((control) => control.name === key);
    controls.forEach((control) => {
      if (control.type === "checkbox") {
        control.checked = Boolean(value);
      } else if (control.type === "radio") {
        control.checked = String(control.value) === String(value);
      } else if (value !== undefined && value !== null) {
        control.value = String(value);
      }
    });
  }
  if (form.dataset.form === "lens-model") {
    setLensPlaneMassComponents(payload.lens_plane_mass_components || [], {
      markStale: false,
      save: false,
    });
    setSymmetricLimitControl(form, payload, "center_lim", "center_low", "center_high");
    setSymmetricLimitControl(form, payload, "e_lim", "e_low", "e_high");
    const shearLimit = shearStrengthLimitFromPayload(payload);
    const shearControl = form.querySelector('[name="shear_lim"]');
    if (shearControl && Number.isFinite(shearLimit)) shearControl.value = String(shearLimit);
    const useBest = payload.use_best_pixel_size ?? payload.source_grid?.use_best_pixel_size;
    const manualFromSavedPayload = Boolean(payload.source_pixel_grid_manual || payload.source2_pixel_grid_manual);
    const manualFromGeneratedConfig = useBest === false;
    lensPixelGridManual.source = Boolean(payload.source_pixel_grid_manual) || manualFromGeneratedConfig || manualFromSavedPayload;
    lensPixelGridManual.source2 = Boolean(payload.source2_pixel_grid_manual) || manualFromGeneratedConfig || manualFromSavedPayload;
    updateLensSigmaMaxAutoHint();
    syncSourceNonlinearPriorControls();
  }
  syncDsplControls();
}

function setSymmetricLimitControl(form, payload, limKey, lowKey, highKey) {
  const input = form.querySelector(`[name="${limKey}"]`);
  if (!input) return;
  const direct = Number(payload[limKey]);
  if (Number.isFinite(direct)) {
    input.value = String(Math.abs(direct));
    return;
  }
  const low = Number(payload[lowKey]);
  const high = Number(payload[highKey]);
  const values = [low, high].filter(Number.isFinite);
  if (values.length) {
    input.value = String(Math.max(...values.map((value) => Math.abs(value))));
  }
}

function lensModelConfigToFormPayload(config) {
  if (!config) return {};
  const lensSigma = config.lens_sigma_lims || config.light?.lens?.sigma_lims || [];
  const sourceSigma = config.source_sigma_lims || config.light?.source?.sigma_lims || [];
  const source2Sigma = config.source2_sigma_lims || config.light?.source2?.sigma_lims || [];
  const sourceGrid = config.source_grid || {};
  const massPrior = config.mass_prior || {};
  return {
    data_folder: config.data_folder,
    dataset: config.dataset,
    dspl_enabled: config.dspl_enabled ?? config.dspl?.enabled,
    apply_lensed_arc_mask:
      config.apply_lensed_arc_mask ??
      config.display?.apply_lensed_arc_mask ??
      (config.dspl_lensed_arcs_unmasked !== undefined
        ? !Boolean(config.dspl_lensed_arcs_unmasked)
        : config.display?.lensed_arcs_unmasked !== undefined
          ? !Boolean(config.display.lensed_arcs_unmasked)
          : undefined),
    dspl_lensed_arcs_unmasked:
      config.dspl_lensed_arcs_unmasked ??
      config.display?.lensed_arcs_unmasked ??
      (config.display?.apply_lensed_arc_mask === false ? true : undefined),
    run_power_init: config.run_power_init,
    seed: config.seed ?? config.svi?.seed,
    num_chains: config.num_chains ?? config.svi?.num_chains,
    max_iter_parametric: config.max_iter_parametric ?? config.svi?.max_iter_parametric,
    max_iter_pixelated: config.max_iter_pixelated ?? config.svi?.max_iter_pixelated,
    rms_mode: config.use_scalar_rms_loguniform || config.data?.use_scalar_rms_loguniform ? "scalar" : "map",
    use_existing_rms_map: config.use_existing_rms_map ?? config.data?.use_existing_rms_map,
    use_scalar_rms_loguniform: config.use_scalar_rms_loguniform ?? config.data?.use_scalar_rms_loguniform,
    automatic_background_rms: config.automatic_background_rms ?? config.data?.automatic_background_rms,
    exposure_time: config.exposure_time ?? config.data?.exposure_time_if_missing ?? config.data?.exposure_time,
    background_rms: config.background_rms ?? config.data?.background_rms,
    background_subtract_enabled: config.background_subtract_enabled ?? config.data?.background_subtract_enabled,
    background_subtract_corner: config.background_subtract_corner ?? config.data?.background_subtract_corner,
    n_gauss_lens: config.n_gauss_lens ?? config.light?.lens?.n_gauss,
    lens_light_center_max_offset:
      config.lens_light_center_max_offset ?? config.light?.lens?.center_max_offset ?? 0.4,
    lens_light_external_mode: config.lens_light_external_mode ?? config.light?.lens?.external_mode,
    lens_light_parametric_mode: config.lens_light_parametric_mode ?? config.light?.lens?.parametric_mode,
    lens_light_pixelated_mode: config.lens_light_pixelated_mode ?? config.light?.lens?.pixelated_mode,
    lens_light_external_path: config.lens_light_external_path ?? config.light?.lens?.external_kwargs_path,
    n_gauss_source: config.n_gauss_source ?? config.light?.source?.n_gauss,
    light_profile: config.light_profile ?? config.light?.lens?.profile,
    point_source_profile: config.point_source_profile ?? config.light?.point_source?.profile,
    point_source_pos_sigma: config.point_source_pos_sigma ?? config.light?.point_source?.pos_sigma,
    point_source_pos_window: config.point_source_pos_window ?? config.light?.point_source?.pos_window,
    point_source_log10_amp_low: config.point_source_log10_amp_low ?? config.light?.point_source?.log10_amp_low,
    point_source_log10_amp_high: config.point_source_log10_amp_high ?? config.light?.point_source?.log10_amp_high,
    exact_two_image_solver: config.exact_two_image_solver?.enabled ?? config.exact_two_image_solver ?? false,
    solver_initial_theta_E: config.solver_initial_theta_E ?? config.exact_two_image_solver?.initial_q?.[0] ?? 0.8,
    solver_initial_ellipticity: config.solver_initial_ellipticity ?? config.exact_two_image_solver?.initial_q?.[1] ?? 0.24,
    lens_sigma_min: lensSigma[0],
    lens_sigma_max: lensSigma[1] ?? "auto",
    source_sigma_min: sourceSigma[0],
    source_sigma_max: sourceSigma[1],
    n_gauss_source2: config.n_gauss_source2 ?? config.light?.source2?.n_gauss,
    source2_sigma_min: source2Sigma[0],
    source2_sigma_max: source2Sigma[1],
    source_pixel_grid_shape:
      config.source_pixel_grid_shape ?? sourceGrid.pixel_grid_shape1 ?? sourceGrid.pixel_grid_shape ?? sourceGrid.default_pixel_grid_shape,
    source2_pixel_grid_shape: config.source2_pixel_grid_shape ?? sourceGrid.pixel_grid_shape2 ?? sourceGrid.default_pixel_grid_shape,
    source2_grid_scale: config.source2_grid_scale ?? sourceGrid.scale2,
    source_display: config.source_display ?? config.display?.source_display ?? "linear",
    run_semilinear: config.run_semilinear ?? true,
    source_positive: config.source_positive ?? config.pixelated_prior?.positive ?? true,
    source_nonlinear_brightness:
      config.source_nonlinear_brightness ??
      config.pixelated_prior?.nonlinear_brightness ??
      true,
    source_log_brightness: config.source_log_brightness ?? config.pixelated_prior?.log_brightness ?? false,
    multiplicative_psf_correction: config.multiplicative_psf_correction ?? false,
    eta_low: config.eta_low ?? config.dspl?.eta_prior?.low,
    eta_high: config.eta_high ?? config.dspl?.eta_prior?.high,
    sis_theta_low: config.sis_theta_low ?? config.dspl?.sis_prior?.theta_low,
    sis_theta_high: config.sis_theta_high ?? config.dspl?.sis_prior?.theta_high,
    source_grid_scale: config.source_grid_scale ?? sourceGrid.scale ?? sourceGrid.scale1,
    supersampling_factor: config.supersampling_factor ?? config.numerics?.supersampling_factor,
    mass_profile: config.mass_profile,
    lens_plane_mass_components: config.lens_plane_mass_components || [],
    center_lim: symmetricLimitFromPair(massPrior.center_low, massPrior.center_high),
    e_lim: symmetricLimitFromPair(massPrior.e_low, massPrior.e_high),
    shear_lim: shearStrengthLimitFromPayload(massPrior),
    ...massPrior,
  };
}

function symmetricLimitFromPair(low, high) {
  const values = [Number(low), Number(high)].filter(Number.isFinite);
  if (!values.length) return undefined;
  return Math.max(...values.map((value) => Math.abs(value)));
}

function shearStrengthLimitFromPayload(payload = {}) {
  const explicit = Number(payload.shear_strength_high ?? payload.shear_lim);
  if (Number.isFinite(explicit)) return Math.max(0, explicit);
  return symmetricLimitFromPair(payload.shear_low, payload.shear_high);
}

function lensMassNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function parseLensPlaneMassComponents(value) {
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return [];
    try {
      value = JSON.parse(text);
    } catch (_error) {
      return [];
    }
  }
  return Array.isArray(value) ? value : [];
}

function normalizeLensPlaneMassComponent(value, index) {
  const component = value && typeof value === "object" ? value : {};
  const requestedProfile = String(component.profile || "SIS").toUpperCase();
  const profile = requestedProfile === "SIE"
    ? "SIE"
    : (requestedProfile === "FIXED_MASS_MAP" || requestedProfile === "PIXELATED_FIXED" ? "FIXED_MASS_MAP" : "SIS");
  const savedELimit = component.e_lim ?? symmetricLimitFromPair(component.e_low, component.e_high);
  return {
    id: `lens-mass-${index}`,
    profile,
    path: String(component.path || ""),
    theta_low: lensMassNumber(component.theta_low, 0),
    theta_high: lensMassNumber(component.theta_high, 1),
    center_x: lensMassNumber(component.center_x, 0),
    center_y: lensMassNumber(component.center_y, 0),
    e_lim: Math.abs(lensMassNumber(savedELimit, 0.3)),
    e_sigma: Math.abs(lensMassNumber(component.e_sigma, 0.15)),
    position_method: component.position_method === "gaussian" ? "gaussian" : "brightest",
  };
}

function serializedLensPlaneMassComponents() {
  return lensPlaneMassState.components.map((component) => ({
    id: component.id,
    profile: component.profile,
    path: component.path,
    theta_low: component.theta_low,
    theta_high: component.theta_high,
    center_x: component.center_x,
    center_y: component.center_y,
    e_lim: component.e_lim,
    e_sigma: component.e_sigma,
    position_method: component.position_method,
  }));
}

function syncLensPlaneMassComponentsInput() {
  if (!lensPlaneMassComponentsInput) return;
  lensPlaneMassComponentsInput.value = JSON.stringify(serializedLensPlaneMassComponents());
}

function renderLensPlaneMassComponents() {
  if (!lensExtraMassList) return;
  if (!lensPlaneMassState.components.length) {
    lensExtraMassList.innerHTML = '<span class="empty-list">No additional mass profiles.</span>';
    syncLensPlaneMassComponentsInput();
    return;
  }
  lensExtraMassList.innerHTML = lensPlaneMassState.components
    .map((component, index) => {
      const isSIE = component.profile === "SIE";
      const isFixedMap = component.profile === "FIXED_MASS_MAP";
      const isActive = lensPlaneMassState.activePickerId === component.id;
      const eFields = isSIE
        ? `
          <label title="Symmetric SIE ellipticity bound">
            e lim
            <input type="number" data-mass-field="e_lim" value="${component.e_lim}" min="0" step="0.01">
          </label>
          <label title="SIE ellipticity Gaussian sigma">
            sigma e
            <input type="number" data-mass-field="e_sigma" value="${component.e_sigma}" min="0.001" step="0.01">
          </label>`
        : "";
      const pathValue = String(component.path || "")
        .replaceAll("&", "&amp;")
        .replaceAll('"', "&quot;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
      const componentFields = isFixedMap
        ? `<div class="lens-extra-mass-fields fixed-mass-map-fields">
            <label title="FITS with POTENTIAL, ALPHA_X, ALPHA_Y, HESS_XX, HESS_YY, HESS_XY extensions">
              Fixed-field FITS path
              <input type="text" data-mass-field="path" value="${pathValue}" placeholder="/absolute/path/fixed_mass_field.fits" spellcheck="false">
            </label>
          </div>`
        : `<div class="lens-extra-mass-fields">
            <label title="Einstein radius lower bound">
              theta min
              <input type="number" data-mass-field="theta_low" value="${component.theta_low}" step="0.01">
            </label>
            <label title="Einstein radius upper bound">
              theta max
              <input type="number" data-mass-field="theta_high" value="${component.theta_high}" step="0.01">
            </label>
            <label title="Fixed lens-plane center x in arcsec">
              x [arcsec]
              <input type="number" data-mass-field="center_x" value="${component.center_x}" step="0.001">
            </label>
            <label title="Fixed lens-plane center y in arcsec">
              y [arcsec]
              <input type="number" data-mass-field="center_y" value="${component.center_y}" step="0.001">
            </label>
            ${eFields}
          </div>`;
      const positionFields = isFixedMap
        ? `<div class="lens-extra-mass-fixed-note">Fixed potential, deflection and Hessian; no sampled amplitude or centroid.</div>`
        : `<div class="lens-extra-mass-position">
            <select data-mass-field="position_method" aria-label="Position measurement method">
              <option value="brightest"${component.position_method === "brightest" ? " selected" : ""}>Brightest 3x3</option>
              <option value="gaussian"${component.position_method === "gaussian" ? " selected" : ""}>Gaussian fit</option>
            </select>
            <button type="button" class="secondary-button small lens-extra-mass-pick${isActive ? " active" : ""}" data-mass-action="pick"${lensPlaneMassState.locating ? " disabled" : ""}>${isActive ? "Cancel" : "Pick"}</button>
          </div>`;
      return `
        <div class="lens-extra-mass-row" data-mass-component-id="${component.id}">
          <div class="lens-extra-mass-row-toolbar">
            <select data-mass-field="profile" aria-label="Mass profile ${index + 1}">
              <option value="SIS"${!isSIE && !isFixedMap ? " selected" : ""}>SIS</option>
              <option value="SIE"${isSIE ? " selected" : ""}>SIE</option>
              <option value="FIXED_MASS_MAP"${isFixedMap ? " selected" : ""}>Fixed mass map</option>
            </select>
            <span class="lens-extra-mass-name">lens plane ${index + 1}</span>
            <button type="button" class="secondary-button lens-extra-mass-remove" data-mass-action="remove" title="Remove mass profile" aria-label="Remove mass profile">&times;</button>
          </div>
          ${componentFields}
          ${positionFields}
        </div>`;
    })
    .join("");
  syncLensPlaneMassComponentsInput();
}

function setLensPlaneMassComponents(value, options = {}) {
  const parsed = parseLensPlaneMassComponents(value).slice(0, 12);
  lensPlaneMassState.components = parsed.map((component, index) => normalizeLensPlaneMassComponent(component, index + 1));
  lensPlaneMassState.nextId = lensPlaneMassState.components.length + 1;
  lensPlaneMassState.activePickerId = "";
  lensPlaneMassState.locating = false;
  renderLensPlaneMassComponents();
  if (options.markStale !== false) markLensScriptStale("Lens-plane mass components changed");
  if (options.save !== false) scheduleProjectSave();
  requestAnimationFrame(syncLensCodePreviewHeight);
}

function addLensPlaneMassComponent() {
  const selectedProfile = String(lensExtraMassProfile?.value || "SIS").toUpperCase();
  const profile = selectedProfile === "SIE"
    ? "SIE"
    : (selectedProfile === "FIXED_MASS_MAP" ? "FIXED_MASS_MAP" : "SIS");
  const component = normalizeLensPlaneMassComponent({ profile }, lensPlaneMassState.nextId);
  lensPlaneMassState.nextId += 1;
  lensPlaneMassState.components.push(component);
  renderLensPlaneMassComponents();
  markLensScriptStale("Lens-plane mass component added");
  scheduleProjectSave();
  requestAnimationFrame(syncLensCodePreviewHeight);
}

function lensPlaneMassComponentFromElement(element) {
  const row = element?.closest?.("[data-mass-component-id]");
  if (!row) return null;
  return lensPlaneMassState.components.find((component) => component.id === row.dataset.massComponentId) || null;
}

function updateLensPlaneMassComponentFromControl(control) {
  const component = lensPlaneMassComponentFromElement(control);
  const field = control?.dataset?.massField;
  if (!component || !field) return;
  if (field === "profile") {
    component.profile = control.value === "SIE"
      ? "SIE"
      : (control.value === "FIXED_MASS_MAP" ? "FIXED_MASS_MAP" : "SIS");
    renderLensPlaneMassComponents();
  } else if (field === "path") {
    component.path = control.value;
  } else if (field === "position_method") {
    component.position_method = control.value === "gaussian" ? "gaussian" : "brightest";
  } else {
    component[field] = lensMassNumber(control.value, component[field]);
    if (field === "e_lim" || field === "e_sigma") component[field] = Math.abs(component[field]);
  }
  syncLensPlaneMassComponentsInput();
  markLensScriptStale("Lens-plane mass component changed");
  scheduleProjectSave();
}

async function toggleLensPlaneMassPositionPicker(componentId) {
  if (lensPlaneMassState.activePickerId === componentId) {
    lensPlaneMassState.activePickerId = "";
    renderLensPlaneMassComponents();
    renderLensScriptCutoutPreview();
    return;
  }
  if (!lensScriptCutoutState.image) {
    await loadLensScriptCutoutPreview({ silent: true });
  }
  if (!lensScriptCutoutState.image?.data?.length) {
    appendLog("Lens-plane position: no project cutout is available.");
    return;
  }
  lensPlaneMassState.activePickerId = componentId;
  renderLensPlaneMassComponents();
  renderLensScriptCutoutPreview();
}

async function locateLensPlaneMassPosition(event) {
  const component = lensPlaneMassState.components.find(
    (entry) => entry.id === lensPlaneMassState.activePickerId,
  );
  const image = lensScriptCutoutState.image;
  if (!component || !image?.path || lensPlaneMassState.locating) return;
  const rect = lensScriptCutoutOverlay.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const sourceHeight = Number(image.source_shape?.[0] || image.shape?.[0] || lensScriptCutoutOverlay.height);
  const sourceWidth = Number(image.source_shape?.[1] || image.shape?.[1] || lensScriptCutoutOverlay.width);
  const x = Math.min(sourceWidth, Math.max(0, ((event.clientX - rect.left) / rect.width) * sourceWidth));
  const y = Math.min(sourceHeight, Math.max(0, ((event.clientY - rect.top) / rect.height) * sourceHeight));
  lensPlaneMassState.locating = true;
  renderLensPlaneMassComponents();
  try {
    const result = await callBackend("/api/mask/conjugate-point", {
      path: image.path,
      x,
      y,
      method: component.position_method,
      size: component.position_method === "gaussian" ? 9 : 4,
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    component.center_x = Number(result.center_arcsec.x);
    component.center_y = Number(result.center_arcsec.y);
    component.position_method = result.method === "gaussian" ? "gaussian" : "brightest";
    appendLog(
      `${component.profile} lens-plane center (${component.position_method}): x=${component.center_x.toFixed(4)}, y=${component.center_y.toFixed(4)} arcsec.`,
    );
    markLensScriptStale("Lens-plane mass position changed");
    scheduleProjectSave();
  } catch (error) {
    appendLog(`Lens-plane position: ${error.message}`);
  } finally {
    lensPlaneMassState.locating = false;
    lensPlaneMassState.activePickerId = "";
    renderLensPlaneMassComponents();
    renderLensScriptCutoutPreview();
  }
}

function syncLensCodePreviewHeight() {
  if (!lensPriorPanel || !lensCodePanel) return;
  if (!window.matchMedia("(min-width: 1021px)").matches) {
    lensCodePanel.style.removeProperty("--lens-code-height");
    return;
  }
  const priorHeight = Math.ceil(lensPriorPanel.getBoundingClientRect().height);
  if (priorHeight > 0) {
    lensCodePanel.style.setProperty("--lens-code-height", `${Math.max(640, priorHeight)}px`);
  }
}

function setupLensCodeHeightSync() {
  if (!lensPriorPanel || !lensCodePanel) return;
  if (typeof ResizeObserver === "function") {
    lensCodeHeightObserver = new ResizeObserver(() => syncLensCodePreviewHeight());
    lensCodeHeightObserver.observe(lensPriorPanel);
  }
  lensPriorPanel.querySelectorAll("details").forEach((details) => {
    details.addEventListener("toggle", () => requestAnimationFrame(syncLensCodePreviewHeight));
  });
  window.addEventListener("resize", syncLensCodePreviewHeight);
  requestAnimationFrame(syncLensCodePreviewHeight);
}

function syncDsplControls() {
  const enabled = Boolean(dsplEnabled?.checked);
  dsplPriorBlocks.forEach((block) => {
    block.classList.toggle("disabled-block", !enabled);
    block.querySelectorAll("input, select").forEach((control) => {
      control.disabled = !enabled;
    });
  });
}

function syncSourceNonlinearPriorControls() {
  if (!sourcePositiveInput || !sourceNonlinearBrightnessInput) return;
  const logBrightness = Boolean(sourceLogBrightnessInput?.checked);
  if (logBrightness) {
    sourcePositiveInput.checked = true;
    sourcePositiveInput.disabled = true;
    sourceNonlinearBrightnessInput.checked = false;
    sourceNonlinearBrightnessInput.disabled = true;
    sourceNonlinearBrightnessInput.title = "Log Matérn uses exp(mu + g), so the legacy nonlinear transform is disabled.";
    return;
  }
  sourcePositiveInput.disabled = false;
  if (!sourcePositiveInput.checked) {
    sourceNonlinearBrightnessInput.checked = false;
    sourceNonlinearBrightnessInput.disabled = true;
    sourceNonlinearBrightnessInput.title = "Requires Source positivity because fractional powers of negative pixels are undefined.";
    return;
  }
  sourceNonlinearBrightnessInput.disabled = false;
  sourceNonlinearBrightnessInput.title = "Apply the learned 10-term nonlinear brightness transform to the positive Matern source.";
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
  if (projectState.id) formData.append("project_id", projectState.id);
  if (projectState.folder) formData.append("project_folder", projectState.folder);

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

function appendRmsProjectContext(formData) {
  if (projectState.id) formData.append("project_id", projectState.id);
  const projectFolder = currentProjectFolderPath({ preferInput: true }) || projectState.folder;
  if (projectFolder) formData.append("project_folder", projectFolder);
  if (cutoutState.bounds) formData.append("cutout_bounds", JSON.stringify(cutoutState.bounds));
  if (cutoutState.shape) formData.append("cutout_shape", JSON.stringify(cutoutState.shape));
  if (imageView.shape) formData.append("image_shape", JSON.stringify(imageView.shape));
}

function rmsProjectPayload(path) {
  return {
    path,
    project_id: projectState.id,
    project_folder: currentProjectFolderPath({ preferInput: true }) || projectState.folder,
    cutout_bounds: cutoutState.bounds || undefined,
    cutout_shape: cutoutState.shape || undefined,
    image_shape: imageView.shape || undefined,
  };
}

async function uploadRmsForLensModel(file) {
  const formData = new FormData();
  formData.append("rms_file", file);
  appendRmsProjectContext(formData);

  const response = await fetch(endpointUrl("/api/image-preprocess/rms"), {
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

async function loadRmsFromPath(path) {
  return callBackend("/api/image-preprocess/rms", rmsProjectPayload(path));
}

function setLensRmsModeMap() {
  const rmsMapInput = document.querySelector('[data-form="lens-model"] input[name="rms_mode"][value="map"]');
  if (rmsMapInput) rmsMapInput.checked = true;
}

function applyRegisteredRms(data = {}) {
  const rmsPath = data.rms_map_path || "";
  if (!rmsPath) return;
  imageView.rmsPath = rmsPath;
  inputRmsPath.value = rmsPath;
  selectedRmsName.textContent = rmsPath;
  setLensRmsModeMap();
  const shapeText = Array.isArray(data.rms_shape) ? ` (${data.rms_shape.join(" x ")})` : "";
  appendLog(`${data.message || "Image preprocess: RMS registered."}${shapeText}`);
  scheduleProjectSave();
}

async function uploadPsfForPreview(file) {
  const formData = new FormData();
  formData.append("psf_file", file);
  if (projectState.id) formData.append("project_id", projectState.id);
  if (projectState.folder) formData.append("project_folder", projectState.folder);

  const response = await fetch(endpointUrl("/api/psf-fit/input-preview"), {
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
  projectState.folder = data.cwd;
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
        selectedImageName.textContent = entry.path;
        inputImagePath.value = entry.path;
        imageView.sourcePath = entry.path;
        await loadImageFromPath(entry.path);
      }
    });
    folderList.appendChild(item);
  }
}

async function loadImageFromPath(path, options = {}) {
  const activity = options.activity !== false;
  const resetArtifacts = options.resetArtifacts !== false;
  if (activity) {
    setActivity(`Loading image ${PathName(path)}`, true);
  }
  selectedImageName.textContent = path;
  inputImagePath.value = path;
  imageView.sourcePath = path;
  appendLog(`Image preprocess: loading ${path}`);
  try {
    const data = await callBackend("/api/image-preprocess/run", {
      path,
      project_id: projectState.id,
      project_folder: projectState.folder,
      register_artifacts: options.registerArtifacts !== false,
      register_default_cutout: options.registerDefaultCutout !== false,
      prefer_project_original: options.preferProjectOriginal !== false,
    });
    imageView.sourcePath = data.source_path || path;
    setImagePreviewMetadata(data);
    if (!psfSciencePath.value.trim()) {
      psfSciencePath.value = imageView.originalFilePath || imageView.sourcePath;
    }
    if (data.image_payload) {
      setImagePreviewFromPayload(data.image_payload, data.preview_url || "", {
        resetArtifacts,
        autoMtf: options.autoMtf !== false,
      });
    } else {
      await setImagePreview(data.preview_url, { resetArtifacts, autoMtf: options.autoMtf !== false });
    }
    applyRegisteredImageArtifacts(data);
    if (imageInputGroup && imageCutoutGroup) {
      imageInputGroup.open = false;
      imageCutoutGroup.open = true;
    }
    scheduleCurrentMtfThumbnailSave();
    scheduleCurrentMtfCutoutPreviewSave();
    appendLog(data.message || "Image preprocess: backend preview loaded.");
    if (activity) {
      setActivity(`Loaded image ${PathName(path)}`, false);
    }
    scheduleProjectSave();
  } catch (error) {
    if (activity) {
      setActivity(`Image load failed: ${error.message}`, false);
    }
    throw error;
  }
}

function updatePreview(pageId, data) {
  const preview = document.querySelector(`[data-preview="${pageId}"]`);
  if (!preview || !data) return;
  if (pageId === "lens-model") {
    data = normalizeLensPreviewData(data);
    lensState.previewData = data;
    renderLensMassSummary(data.mass_parameters);
    const chainPreviews = Array.isArray(data.chain_previews)
      ? data.chain_previews.filter((item) => item && item.preview_url)
      : [];
    lensState.chainPreviews = chainPreviews;
    if (chainPreviews.length) {
      const selected = chainPreviews.find((item) => item.selected) || chainPreviews[0];
      if (!chainPreviews.some((item) => String(item.chain) === String(lensState.selectedChain))) {
        lensState.selectedChain = String(selected.chain);
      }
      renderLensChainSelector();
      renderSelectedLensModelPreview();
      return;
    }
    lensState.selectedChain = "";
    renderLensChainSelector();
  }

  const imageUrl = data.preview_url || data.image_url || data.figure_url;
  if (imageUrl) {
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = pageId === "lens-model" ? cacheBustUrl(imageUrl) : imageUrl;
    img.alt = `${pageCopy[pageId].title} preview`;
    preview.appendChild(img);
    return;
  }

  if (data.message) {
    preview.textContent = data.message;
  }
}

function renderLensChainSelector() {
  if (!lensChainSelect || !lensChainSelectLabel) return;
  const previews = lensState.chainPreviews || [];
  lensChainSelect.innerHTML = "";
  if (previews.length <= 1) {
    lensChainSelectLabel.hidden = true;
    return;
  }
  previews.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(item.chain ?? index + 1);
    const loss = Number.isFinite(Number(item.final_pixelated_loss))
      ? ` loss ${Number(item.final_pixelated_loss).toPrecision(4)}`
      : "";
    option.textContent = `${item.label || `Chain ${item.chain ?? index + 1}`}${loss}`;
    lensChainSelect.append(option);
  });
  lensChainSelect.value = String(lensState.selectedChain || previews[0].chain);
  lensChainSelectLabel.hidden = false;
}

function selectedLensChainPreview() {
  const previews = lensState.chainPreviews || [];
  return previews.find((item) => String(item.chain) === String(lensState.selectedChain)) || previews[0] || null;
}

function currentLensMassParameters() {
  const selected = selectedLensChainPreview();
  return selected?.mass_parameters || lensState.previewData?.mass_parameters || null;
}

function renderCurrentLensMassSummary() {
  renderLensMassSummary(currentLensMassParameters());
}

function renderSelectedLensModelPreview() {
  const preview = document.querySelector(`[data-preview="lens-model"]`);
  if (!preview) return;
  const selected = selectedLensChainPreview();
  const imageUrl = selected?.preview_url || lensState.previewData?.preview_url || "";
  renderCurrentLensMassSummary();
  if (!imageUrl) {
    preview.innerHTML = '<span class="empty-list">No model result loaded for this project.</span>';
    return;
  }
  preview.innerHTML = "";
  const img = document.createElement("img");
  img.src = cacheBustUrl(imageUrl);
  img.alt = selected ? `Lens model chain ${selected.chain}` : "Lens model preview";
  img.onerror = () => {
    const fallbackUrl = lensState.previewData?.preview_url;
    if (fallbackUrl && fallbackUrl !== imageUrl) {
      img.onerror = null;
      img.src = cacheBustUrl(fallbackUrl);
      return;
    }
    preview.innerHTML = '<span class="empty-list">No model result loaded for this project.</span>';
  };
  preview.appendChild(img);
}

function lensPanelDataKey(selected, panels) {
  return [
    selected?.chain ?? "",
    panels?.data?.path || "",
    panels?.model?.path || "",
    panels?.data_minus_lens_light?.path || "",
  ].join("|");
}

function applyLensPanelAutoMtf(selected, panels, width, height) {
  const key = lensPanelDataKey(selected, panels);
  if (!key || lensState.panelAutoMtfKey === key) return;
  const dataPanel = panels?.data;
  if (!dataPanel?.data?.length) return;
  const autoMidtones = autoMidtonesFromRows(dataPanel.data, width, height, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET);
  if (Number.isFinite(autoMidtones)) {
    applySharedAutoMidtones(autoMidtones, { resetBounds: true });
    lensState.panelAutoMtfKey = key;
  }
}

const TWILIGHT_COLOR_STOPS = [
  [0.000000, 226, 217, 226],
  [0.015625, 222, 217, 225],
  [0.031250, 215, 215, 221],
  [0.046875, 206, 211, 217],
  [0.062500, 196, 206, 212],
  [0.078125, 184, 201, 208],
  [0.093750, 172, 194, 204],
  [0.109375, 160, 188, 201],
  [0.125000, 149, 181, 199],
  [0.140625, 138, 174, 197],
  [0.156250, 129, 166, 195],
  [0.171875, 121, 159, 194],
  [0.187500, 114, 151, 193],
  [0.203125, 108, 143, 191],
  [0.218750, 104, 135, 190],
  [0.234375, 100, 126, 188],
  [0.250000, 98, 118, 186],
  [0.265625, 96, 109, 184],
  [0.281250, 95, 100, 181],
  [0.296875, 95, 90, 177],
  [0.312500, 94, 81, 173],
  [0.328125, 94, 71, 167],
  [0.343750, 93, 61, 161],
  [0.359375, 92, 52, 153],
  [0.375000, 89, 42, 143],
  [0.390625, 86, 34, 132],
  [0.406250, 81, 27, 119],
  [0.421875, 76, 22, 105],
  [0.437500, 69, 19, 92],
  [0.453125, 62, 17, 80],
  [0.468750, 56, 17, 69],
  [0.484375, 51, 17, 61],
  [0.500000, 47, 20, 54],
  [0.515625, 52, 18, 56],
  [0.531250, 58, 17, 58],
  [0.546875, 65, 18, 61],
  [0.562500, 74, 19, 66],
  [0.578125, 84, 21, 70],
  [0.593750, 95, 23, 74],
  [0.609375, 105, 26, 77],
  [0.625000, 116, 30, 79],
  [0.640625, 126, 34, 80],
  [0.656250, 135, 39, 80],
  [0.671875, 144, 46, 80],
  [0.687500, 152, 53, 80],
  [0.703125, 160, 61, 80],
  [0.718750, 166, 69, 80],
  [0.734375, 172, 77, 81],
  [0.750000, 178, 86, 82],
  [0.765625, 183, 95, 85],
  [0.781250, 187, 105, 88],
  [0.796875, 191, 114, 93],
  [0.812500, 194, 124, 99],
  [0.828125, 197, 134, 106],
  [0.843750, 200, 144, 115],
  [0.859375, 202, 154, 125],
  [0.875000, 204, 163, 137],
  [0.890625, 207, 173, 150],
  [0.906250, 209, 182, 163],
  [0.921875, 213, 191, 177],
  [0.937500, 216, 199, 190],
  [0.953125, 220, 206, 203],
  [0.968750, 223, 212, 214],
  [0.984375, 225, 216, 221],
  [1.000000, 226, 217, 226],
];

function interpolateColorStops(stops, value) {
  const t = Math.min(1, Math.max(0, Number(value) || 0));
  for (let i = 1; i < stops.length; i += 1) {
    const left = stops[i - 1];
    const right = stops[i];
    if (t <= right[0]) {
      const local = (t - left[0]) / Math.max(right[0] - left[0], 1e-6);
      return [
        Math.round(left[1] + local * (right[1] - left[1])),
        Math.round(left[2] + local * (right[2] - left[2])),
        Math.round(left[3] + local * (right[3] - left[3])),
      ];
    }
  }
  const last = stops[stops.length - 1];
  return [last[1], last[2], last[3]];
}

function bwrColor(value) {
  const t = Math.min(1, Math.max(0, Number(value) || 0));
  if (t < 0.5) {
    const local = t / 0.5;
    return [
      Math.round(255 * local),
      Math.round(255 * local),
      255,
    ];
  }
  const local = (t - 0.5) / 0.5;
  return [
    255,
    Math.round(255 * (1 - local)),
    Math.round(255 * (1 - local)),
  ];
}

function colorMapValue(name, value) {
  if (name === "twilight") return interpolateColorStops(TWILIGHT_COLOR_STOPS, value);
  if (name === "bwr") return bwrColor(value);
  const gray = Math.round(255 * Math.min(1, Math.max(0, Number(value) || 0)));
  return [gray, gray, gray];
}

function drawResidualPayloadToCanvas(canvas, image) {
  if (!canvas || !image?.data?.length) return false;
  const height = Number(image.shape?.[0] || image.data.length || 0);
  const width = Number(image.shape?.[1] || image.data[0]?.length || 0);
  if (!width || !height) return false;
  canvas.width = width;
  canvas.height = height;
  const output = new ImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    const row = image.data[y] || [];
    for (let x = 0; x < width; x += 1) {
      const value = Number(row[x]);
      const t = Math.min(1, Math.max(0, (value + 3) / 6));
      const [r, g, b] = colorMapValue("bwr", t);
      const i = (y * width + x) * 4;
      output.data[i] = r;
      output.data[i + 1] = g;
      output.data[i + 2] = b;
      output.data[i + 3] = 255;
    }
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(output, 0, 0);
  canvas.classList.add("visible");
  renderCanvasAxes(canvas, image.pixel_scale_arcsec ?? currentImagePixelScaleArcsec());
  return { width, height };
}

function positiveLogRange(rows, width, height) {
  let min = Infinity;
  let max = -Infinity;
  for (let y = 0; y < height; y += 1) {
    const row = rows[y] || [];
    for (let x = 0; x < width; x += 1) {
      const value = Number(row[x]);
      if (!Number.isFinite(value) || value <= 0) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  return { min, max, logMin: Math.log(min), logMax: Math.log(max) };
}

function drawLogNormPayloadToCanvas(canvas, image) {
  if (!canvas || !image?.data?.length) return false;
  const height = Number(image.shape?.[0] || image.data.length || 0);
  const width = Number(image.shape?.[1] || image.data[0]?.length || 0);
  if (!width || !height) return false;
  const range = positiveLogRange(image.data, width, height);
  if (!range) return drawImagePayloadToCanvas(canvas, image, { colorMap: "twilight" });
  canvas.width = width;
  canvas.height = height;
  const output = new ImageData(width, height);
  const denom = Math.max(range.logMax - range.logMin, 1e-12);
  for (let y = 0; y < height; y += 1) {
    const row = image.data[y] || [];
    for (let x = 0; x < width; x += 1) {
      const value = Number(row[x]);
      const normalized = value > 0 && Number.isFinite(value)
        ? (Math.log(value) - range.logMin) / denom
        : 0;
      const [r, g, b] = colorMapValue("twilight", normalized);
      const i = (y * width + x) * 4;
      output.data[i] = r;
      output.data[i + 1] = g;
      output.data[i + 2] = b;
      output.data[i + 3] = 255;
    }
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(output, 0, 0);
  canvas.classList.add("visible");
  renderCanvasAxes(canvas, image.pixel_scale_arcsec ?? currentImagePixelScaleArcsec());
  return { width, height };
}

function renderLensPanelSection(section, title, panels, selected, imageUrl = "") {
  const dataPanel = panels.data;
  const height = Number(dataPanel?.shape?.[0] || dataPanel?.data?.length || 0);
  const width = Number(dataPanel?.shape?.[1] || dataPanel?.data?.[0]?.length || 0);
  if (!dataPanel?.data?.length || !width || !height) {
    if (!imageUrl) return false;
    const img = document.createElement("img");
    img.src = cacheBustUrl(imageUrl);
    img.alt = title;
    section.append(img);
    return true;
  }

  const sharedRange = imageRowsFiniteRange(dataPanel.data, width, height);
  const panelOrder = [
    ["data", "Data"],
    ["model", "LensModel"],
    ["data_minus_model_over_rms", "Residual"],
    ["source", "Source"],
    ["data_minus_lens_light", "Lens Light subtracted"],
    ["lensed_source_without_lens_light", "Lensed source without lens light"],
    ["lensed_arc1", "Lensed arc 1"],
    ["lensed_arc2", "Lensed arc 2"],
    ["source1", "Source 1"],
    ["source2", "Source 2"],
  ];
  const grid = document.createElement("div");
  grid.className = "lens-model-panel-grid";
  const renderedCanvases = [];
  const isParametric = title.toLowerCase().startsWith("parametric");
  panelOrder.forEach(([key, fallbackLabel]) => {
    const image = panels[key];
    if (!image?.data?.length) return;
    const pane = document.createElement("figure");
    pane.className = "lens-model-panel";
    const canvas = document.createElement("canvas");
    const caption = document.createElement("figcaption");
    caption.textContent = image.label || fallbackLabel;
    if (key === "data_minus_model_over_rms") {
      drawResidualPayloadToCanvas(canvas, image);
    } else if (["source", "source1", "source2"].includes(key)) {
      drawLogNormPayloadToCanvas(canvas, image);
    } else if (isParametric && ["data", "data_minus_lens_light", "model"].includes(key)) {
      drawLogNormPayloadToCanvas(canvas, image);
    } else {
      const useSharedRange = ["data", "model", "data_minus_lens_light", "lensed_source_without_lens_light", "lensed_arc1", "lensed_arc2"].includes(key);
      drawImagePayloadToCanvas(canvas, image, {
        mtf: maskSubtractionMtfParameters(),
        range: useSharedRange ? sharedRange : undefined,
        colorMap: "twilight",
        pixelScaleArcsec: image.pixel_scale_arcsec ?? dataPanel.pixel_scaleArcsec ?? dataPanel.pixel_scale_arcsec ?? currentImagePixelScaleArcsec(),
      });
    }
    renderedCanvases.push({
      canvas,
      pixelScaleArcsec: image.pixel_scale_arcsec ?? dataPanel.pixel_scale_arcsec ?? currentImagePixelScaleArcsec(),
    });
    pane.append(caption, canvas);
    grid.append(pane);
  });
  if (!grid.children.length) return false;
  section.append(grid);
  window.requestAnimationFrame(() => {
    renderedCanvases.forEach(({ canvas, pixelScaleArcsec }) => renderCanvasAxes(canvas, pixelScaleArcsec));
  });
  return true;
}

function renderLensModelComparisonStack(preview, selected) {
  preview.innerHTML = "";
  const stack = document.createElement("div");
  stack.className = "lens-model-comparison-stack";
  const sections = [
    ["Parametric model", selected.parametric_panels || {}, selected.parametric_url || ""],
    ["Pixelated SVI model", selected.pixelated_panels || {}, selected.pixelated_url || ""],
    ["Semilinear source model", selected.semilinear_panels || {}, selected.semilinear_url || ""],
  ];
  let rendered = false;
  sections.forEach(([title, panels, imageUrl]) => {
    if (!panels?.data && !imageUrl) return;
    const section = document.createElement("section");
    section.className = "lens-model-comparison-section";
    section.dataset.modelStage = title.toLowerCase().startsWith("parametric") ? "parametric" : "pixelated";
    const heading = document.createElement("h4");
    heading.textContent = title;
    section.append(heading);
    if (renderLensPanelSection(section, title, panels, selected, imageUrl)) {
      stack.append(section);
      rendered = true;
    }
  });
  if (!rendered) return false;
  preview.append(stack);
  return true;
}

function cacheBustUrl(url) {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}t=${Date.now()}`;
}

function formatMassValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => formatMassValue(item)).join(", ");
  }
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return String(value ?? "");
  const absValue = Math.abs(numberValue);
  if (absValue !== 0 && (absValue < 1e-3 || absValue >= 1e4)) {
    return numberValue.toExponential(3);
  }
  return numberValue.toFixed(5).replace(/\.?0+$/, "");
}

function renderLensMassSummary(parameters) {
  if (!lensMassSummary) return;
  const entries = Object.entries(parameters || {}).filter(([, value]) => value !== undefined && value !== null);
  if (!entries.length) {
    lensMassSummary.hidden = true;
    lensMassSummary.innerHTML = "";
    return;
  }
  lensMassSummary.hidden = false;
  lensMassSummary.innerHTML = "";
  const title = document.createElement("h4");
  title.textContent = "Mass parameter median";
  lensMassSummary.appendChild(title);
  const grid = document.createElement("div");
  grid.className = "mass-summary-grid";
  entries.forEach(([key, value]) => {
    const item = document.createElement("div");
    item.className = "mass-summary-item";
    const label = document.createElement("span");
    label.className = "mass-summary-key";
    label.textContent = key;
    const number = document.createElement("span");
    number.className = "mass-summary-value";
    number.textContent = formatMassValue(value);
    item.append(label, number);
    grid.appendChild(item);
  });
  lensMassSummary.appendChild(grid);
}

function drawRowsContour(ctx, rows, color, width, height) {
  if (!rows || !width || !height) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.95;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  const valueAt = (x, y) => Number(rows[y]?.[x] || 0) > 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      if (valueAt(x, y) !== valueAt(x + 1, y)) {
        ctx.moveTo(x + 1, y);
        ctx.lineTo(x + 1, y + 1);
      }
    }
  }
  for (let y = 0; y < height - 1; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (valueAt(x, y) !== valueAt(x, y + 1)) {
        ctx.moveTo(x, y + 1);
        ctx.lineTo(x + 1, y + 1);
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}

function currentImagePixelScaleArcsec() {
  const scale = Number(imageView.pixelScaleArcsec);
  return Number.isFinite(scale) && scale > 0 ? scale : null;
}

function currentSourcePixelScaleArcsec() {
  const scale = currentImagePixelScaleArcsec();
  return scale ? scale / Math.max(1, Number(imageView.displayStride || 1)) : null;
}

function imageRowsFiniteRange(rows, width, height) {
  let min = Infinity;
  let max = -Infinity;
  for (let y = 0; y < height; y += 1) {
    const row = rows[y] || [];
    for (let x = 0; x < width; x += 1) {
      const value = Number(row[x]);
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return min === Infinity || max === -Infinity ? null : { min, max };
}

function imageRowsFiniteMedian(rows, width, height) {
  const values = [];
  const stride = Math.max(1, Math.floor(Math.sqrt((width * height) / 250000)));
  for (let y = 0; y < height; y += stride) {
    const row = rows[y] || [];
    for (let x = 0; x < width; x += stride) {
      const value = Number(row[x]);
      if (Number.isFinite(value)) values.push(value);
    }
  }
  if (!values.length) return null;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

function imageDataFiniteRange(imageData) {
  if (!imageData) return null;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < imageData.data.length; i += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const value = imageData.data[i + channel];
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return min === Infinity || max === -Infinity ? null : { min, max };
}

function normalizeDisplayRange(range) {
  if (!range) return null;
  const min = Number(range.min);
  const max = Number(range.max);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  return { min, max };
}

function currentImageMtfDisplayRange() {
  if (imageView.rawPayload?.data?.length) {
    const height = Number(imageView.rawPayload.shape?.[0] || imageView.rawPayload.data.length || 0);
    const width = Number(imageView.rawPayload.shape?.[1] || imageView.rawPayload.data[0]?.length || 0);
    return normalizeDisplayRange(imageRowsFiniteRange(imageView.rawPayload.data, width, height));
  }
  return normalizeDisplayRange(imageDataFiniteRange(imageView.original));
}

function subtractedImageDisplaySettings(subtractedImage, baseMtfValues = mtfParameters()) {
  const height = Number(subtractedImage?.shape?.[0] || subtractedImage?.data?.length || 0);
  const width = Number(subtractedImage?.shape?.[1] || subtractedImage?.data?.[0]?.length || 0);
  const subtractedRange = subtractedImage?.data ? normalizeDisplayRange(imageRowsFiniteRange(subtractedImage.data, width, height)) : null;
  const baseMtf = normalizeMtfValues(baseMtfValues);
  return {
    mtf: {
      shadows: 0,
      midtones: baseMtf.midtones,
      highlights: 1,
    },
    range: subtractedRange,
  };
}

function imageDataFiniteMedian(imageData) {
  if (!imageData) return null;
  const values = [];
  const stride = Math.max(1, Math.floor(Math.sqrt((imageData.width * imageData.height) / 250000)));
  for (let y = 0; y < imageData.height; y += stride) {
    for (let x = 0; x < imageData.width; x += stride) {
      const i = 4 * (y * imageData.width + x);
      values.push(imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]);
    }
  }
  if (!values.length) return null;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

function normalizePixelValue(value, range) {
  if (!Number.isFinite(value) || !range) return 0;
  const width = range.max - range.min;
  if (Math.abs(width) < 1e-12) return 0.5;
  return (value - range.min) / width;
}

function normalizeLogPixelValue(value, vmin = 1e-5, vmax = 1) {
  const lower = Number(vmin);
  const upper = Number(vmax);
  if (!Number.isFinite(value) || !Number.isFinite(lower) || !Number.isFinite(upper) || lower <= 0 || upper <= lower) return 0;
  const clipped = Math.min(upper, Math.max(lower, value));
  return (Math.log(clipped) - Math.log(lower)) / (Math.log(upper) - Math.log(lower));
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function solveMidtonesForMedian(normalizedMedian, shadows, highlights, target = MTF_DATA_DISPLAY_TARGET) {
  const width = Math.max(highlights - shadows, 1e-6);
  const x = clamp01((normalizedMedian - shadows) / width);
  if (x <= 0) return 0.001;
  if (x >= 1) return 0.999;
  const displayTarget = clamp01(target);
  const denominator = displayTarget + x - 2 * displayTarget * x;
  if (Math.abs(denominator) < 1e-12) return 0.5;
  return Math.min(0.999, Math.max(0.001, (x * (1 - displayTarget)) / denominator));
}

function sliderDecimalPlaces(slider, fallback = 3) {
  const step = String(slider?.step || "");
  if (!step || step === "any") return fallback;
  const decimal = step.includes(".") ? step.split(".")[1].replace(/0+$/, "").length : 0;
  return Math.max(fallback, decimal);
}

function formatSliderNumber(slider, value, fallback = 3) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toFixed(sliderDecimalPlaces(slider, fallback)) : "";
}

function setMidtonesSliderAuto(slider, autoMidtones) {
  if (!slider || !Number.isFinite(autoMidtones)) return;
  const max = Math.max(0.001, 2 * autoMidtones);
  slider.min = "0";
  slider.max = formatSliderNumber(slider, max);
  slider.value = formatSliderNumber(slider, autoMidtones);
}

function ensureMidtonesSliderCanRepresent(slider, value) {
  if (!slider || !Number.isFinite(Number(value))) return;
  slider.min = "0";
  const currentMax = Number(slider.max);
  const neededMax = Math.max(0.001, Number(value));
  if (!Number.isFinite(currentMax) || currentMax < neededMax) {
    slider.max = neededMax.toFixed(3);
  }
}

function autoMidtonesFromRangeAndMedian(range, median, mtfValues, target = MTF_DATA_DISPLAY_TARGET) {
  if (!range || !Number.isFinite(median)) return null;
  const normalizedMedian = normalizePixelValue(median, range);
  return solveMidtonesForMedian(normalizedMedian, mtfValues.shadows, mtfValues.highlights, target);
}

function autoMidtonesFromRows(rows, width, height, mtfValues, target = MTF_DATA_DISPLAY_TARGET) {
  return autoMidtonesFromRangeAndMedian(
    imageRowsFiniteRange(rows, width, height),
    imageRowsFiniteMedian(rows, width, height),
    mtfValues,
    target,
  );
}

function autoMidtonesFromImageData(imageData, mtfValues, target = MTF_DATA_DISPLAY_TARGET) {
  return autoMidtonesFromRangeAndMedian(
    imageDataFiniteRange(imageData),
    imageDataFiniteMedian(imageData),
    mtfValues,
    target,
  );
}

function applySharedAutoMidtones(autoMidtones, options = {}) {
  if (!Number.isFinite(autoMidtones)) return;
  if (options.resetBounds) {
    mtfShadows.value = String(DEFAULT_MTF_VALUES.shadows);
    mtfHighlights.value = String(DEFAULT_MTF_VALUES.highlights);
    euclidMtfShadows.value = String(DEFAULT_MTF_VALUES.shadows);
    euclidMtfHighlights.value = String(DEFAULT_MTF_VALUES.highlights);
  }
  setMidtonesSliderAuto(mtfMidtones, autoMidtones);
  setMidtonesSliderAuto(euclidMtfMidtones, autoMidtones);
  updateMtfLabels();
  updateEuclidMtfLabels();
}

function applyMaskSubtractionAutoMidtones(autoMidtones) {
  if (!Number.isFinite(autoMidtones)) return;
  maskSubtractionMtfShadows.value = String(DEFAULT_MTF_VALUES.shadows);
  maskSubtractionMtfHighlights.value = String(DEFAULT_MTF_VALUES.highlights);
  setMidtonesSliderAuto(maskSubtractionMtfMidtones, autoMidtones);
  updateMaskSubtractionMtfLabels();
}

function applyMaskCutoutAutoMidtones(autoMidtones) {
  if (!Number.isFinite(autoMidtones)) return;
  maskCutoutMtfShadows.value = String(DEFAULT_MTF_VALUES.shadows);
  maskCutoutMtfHighlights.value = String(DEFAULT_MTF_VALUES.highlights);
  setMidtonesSliderAuto(maskCutoutMtfMidtones, autoMidtones);
  updateMaskCutoutMtfLabels();
}

function autoMidtonesFromImagePayload(image, target = MTF_DATA_DISPLAY_TARGET) {
  const height = Number(image?.shape?.[0] || image?.data?.length || 0);
  const width = Number(image?.shape?.[1] || image?.data?.[0]?.length || 0);
  if (!image?.data?.length || !width || !height) return null;
  return autoMidtonesFromRows(image.data, width, height, DEFAULT_MTF_VALUES, target);
}

function drawImagePayloadToCanvas(canvas, image, options = {}) {
  if (!canvas || !image?.data?.length) return false;
  const height = Number(image.shape?.[0] || image.data.length || 0);
  const width = Number(image.shape?.[1] || image.data[0]?.length || 0);
  if (!width || !height) return false;
  if (options.currentImagePreview === true && drawCurrentImagePreviewRegion(canvas, { x0: 0, y0: 0, width, height }, width, height)) {
    return { width, height };
  }
  if (options.projectThumbnail === true && drawProjectThumbnailRegion(canvas, { x0: 0, y0: 0, width, height }, width, height, renderLensScriptCutoutPreview)) {
    return { width, height };
  }
  canvas.width = width;
  canvas.height = height;

  const output = new ImageData(width, height);
  const { shadows, midtones, highlights } = options.mtf || mtfParameters();
  const range = options.range || imageRowsFiniteRange(image.data, width, height);
  const useLogNorm = options.norm === "log";
  const logVmin = options.vmin ?? 1e-5;
  const logVmax = options.vmax ?? 1;
  for (let y = 0; y < height; y += 1) {
    const row = image.data[y] || [];
    for (let x = 0; x < width; x += 1) {
      const normalized = useLogNorm
        ? normalizeLogPixelValue(Number(row[x]), logVmin, logVmax)
        : normalizePixelValue(Number(row[x]), range);
      const stretched = useLogNorm ? clamp01(normalized) : histogramTransformValue(normalized, shadows, midtones, highlights);
      const [r, g, b] = colorMapValue(options.colorMap, stretched);
      const i = (y * width + x) * 4;
      output.data[i] = r;
      output.data[i + 1] = g;
      output.data[i + 2] = b;
      output.data[i + 3] = 255;
    }
  }

  const imageCtx = canvas.getContext("2d");
  imageCtx.putImageData(output, 0, 0);
  canvas.classList.add("visible");
  renderCanvasAxes(canvas, options.pixelScaleArcsec ?? currentImagePixelScaleArcsec());
  return { width, height };
}

function renderLensScriptCutoutOverlay(width, height, label) {
  lensScriptCutoutOverlay.width = width;
  lensScriptCutoutOverlay.height = height;
  lensScriptCutoutOverlay.classList.add("visible", "overlay");
  lensScriptCutoutOverlay.classList.toggle(
    "mass-position-picker",
    Boolean(lensPlaneMassState.activePickerId),
  );
  const overlayCtx = lensScriptCutoutOverlay.getContext("2d");
  overlayCtx.clearRect(0, 0, width, height);
  drawRowsContour(overlayCtx, lensScriptCutoutState.masks.mask_1, "#9933ff", width, height);
  drawRowsContour(overlayCtx, lensScriptCutoutState.masks.mask_2, "#ff3300", width, height);
  drawRowsContour(overlayCtx, lensScriptCutoutState.masks.mask_out, "#ff4f5f", width, height);
  drawLensScriptConjugatePoints(overlayCtx, lensScriptCutoutState.conjugatePointsSource1, "black", width, height);
  drawLensScriptConjugatePoints(overlayCtx, lensScriptCutoutState.conjugatePointsSource2, "#ff7f0e", width, height);
  drawLensPlaneMassPositions(overlayCtx, width, height);
  lensScriptCutoutEmpty.style.display = "none";
  lensScriptCutoutStatus.textContent = `${label} ${width}x${height}`;
}

function lensScriptConjugateArcsec(point) {
  if (Array.isArray(point) && point.length >= 2) {
    return { x: Number(point[0]), y: Number(point[1]) };
  }
  const arcsec = point?.arcsec || point?.center_arcsec;
  if (!arcsec) return null;
  return { x: Number(arcsec.x), y: Number(arcsec.y) };
}

function drawLensScriptConjugatePoints(ctx, points, color, width, height) {
  const scale = Number(lensScriptCutoutState.image?.pixel_scale_arcsec ?? currentImagePixelScaleArcsec());
  if (!Array.isArray(points) || !points.length || !Number.isFinite(scale) || scale <= 0) return;
  ctx.save();
  ctx.fillStyle = color;
  points.forEach((point) => {
    const arcsec = lensScriptConjugateArcsec(point);
    if (!arcsec || !Number.isFinite(arcsec.x) || !Number.isFinite(arcsec.y)) return;
    const x = width / 2 + arcsec.x / scale;
    const y = height / 2 - arcsec.y / scale;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < -2 || x > width + 2 || y < -2 || y > height + 2) return;
    ctx.beginPath();
    ctx.arc(x, y, 0.65, 0, 2 * Math.PI);
    ctx.fill();
  });
  ctx.restore();
}

function drawLensPlaneMassPositions(ctx, width, height) {
  const scale = Number(lensScriptCutoutState.image?.pixel_scale_arcsec ?? currentImagePixelScaleArcsec());
  if (!lensPlaneMassState.components.length || !Number.isFinite(scale) || scale <= 0) return;
  ctx.save();
  ctx.lineWidth = 1;
  lensPlaneMassState.components.forEach((component) => {
    if (component.profile === "FIXED_MASS_MAP") return;
    const x = width / 2 + Number(component.center_x) / scale;
    const y = height / 2 - Number(component.center_y) / scale;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < -4 || x > width + 4 || y < -4 || y > height + 4) return;
    ctx.strokeStyle = component.profile === "SIE" ? "#00b7d6" : "#ffd166";
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctx.moveTo(x - 4, y);
    ctx.lineTo(x + 4, y);
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x, y + 4);
    ctx.stroke();
  });
  ctx.restore();
}

function drawLensScriptSavedMtfPreview(src, image) {
  const height = Number(image?.shape?.[0] || 0);
  const width = Number(image?.shape?.[1] || 0);
  if (!src || !width || !height) return null;
  const url = cacheBustUrl(src);
  const token = `${url}:${width}x${height}`;
  lensScriptCutoutState.previewToken = token;
  const img = new Image();
  img.onload = () => {
    if (lensScriptCutoutState.previewToken !== token) return;
    lensScriptCutoutCanvas.width = width;
    lensScriptCutoutCanvas.height = height;
    const ctx = lensScriptCutoutCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    lensScriptCutoutCanvas.classList.add("visible");
    renderCanvasAxes(lensScriptCutoutCanvas, currentImagePixelScaleArcsec());
    renderLensScriptCutoutOverlay(width, height, PathName(image?.path || "Data_cutout_preview.png"));
  };
  img.onerror = () => {
    if (lensScriptCutoutState.previewToken !== token) return;
    lensScriptCutoutState.previewSrc = "";
    renderLensScriptCutoutPreview();
  };
  img.src = url;
  lensScriptCutoutEmpty.style.display = "grid";
  lensScriptCutoutEmpty.textContent = "Loading MTF cutout preview...";
  lensScriptCutoutStatus.textContent = "Loading MTF cutout preview...";
  return { width, height, pending: true };
}

function drawLensScriptMatplotlibPreview(src) {
  if (!src) return null;
  const url = cacheBustUrl(src);
  const token = `matplotlib:${url}`;
  lensScriptCutoutState.previewToken = token;
  const img = new Image();
  img.onload = () => {
    if (lensScriptCutoutState.previewToken !== token) return;
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return;
    lensScriptCutoutCanvas.width = width;
    lensScriptCutoutCanvas.height = height;
    const ctx = lensScriptCutoutCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    lensScriptCutoutCanvas.classList.add("visible");
    lensScriptCutoutOverlay.classList.remove("visible");
    removePreviewAxes(lensScriptCutoutCanvas);
    lensScriptCutoutEmpty.style.display = "none";
    lensScriptCutoutStatus.textContent = "matplotlib cutout overlay";
  };
  img.onerror = () => {
    if (lensScriptCutoutState.previewToken !== token) return;
    lensScriptCutoutState.matplotlibPreviewSrc = "";
    renderLensScriptCutoutPreview();
  };
  img.src = url;
  lensScriptCutoutEmpty.style.display = "grid";
  lensScriptCutoutEmpty.textContent = "Loading matplotlib cutout preview...";
  lensScriptCutoutStatus.textContent = "Loading matplotlib cutout preview...";
  return { width: 1, height: 1, pending: true, matplotlib: true };
}

function drawLensScriptSubtractedMatplotlibPreview(src) {
  if (!src) return null;
  const url = cacheBustUrl(src);
  const token = `subtracted-matplotlib:${url}`;
  lensScriptCutoutState.subtractedPreviewToken = token;
  const img = new Image();
  img.onload = () => {
    if (lensScriptCutoutState.subtractedPreviewToken !== token) return;
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return;
    lensScriptSubtractedCanvas.width = width;
    lensScriptSubtractedCanvas.height = height;
    const ctx = lensScriptSubtractedCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    lensScriptSubtractedCanvas.classList.add("visible");
    removePreviewAxes(lensScriptSubtractedCanvas);
    lensScriptSubtractedEmpty.style.display = "none";
  };
  img.onerror = () => {
    if (lensScriptCutoutState.subtractedPreviewToken !== token) return;
    lensScriptCutoutState.subtractedMatplotlibPreviewSrc = "";
    renderLensScriptCutoutPreview();
  };
  img.src = url;
  lensScriptSubtractedEmpty.style.display = "grid";
  lensScriptSubtractedEmpty.textContent = "Loading matplotlib lens-light-subtracted preview...";
  return { width: 1, height: 1, pending: true, matplotlib: true };
}

function drawLensScriptCurrentMtfCutout() {
  if (!renderCurrentMtfCutoutToCanvas(lensScriptCutoutCanvas)) return null;
  lensScriptCutoutCanvas.classList.add("visible");
  renderCanvasAxes(lensScriptCutoutCanvas, currentImagePixelScaleArcsec());
  return {
    width: lensScriptCutoutCanvas.width,
    height: lensScriptCutoutCanvas.height,
  };
}

function renderLensScriptCutoutPreview() {
  const image = lensScriptCutoutState.image;
  lensScriptCutoutCanvas.classList.remove("visible");
  lensScriptCutoutOverlay.classList.remove("visible");
  lensScriptSubtractedCanvas.classList.remove("visible");

  let currentDims = null;
  if (lensPlaneMassState.activePickerId && image?.data?.length) {
    lensScriptCutoutState.previewToken = `mass-picker:${Date.now()}`;
    currentDims = drawImagePayloadToCanvas(lensScriptCutoutCanvas, image, {
      colorMap: maskState.previewColormap,
      mtf: maskCutoutMtfParameters(),
      pixelScaleArcsec: image.pixel_scale_arcsec,
    });
  } else {
    currentDims = drawLensScriptMatplotlibPreview(lensScriptCutoutState.matplotlibPreviewSrc);
  }
  if (!currentDims) {
    currentDims = drawLensScriptCurrentMtfCutout();
  }
  if (!currentDims) {
    currentDims = drawLensScriptSavedMtfPreview(lensScriptCutoutState.previewSrc, image);
  }
  if (!currentDims) {
    currentDims = drawImagePayloadToCanvas(lensScriptCutoutCanvas, image, {
      colorMap: "twilight",
      norm: "log",
      vmin: 1e-5,
      vmax: 1,
    });
  }
  if (currentDims && !currentDims.pending) {
    renderLensScriptCutoutOverlay(currentDims.width, currentDims.height, PathName(image?.path || "Data_cutout.fits"));
  } else if (!currentDims) {
    removePreviewAxes(lensScriptCutoutCanvas);
    lensScriptCutoutEmpty.style.display = "grid";
    lensScriptCutoutEmpty.textContent = "No project cutout loaded.";
    lensScriptCutoutStatus.textContent = "No project cutout loaded.";
  }

  const subtractedDims = drawLensScriptSubtractedMatplotlibPreview(lensScriptCutoutState.subtractedMatplotlibPreviewSrc);
  if (subtractedDims && !subtractedDims.pending) {
    lensScriptSubtractedEmpty.style.display = "none";
  } else if (!subtractedDims) {
    lensScriptSubtractedEmpty.style.display = "grid";
    removePreviewAxes(lensScriptSubtractedCanvas);
    lensScriptSubtractedEmpty.textContent =
      lensScriptCutoutState.subtractedStaleMessage || "Run lens-light subtraction.";
  }
}

async function loadLensScriptCutoutPreview(options = {}) {
  if (!projectState.folder && !projectState.id) return false;
  try {
    const data = await callBackend("/api/lens-model/cutout-overlay", {
      project_id: projectState.id,
      project_folder: projectState.folder,
      display_mtf: maskCutoutMtfParameters(),
      subtracted_display_mtf: maskSubtractionMtfParameters(),
      mask_type: maskState.type,
      mask: maskState.data ? maskRowsForSave() : null,
      conjugate_points_source1: maskState.conjugatePoints,
      conjugate_points_source2: maskState.conjugatePointsSource2,
    });
    lensScriptCutoutState.image = data.image || null;
    lensScriptCutoutState.previewSrc = data.preview_url || cutoutState.previewUrl || "";
    lensScriptCutoutState.matplotlibPreviewSrc = data.matplotlib_preview_url || "";
    lensScriptCutoutState.subtractedImage = data.subtracted_image || null;
    lensScriptCutoutState.subtractedMatplotlibPreviewSrc = data.subtracted_matplotlib_preview_url || "";
    lensScriptCutoutState.subtractedStaleMessage = data.subtracted_stale_message || "";
    lensScriptCutoutState.masks = data.masks || {};
    lensScriptCutoutState.conjugatePointsSource1 = data.conjugate_points_source1 || [];
    lensScriptCutoutState.conjugatePointsSource2 = data.conjugate_points_source2 || [];
    renderLensScriptCutoutPreview();
    return true;
  } catch (error) {
    lensScriptCutoutState.image = null;
    lensScriptCutoutState.previewSrc = "";
    lensScriptCutoutState.matplotlibPreviewSrc = "";
    lensScriptCutoutState.previewToken = "";
    lensScriptCutoutState.subtractedImage = null;
    lensScriptCutoutState.subtractedMatplotlibPreviewSrc = "";
    lensScriptCutoutState.subtractedPreviewToken = "";
    lensScriptCutoutState.subtractedStaleMessage = "";
    lensScriptCutoutState.masks = {};
    lensScriptCutoutState.conjugatePointsSource1 = [];
    lensScriptCutoutState.conjugatePointsSource2 = [];
    lensScriptCutoutCanvas.classList.remove("visible");
    lensScriptCutoutOverlay.classList.remove("visible");
    lensScriptSubtractedCanvas.classList.remove("visible");
    lensScriptCutoutEmpty.style.display = "grid";
    lensScriptSubtractedEmpty.style.display = "grid";
    lensScriptCutoutStatus.textContent = "No project cutout loaded.";
    if (!options.silent) appendLog(`Lens model cutout preview: ${error.message}`);
    return false;
  }
}

function syncLensScriptConjugatesFromMask() {
  lensScriptCutoutState.conjugatePointsSource1 = maskState.conjugatePoints || [];
  lensScriptCutoutState.conjugatePointsSource2 = maskState.conjugatePointsSource2 || [];
  renderLensScriptCutoutPreview();
  scheduleLensScriptCutoutPreviewRefresh(80);
}

function scheduleLensScriptCutoutPreviewRefresh(delay = 250) {
  if (!projectState.folder && !projectState.id) return;
  if (lensScriptCutoutState.refreshTimer) {
    window.clearTimeout(lensScriptCutoutState.refreshTimer);
  }
  lensScriptCutoutState.refreshTimer = window.setTimeout(() => {
    lensScriptCutoutState.refreshTimer = null;
    void loadLensScriptCutoutPreview({ silent: true });
  }, delay);
}

async function restoreLensModelResultFromProject(options = {}) {
  if (!projectState.id) return false;
  const silent = options.silent !== false;
  try {
    const data = await callBackend(`/api/lens-model/latest?project_id=${encodeURIComponent(projectState.id)}`);
    updatePreview("lens-model", data);
    if (data.job_id) {
      lensState.jobId = data.job_id;
    }
    if (data.progress) {
      setLensProgress(data.progress);
    }
    if (data.generated_code) {
      setLensGeneratedCode(data.generated_code, data.model_config || null);
    }
    if (options.save !== false) {
      scheduleProjectSave();
    }
    if (!silent) {
      appendLog(data.message || "Loaded existing lens model result.");
    }
    return true;
  } catch (error) {
    clearLensModelPreview();
    lensState.jobId = "";
    lensState.progress = null;
    lensProgress.hidden = true;
    if (!silent) {
      appendLog(`Lens model result restore: ${error.message}`);
    }
    return false;
  }
}

function setLensGeneratedCode(code, modelConfig = null, options = {}) {
  lensState.generatedCode = code || "";
  lensState.modelConfig = modelConfig || (lensState.generatedCode ? lensState.modelConfig : null);
  lensState.scriptEdited = Boolean(options.edited);
  lensState.scriptStale = false;
  lensState.scriptStaleReason = "";
  lensGenerateCode.textContent = "Generate code";
  lensGenerateCode.classList.remove("needs-attention");
  lensGeneratedCode.className = "language-python";
  lensGeneratedCode.removeAttribute("data-highlighted");
  lensGeneratedCode.textContent = lensState.generatedCode || "Click Generate code to build the Lens Model script.";
  if (!lensState.scriptEditMode) {
    lensScriptEditor.value = lensState.generatedCode;
  }
  if (window.hljs && lensState.generatedCode) {
    window.hljs.highlightElement(lensGeneratedCode);
  }
  void loadLensScriptCutoutPreview({ silent: true });
}

function clearLensGeneratedCode() {
  lensState.generatedCode = "";
  lensState.modelConfig = null;
  lensState.scriptEditMode = false;
  lensState.scriptEdited = false;
  lensState.scriptStale = false;
  lensState.scriptStaleReason = "";
  lensGenerateCode.textContent = "Generate code";
  lensGenerateCode.classList.remove("needs-attention");
  lensGeneratedCode.className = "language-python";
  lensGeneratedCode.removeAttribute("data-highlighted");
  lensGeneratedCode.textContent = "Click Generate code to build the Lens Model script.";
  lensScriptEditor.value = "";
  lensScriptEditor.hidden = true;
  lensGeneratedCode.closest(".code-preview").hidden = false;
  lensEditCode.textContent = "Edit script";
  appendLog("Lens model script cleared.");
  scheduleProjectSave();
}

function markLensScriptStale(reason = "Lens model settings changed") {
  if (!lensState.generatedCode.trim()) return;
  lensState.scriptStale = true;
  lensState.scriptStaleReason = reason;
  lensGenerateCode.textContent = "Regenerate code";
  lensGenerateCode.classList.add("needs-attention");
  updateWorkflowStatus();
}

function setLensScriptEditMode(editing) {
  lensState.scriptEditMode = Boolean(editing);
  const preview = lensGeneratedCode.closest(".code-preview");
  if (lensState.scriptEditMode) {
    lensScriptEditor.value = lensState.generatedCode || "";
    lensScriptEditor.hidden = false;
    preview.hidden = true;
    lensEditCode.textContent = "Apply script";
    lensScriptEditor.focus();
  } else {
    setLensGeneratedCode(lensScriptEditor.value, lensState.modelConfig, { edited: true });
    lensScriptEditor.hidden = true;
    preview.hidden = false;
    lensEditCode.textContent = "Edit script";
    scheduleProjectSave();
  }
}

function applyLensScriptEditorIfNeeded() {
  if (!lensState.scriptEditMode) return;
  lensState.scriptEditMode = false;
  setLensGeneratedCode(lensScriptEditor.value, lensState.modelConfig, { edited: true });
  lensScriptEditor.hidden = true;
  lensGeneratedCode.closest(".code-preview").hidden = false;
  lensEditCode.textContent = "Edit script";
}

function setLensRunState(running) {
  lensState.jobRunning = Boolean(running);
  lensRunModel.disabled = lensState.jobRunning;
  lensRunGpuModel.disabled = lensState.jobRunning;
  lensRunAllGpuModel.disabled = lensState.jobRunning;
  lensStopRun.disabled = !lensState.jobRunning;
}

async function generateLensCode(options = {}) {
  const silent = Boolean(options.silent);
  const updateActivity = options.activity !== false;
  const saveProject = options.save !== false;
  const form = document.querySelector('[data-form="lens-model"]');
  syncLensLightPriorFromMask(maskState.lensLightJobStatus, { silent: true });
  const payload = collectFormPayload(form);
  if (!silent) appendLog("Lens model: generating code.");
  if (updateActivity) {
    setActivity(silent ? "Preparing initial lens model code" : "Generating lens model code", true);
  }
  lensGenerateCode.disabled = true;
  try {
    const data = await callBackend("/api/lens-model/generate-code", payload);
    setLensGeneratedCode(data.generated_code || "", data.model_config || null);
    syncLensLightPriorFromMask(maskState.lensLightJobStatus, { silent: true });
    if (!silent) appendLog(data.message || "Lens model code generated.");
    if (!silent && data.hmc_script_path) appendLog(`Generated HMC script: ${data.hmc_script_path}`);
    if (updateActivity) {
      setActivity(silent ? "Initial lens model code ready" : "Lens model code generated", false);
    }
    if (saveProject) {
      scheduleProjectSave();
    }
  } catch (error) {
    if (updateActivity) {
      setActivity(`Code generation failed: ${error.message}`, false);
    }
    appendLog(`Lens model code generation: ${error.message}`);
  } finally {
    lensGenerateCode.disabled = false;
  }
}

function setLensProgress(progress) {
  if (!progress) return;
  lensState.progress = progress;
  lensProgress.hidden = false;
  const fraction = Math.min(1, Math.max(0, Number(progress.fraction ?? 0)));
  const percent = Math.round(100 * fraction);
  lensProgressLabel.textContent = progress.message || "Lens model running";
  lensProgressPercent.textContent = `overall ${percent}%`;

  let parametric = progress.stages?.parametric || null;
  let pixelated = progress.stages?.pixelated || null;
  if (!parametric || !pixelated) {
    const local = progress.total ? Number(progress.step || 0) / Math.max(Number(progress.total), 1) : 0;
    if (progress.stage === "pixelated") {
      parametric = { fraction: 1, message: "Parametric SVI completed." };
      pixelated = { fraction: local, message: progress.message || "Pixelated SVI running." };
    } else if (fraction >= 1) {
      parametric = { fraction: 1, message: "Parametric SVI completed." };
      pixelated = { fraction: 1, message: "Pixelated SVI completed." };
    } else {
      parametric = { fraction: progress.stage === "parametric" ? local : 0, message: progress.message || "Parametric SVI running." };
      pixelated = { fraction: 0, message: "Pixelated SVI pending." };
    }
  }

  updateLensStageProgress(parametric, lensParametricProgressLabel, lensParametricProgressPercent, lensParametricProgressBar);
  updateLensStageProgress(pixelated, lensPixelatedProgressLabel, lensPixelatedProgressPercent, lensPixelatedProgressBar);
}

function updateLensStageProgress(stage, labelEl, percentEl, barEl) {
  const fraction = Math.min(1, Math.max(0, Number(stage?.fraction ?? 0)));
  const percent = Math.round(100 * fraction);
  labelEl.textContent = stage?.message || "SVI pending.";
  percentEl.textContent = `${percent}%`;
  barEl.style.width = `${percent}%`;
}

async function pollLensJob(jobId) {
  lensState.jobId = jobId;
  setLensRunState(true);
  for (let i = 0; i < 7200; i += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    try {
      const data = await callBackend(`/api/lens-model/job/${jobId}`);
      const progress = data.progress || {
        fraction: data.state === "completed" ? 1 : 0,
        message: data.message || `Lens job ${data.state}`,
      };
      setLensProgress(progress);
      setActivity(progress.message || `Lens job ${data.state}`, data.state !== "completed" && data.state !== "failed");
      updateLensLogProgress(jobId, progress, data.state);
      if (data.preview_url) {
        updatePreview("lens-model", data);
      }
      if (data.state === "completed" || data.state === "failed" || data.state === "stopped") {
        appendLog(`Lens job ${jobId}: ${data.state}.`);
        setActivity(`Lens job ${jobId}: ${data.state}`, false);
        setLensRunState(false);
        scheduleProjectSave();
        return;
      }
    } catch (error) {
      setActivity(`Lens job failed: ${error.message}`, false);
      appendLog(`Lens job ${jobId}: ${error.message}`);
      setLensRunState(false);
      return;
    }
  }
  appendLog(`Lens job ${jobId}: still running.`);
  setLensRunState(false);
}

async function stopLensJob() {
  const jobId = lensState.jobId;
  if (!jobId) {
    appendLog("Lens model: no active job id to stop.");
    return;
  }
  setActivity(`Stopping lens job ${jobId}`, true);
  lensStopRun.disabled = true;
  try {
    const data = await callBackend("/api/lens-model/stop", { job_id: jobId });
    if (data.progress) {
      setLensProgress(data.progress);
      updateLensLogProgress(jobId, data.progress, data.state || "stopped");
    }
    appendLog(data.message || `Lens job ${jobId}: stop requested.`);
    setActivity(data.message || `Lens job ${jobId}: stopped`, false);
    setLensRunState(false);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`Lens job ${jobId}: stop failed: ${error.message}`);
    setActivity(`Lens stop failed: ${error.message}`, false);
    lensStopRun.disabled = false;
  }
}

function compactLensPreviewForProject(value) {
  if (Array.isArray(value)) return value.map((item) => compactLensPreviewForProject(item));
  if (!value || typeof value !== "object") return value;
  const compacted = {};
  Object.entries(value).forEach(([key, item]) => {
    if (key === "data" && value.panel_key) return;
    compacted[key] = compactLensPreviewForProject(item);
  });
  return compacted;
}

function snapshotProjectState() {
  const lensPreviewData = currentProjectLensPreviewData();
  const lensJobId = lensPreviewData || lensState.jobRunning ? lensState.jobId : "";
  const lensProgressState = lensPreviewData || lensState.jobRunning ? lensState.progress : null;
  return {
    active_page: activePageId(),
    theme: document.body.dataset.theme || "light",
    data_folder: folderPath.value.trim(),
    euclid: {
      ra: Number(euclidRa.value),
      dec: Number(euclidDec.value),
      cutout_size: Number(euclidCutoutSize.value),
      username: euclidUsername.value.trim(),
      image_path: euclidState.path,
      rms_map_path: euclidState.rmsMapPath,
      rms_bundle_path: euclidState.rmsBundlePath,
      image_options: euclidState.imageOptions,
      image: euclidState.data
        ? {
            path: euclidState.path,
            shape: [euclidState.height, euclidState.width],
            data: euclidState.data,
            vmin: euclidState.vmin,
            vmax: euclidState.vmax,
            zeropoint: euclidState.zeropoint,
            zeropoints: euclidState.zeropoints,
            zeropoint_source: euclidState.zeropointSource,
            flux_unit: euclidState.fluxUnit,
            pixel_scale_arcsec: euclidState.pixelScaleArcsec,
            band: euclidState.band,
            source: euclidState.source,
            label: euclidState.imageOptions[euclidState.currentImageKey]?.label || "",
            image_key: euclidState.currentImageKey,
          }
        : null,
      mtf: {
        shadows: Number(euclidMtfShadows.value),
        midtones: Number(euclidMtfMidtones.value),
        highlights: Number(euclidMtfHighlights.value),
      },
      photometry: {
        mag_system: euclidMagSystem.value,
        aperture_radius: Number(euclidApertureRadius.value),
        annulus_width: Number(euclidAnnulusWidth.value),
        preview_scale: Number(euclidPreviewScale.value),
        display_mode: euclidState.displayMode,
        sb_lower: euclidSbLower.value.trim(),
        sb_upper: euclidSbUpper.value.trim(),
        photoz_enabled: euclidState.photozEnabled,
        photoz_bands: Array.from(selectedPhotozBands()),
      },
    },
    image: {
      source_path: imageView.sourcePath,
      input_path: inputImagePath.value.trim(),
      rms_path: imageView.rmsPath,
      rms_input_path: inputRmsPath.value.trim(),
      rms_selected_name: selectedRmsName.textContent,
      preview_src: imageView.previewSrc,
      selected_name: selectedImageName.textContent,
      image_shape: imageView.shape,
      source_shape: imageView.sourceShape,
      display_stride: imageView.displayStride,
      pixel_scale_arcsec: imageView.pixelScaleArcsec,
      view: {
        scale: imageView.scale,
        x: imageView.x,
        y: imageView.y,
      },
      cutout_center: imageView.cutoutCenter,
      cutout_size: Number(cutoutSize.value),
      cutout_artifact: {
        fits_path: cutoutState.fitsPath,
        preview_url: cutoutState.previewUrl,
        shape: cutoutState.shape,
        bounds: cutoutState.bounds,
      },
      bg_box_center: imageView.bgBoxCenter,
      bg_box_size: Number(bgBoxSize.value),
      mtf: {
        shadows: Number(mtfShadows.value),
        midtones: Number(mtfMidtones.value),
        highlights: Number(mtfHighlights.value),
      },
    },
    psf: {
      mode: psfState.mode,
      input_path: psfInputPath.value.trim() || psfState.inputPath,
      science_path: psfSciencePath.value.trim(),
      kernel_size: Number(psfKernelSize.value),
      fwhm: Number(psfFwhm.value),
      threshold_sigma: Number(psfThresholdSigma.value),
      selected_ids: psfSelectedIds.value,
      svi_steps: Number(psfSviSteps.value),
      ss_factor: Number(psfSsFactor.value),
      preview_view: psfState.previewView,
      detected_stars: psfState.detectedStars,
      full_image_source_path: psfState.fullImageSourcePath,
      full_image_preview_src: psfState.fullImagePreviewSrc,
      full_image_shape: psfState.fullImageShape,
      full_image_display_stride: psfState.fullImageDisplayStride,
      main_preview_data: psfState.mainPreviewData,
      job_id: psfState.jobId,
      job_status: psfState.jobStatus,
    },
    mask: {
      tool: maskState.tool,
      mode: maskState.mode,
      type: maskState.type,
      brush_radius: Number(maskBrushRadius.value),
      source_path: maskState.sourcePath,
      preview_source: maskState.previewSource,
      preview_colormap: maskState.previewColormap,
      bounds: maskState.bounds,
      saved_path: maskState.savedPath,
      polygon_points: maskState.polygonPoints,
      conjugate_points: maskState.conjugatePoints,
      conjugate_points_source1: maskState.conjugatePoints,
      conjugate_points_source2: maskState.conjugatePointsSource2,
      conjugate_plane: maskState.conjugatePlane,
      conjugate_measure_source: maskState.conjugateMeasureSource,
      conjugate_click_mode: maskState.conjugateClickMode,
      lens_light_job_id: maskState.lensLightJobId,
      lens_light_preview_url: maskState.lensLightPreviewUrl,
      lens_light_job_status: maskState.lensLightJobStatus,
      lens_light_image: maskState.lensLightImage,
      mask_mtf: {
        shadows: Number(maskCutoutMtfShadows.value),
        midtones: Number(maskCutoutMtfMidtones.value),
        highlights: Number(maskCutoutMtfHighlights.value),
      },
      lens_light_mtf: {
        shadows: Number(maskSubtractionMtfShadows.value),
        midtones: Number(maskSubtractionMtfMidtones.value),
        highlights: Number(maskSubtractionMtfHighlights.value),
      },
      lens_light_settings: {
        n_gauss: Number(maskLensLightNGauss.value),
        sigma_min: Number(maskLensLightSigmaMin.value),
        sigma_max: maskLensLightSigmaMax.value.trim(),
        center_max_offset: Number(maskLensLightCenterMaxOffset.value),
        semilinear_mge_steps: Number(maskLensLightSviSteps.value),
        constrained_svi_steps: Number(maskLensLightSviSteps.value),
        unconstrained_svi_steps: Number(maskLensLightUnconstrainedSviSteps.value),
        svi_steps: Number(maskLensLightUnconstrainedSviSteps.value),
        result_choice: maskState.lensLightResultChoice,
        background_corner: maskLensLightBgCorner.value.trim(),
      },
      data:
        maskState.data && maskState.width > 0 && maskState.height > 0
          ? maskRowsForSave()
          : null,
    },
    lens: {
      preview_data: compactLensPreviewForProject(lensPreviewData),
      selected_chain: lensState.selectedChain,
      rating: currentLensRating(),
      lens_light_prior_auto: lensState.lensLightPriorAuto,
      script_stale: lensState.scriptStale,
      script_stale_reason: lensState.scriptStaleReason,
      progress: lensProgressState,
      job_id: lensJobId,
      form_payload: collectFormPayload(document.querySelector('[data-form="lens-model"]')),
      generated_code: lensState.generatedCode,
      model_config: lensState.modelConfig,
    },
  };
}

async function saveProjectNow() {
  if (projectState.restoring) return;
  const currentProjectName = canonicalProjectName({
    project_id: projectState.id,
    project_name: projectName.value.trim() || projectState.name,
    project_folder: projectState.folder || currentProjectFolderPath() || "",
  });
  const projectFolderValue = projectState.folder || currentProjectFolderPath() || "";
  const payload = {
    project_id: projectState.id || undefined,
    project_name: currentProjectName || projectState.name || projectState.id || undefined,
    project_folder: projectFolderValue || undefined,
    comment: projectState.comment || "",
    rating: currentLensRating(),
    state: snapshotProjectState(),
  };
  try {
    const data = await callBackend("/api/project/save", payload);
    projectState.id = data.project_id || projectState.id;
    const previousFolder = projectState.folder;
    projectState.folder = data.project_folder || projectState.folder;
    projectState.name = canonicalProjectName({
      project_id: projectState.id,
      project_name: data.project_name || currentProjectName || projectState.name,
      project_folder: projectState.folder,
    });
    if (previousFolder !== projectState.folder) resetProjectThumbnailCache();
    if (projectName.value !== projectState.name) {
      projectName.value = projectState.name;
    }
    window.clearTimeout(projectState.saveRetryTimer);
    projectState.saveRetryTimer = null;
    projectState.saveRetryAttempt = 0;
    await refreshProjectList();
    return data;
  } catch (error) {
    appendLog(`Project save: ${error.message}`);
    const retryable = /failed to fetch|networkerror|load failed/i.test(String(error.message || ""));
    if (retryable && projectState.saveRetryAttempt < 5 && !projectState.saveRetryTimer) {
      const delay = Math.min(2000 * 2 ** projectState.saveRetryAttempt, 30000);
      projectState.saveRetryAttempt += 1;
      projectState.saveRetryTimer = window.setTimeout(() => {
        projectState.saveRetryTimer = null;
        saveProjectNow();
      }, delay);
      appendLog(`Project save: retrying in ${Math.round(delay / 1000)} s.`);
    }
    return null;
  }
}

async function renameProject() {
  const requestedName = projectName.value.trim();
  if (!requestedName) {
    appendLog("Rename project: enter a project name.");
    setActivity("Rename project: enter a project name.", false);
    return;
  }
  window.clearTimeout(projectState.saveTimer);
  const previousName = projectState.name || projectState.id || "";
  const previousFolder = projectState.folder || "";
  const payload = {
    project_id: projectState.id || undefined,
    project_name: requestedName,
    project_folder: currentProjectFolderPath() || undefined,
    state: snapshotProjectState(),
  };
  try {
    const data = await callBackend("/api/project/rename", payload);
    projectState.id = data.project_id || projectState.id;
    projectState.name = data.project_name || requestedName;
    projectState.folder = data.project_folder || projectState.folder;
    if (previousFolder !== projectState.folder) resetProjectThumbnailCache();
    projectName.value = projectState.name;
    folderPath.value = projectState.folder;
    await refreshProjectList();
    syncProjectSelect();
    appendLog(`Renamed project${previousName ? ` ${previousName}` : ""} -> ${projectState.name}.`);
    if (previousFolder && previousFolder !== projectState.folder) {
      appendLog(`Project folder moved: ${previousFolder} -> ${projectState.folder}`);
    }
    setActivity(`Project renamed: ${projectState.name}`, false);
  } catch (error) {
    appendLog(`Rename project: ${error.message}`);
    setActivity(`Rename project failed: ${error.message}`, false);
  }
}

async function refreshProjectList() {
  try {
    const data = await callBackend("/api/projects");
    projectSelect.innerHTML = "";
    const projects = data.projects || [];
    if (!projects.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No saved projects";
      projectSelect.appendChild(option);
      return;
    }
    const nameCounts = new Map();
    projects.forEach((project) => {
      const label = canonicalProjectName(project);
      nameCounts.set(label, (nameCounts.get(label) || 0) + 1);
    });
    for (const project of projects) {
      const option = document.createElement("option");
      const label = canonicalProjectName(project);
      const idSuffix = nameCounts.get(label) > 1 ? ` (${project.project_id})` : "";
      option.value = project.project_id;
      option.textContent = `${label}${idSuffix}`;
      option.title = `id: ${project.project_id}\nfolder: ${project.project_folder || ""}${project.saved_at ? `\nsaved: ${project.saved_at}` : ""}`;
      option.selected = project.project_id === projectState.id;
      projectSelect.appendChild(option);
    }
    syncProjectSelect();
  } catch (error) {
    appendLog(`Project list: ${error.message}`);
  }
}

function syncProjectSelect() {
  if (!projectState.id) return;
  const option = [...projectSelect.options].find((item) => item.value === projectState.id);
  if (option) {
    projectSelect.value = projectState.id;
  } else {
    const current = document.createElement("option");
    current.value = projectState.id;
    current.textContent = `${projectState.name || projectState.id} (current)`;
    current.selected = true;
    projectSelect.prepend(current);
    projectSelect.value = projectState.id;
  }
}

function scheduleProjectSave(event) {
  if (event?.target?.closest?.("#project-chooser-modal, #settings-modal")) return;
  updateWorkflowStatus();
  if (projectState.restoring) return;
  window.clearTimeout(projectState.saveRetryTimer);
  projectState.saveRetryTimer = null;
  projectState.saveRetryAttempt = 0;
  window.clearTimeout(projectState.saveTimer);
  projectState.saveTimer = window.setTimeout(saveProjectNow, 350);
}

function refreshThumbnailBackedPreviews() {
  refreshVisibleMaskPreview();
  renderLensScriptCutoutPreview();
}

function currentCutoutMtfPreviewUrl() {
  const folderName = PathName(projectState.folder || projectState.id || "");
  if (!folderName) return "";
  return `/runs/${encodeURIComponent(folderName)}/previews/Data_cutout_preview.png`;
}

async function saveCurrentMtfCutoutPreviewNow() {
  if (projectState.restoring || !projectState.folder || !cutoutState.fitsPath) return;
  if (!imagePreviewCanvas.width || !imagePreviewCanvas.height) return;
  if (!renderCurrentMtfCutoutToCanvas(cutoutPreviewCanvas)) return;
  try {
    const data = await callBackend("/api/project/cutout-preview", {
      project_id: projectState.id || undefined,
      project_folder: projectState.folder,
      preview_data_url: cutoutPreviewCanvas.toDataURL("image/png"),
    });
    cutoutState.previewUrl = data.preview_url || cutoutState.previewUrl || currentCutoutMtfPreviewUrl();
    lensScriptCutoutState.previewSrc = cutoutState.previewUrl;
    if (maskState.data) {
      maskState.previewSrc = cutoutState.previewUrl;
      refreshVisibleMaskPreview();
    }
    renderLensScriptCutoutPreview();
    projectThumbnailState.version = String(Date.now());
    resetProjectThumbnailCache(projectThumbnailUrl());
  } catch (error) {
    appendLog(`Cutout preview: ${error.message}`);
  }
}

function scheduleCurrentMtfCutoutPreviewSave() {
  if (projectState.restoring) return;
  window.clearTimeout(projectState.cutoutPreviewTimer);
  projectState.cutoutPreviewTimer = window.setTimeout(saveCurrentMtfCutoutPreviewNow, 300);
}

async function saveCurrentMtfThumbnailNow() {
  if (projectState.restoring || !projectState.folder || !imageView.original) return;
  if (!imagePreviewCanvas.width || !imagePreviewCanvas.height) return;
  try {
    const data = await callBackend("/api/project/thumbnail", {
      project_id: projectState.id || undefined,
      project_folder: projectState.folder,
      thumbnail_data_url: imagePreviewCanvas.toDataURL("image/png"),
    });
    projectThumbnailState.dirty = false;
    projectThumbnailState.version = String(Date.now());
    resetProjectThumbnailCache(projectThumbnailUrl());
    refreshThumbnailBackedPreviews();
  } catch (error) {
    appendLog(`Project thumbnail: ${error.message}`);
  }
}

function scheduleCurrentMtfThumbnailSave() {
  if (projectState.restoring) return;
  window.clearTimeout(projectState.thumbnailTimer);
  projectState.thumbnailTimer = window.setTimeout(saveCurrentMtfThumbnailNow, 300);
}

function coordinateProjectToken(value, { signed = false, dspl = false } = {}) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "";
  if (dspl) {
    const sign = signed && numeric < 0 ? "NEG" : "";
    const valueText = Math.abs(numeric).toFixed(6).replace(/\.?0+$/, "").replace(".", "_");
    return `${sign}${valueText}`;
  }
  const sign = signed ? (numeric < 0 ? "m" : "p") : "";
  return `${sign}${Math.abs(numeric).toFixed(6).replace(".", "p")}`;
}

function coordinateProjectNameFromInputs(prefix = "") {
  const isDspl = prefix === "DSPL";
  const raToken = coordinateProjectToken(euclidRa.value, { dspl: isDspl });
  const decToken = coordinateProjectToken(euclidDec.value, { signed: true, dspl: isDspl });
  if (isDspl) return raToken && decToken ? `DSPL_RA${raToken}DEC${decToken}` : "";
  const prefixText = prefix ? `${prefix}_` : "";
  return raToken && decToken ? `${prefixText}RA${raToken}_DEC${decToken}` : "";
}

function euclidProjectNameFromInputs() {
  return coordinateProjectNameFromInputs("Euclid");
}

function isAutoProjectName(value) {
  const text = String(value || "").trim();
  return !text || text === projectState.id || /^\d{8}_\d{6}$/.test(text) || /^Euclid_RA/.test(text) || /^RA/.test(text) || /^DSPL_?RA/.test(text);
}

function canonicalProjectName(project = {}) {
  const rawName = String(project.project_name || "").trim();
  const projectId = String(project.project_id || "").trim();
  const folderName = PathName(project.project_folder || "");
  return folderName || projectId || rawName || "Unnamed project";
}

function setProjectIdentity(data = {}, fallbackId = "") {
  projectState.id = data.project_id || fallbackId || projectState.id || "";
  projectState.folder = data.project_folder || projectState.folder || "";
  projectState.comment = Object.prototype.hasOwnProperty.call(data, "comment") ? String(data.comment || "") : projectState.comment || "";
  projectState.name = canonicalProjectName({
    project_id: projectState.id,
    project_name: data.project_name || projectState.name,
    project_folder: projectState.folder,
  });
  projectName.value = projectState.name;
  projectName.title = projectState.name;
  updateWorkflowStatus();
}

function autoProjectNameFromCoordinates(options = {}) {
  const currentName = String(projectName.value || "").trim();
  if (options.keepEuclidPrefix && /^Euclid_RA/.test(currentName)) return true;
  const prefix = options.prefix || (dsplEnabled?.checked ? "DSPL" : "");
  const name = coordinateProjectNameFromInputs(prefix);
  if (!name) return false;
  if (!options.force && projectState.folder) return false;
  if (!options.force && !isAutoProjectName(projectName.value)) return false;
  projectName.value = name;
  projectState.name = name;
  if (options.save !== false) {
    scheduleProjectSave();
  }
  return true;
}

async function generateEuclidProjectName() {
  if (!autoProjectNameFromCoordinates({ force: true, save: false })) {
    appendLog("Project name: enter finite RA and Dec first.");
    return;
  }
  if (projectState.id || projectState.folder) {
    await renameProject();
  } else {
    await saveProjectNow();
  }
  appendLog(`Project name set: ${projectName.value}`);
}

async function useImageFolderAsProjectFolder() {
  const imagePath = inputImagePath.value.trim() || imageView.sourcePath;
  const imageFolder = dirname(imagePath);
  if (!imageFolder) {
    appendLog("Project folder: choose or enter an image file path first.");
    return;
  }
  folderPath.value = imageFolder;
  projectState.folder = imageFolder;
  projectState.name = canonicalProjectName(projectState);
  projectName.value = projectState.name;
  appendLog(`Project folder set from image folder: ${imageFolder}`);
  await loadFolder(imageFolder);
  await saveProjectNow();
}

async function defaultImageDataFolderToProject(options = {}) {
  if (!projectState.folder) return false;
  folderPath.value = projectState.folder;
  if (options.load !== false) {
    await loadFolder(projectState.folder);
  }
  if (options.save) {
    scheduleProjectSave();
  }
  return true;
}

async function restoreProjectState(state) {
  if (!state) return;
  projectState.restoring = true;
  window.clearTimeout(projectState.saveTimer);
  window.clearTimeout(projectState.thumbnailTimer);
  resetMaskRuntimeState();
  resetLensRuntimeState();
  lensState.lensLightPriorAuto = state.lens?.lens_light_prior_auto !== false;
  lensLightHandoffKey = "";
  try {
    setActivity(`Restoring project ${projectState.name || projectState.id || ""}`.trim(), true);
    if (state.theme) applyTheme(state.theme);
    if (projectState.folder) {
      folderPath.value = projectState.folder;
    } else if (state.data_folder) {
      folderPath.value = state.data_folder;
    }
    if (state.euclid) {
      euclidRa.value = Number.isFinite(Number(state.euclid.ra)) ? String(state.euclid.ra) : "";
      euclidDec.value = Number.isFinite(Number(state.euclid.dec)) ? String(state.euclid.dec) : "";
      euclidCutoutSize.value = String(state.euclid.cutout_size || 10);
      euclidUsername.value = state.euclid.username || "";
      euclidState.rmsMapPath = projectReferenceIsCurrentOrExternal(state.euclid.rms_map_path) ? state.euclid.rms_map_path || "" : "";
      euclidState.rmsBundlePath = projectReferenceIsCurrentOrExternal(state.euclid.rms_bundle_path) ? state.euclid.rms_bundle_path || "" : "";
      if (state.euclid.mtf) {
        applySharedMtfValues(state.euclid.mtf, { save: false });
      }
      if (state.euclid.photometry) {
        euclidMagSystem.value = state.euclid.photometry.mag_system || "ab";
        euclidApertureRadius.value = String(state.euclid.photometry.aperture_radius ?? 2);
        euclidAnnulusWidth.value = String(state.euclid.photometry.annulus_width ?? 2);
        euclidPreviewScale.value = String(state.euclid.photometry.preview_scale ?? 100);
        euclidState.displayMode = state.euclid.photometry.display_mode || "flux";
        euclidSbLower.value = state.euclid.photometry.sb_lower ?? "";
        euclidSbUpper.value = state.euclid.photometry.sb_upper ?? "";
        euclidState.photozEnabled = Boolean(state.euclid.photometry.photoz_enabled);
        applyPhotozBandSelection(state.euclid.photometry.photoz_bands, { save: false });
        euclidPhotozToggle.classList.toggle("active", euclidState.photozEnabled);
        euclidPhotozToggle.textContent = euclidState.photozEnabled ? "测光红移 On" : "测光红移";
        updateEuclidDisplayModeButton();
      }
      if (state.euclid.image_options) {
        Object.entries(state.euclid.image_options).forEach(([key, entry]) => {
          if (!projectReferenceIsCurrentOrExternal(entry)) {
            appendLog(`Project restore: ignored Euclid cutout option "${key}" from another project.`);
            return;
          }
          if (entry?.image?.data) {
            registerCutoutImage(key, entry.label || entry.image.label || key, entry.image);
          }
        });
      }
      if (state.euclid.image && !projectReferenceIsCurrentOrExternal(state.euclid.image)) {
        clearEuclidImagePreview("Stored cutout preview belonged to another project.");
        appendLog("Project restore: ignored Euclid cutout preview from another project.");
      } else if (state.euclid.image?.source === "legacy" && state.euclid.image?.data) {
        registerCutoutImage(state.euclid.image.image_key || "restored-cutout", state.euclid.image.label || "Restored cutout", state.euclid.image);
        loadEuclidImagePayload(state.euclid.image);
      } else if (state.euclid.image?.data && state.euclid.image?.path && state.euclid.image?.source === "euclid") {
        const preferredBand = state.euclid.image.band || "VIS";
        const restoredImage = await loadEuclidFitsFamily(state.euclid.image.path, preferredBand);
        if (restoredImage) {
          loadEuclidImagePayload(restoredImage);
        } else {
          registerCutoutImage(state.euclid.image.image_key || "restored-cutout", state.euclid.image.label || "Restored cutout", state.euclid.image);
          loadEuclidImagePayload(state.euclid.image);
        }
      } else if (state.euclid.image?.path) {
        setActivity(`Loading Euclid FITS ${PathName(state.euclid.image.path)}`, true);
        try {
          const preferredBand = state.euclid.image.band || "VIS";
          const restoredImage = await loadEuclidFitsFamily(state.euclid.image.path, preferredBand);
          if (!restoredImage) throw new Error("No Euclid sibling FITS could be loaded.");
          loadEuclidImagePayload(restoredImage);
        } catch (error) {
          appendLog(`Euclid cutout restore: ${error.message}`);
          if (state.euclid.image?.data) {
            registerCutoutImage(state.euclid.image.image_key || "restored-cutout", state.euclid.image.label || "Restored cutout", state.euclid.image);
            loadEuclidImagePayload(state.euclid.image);
          }
        }
      } else if (state.euclid.image?.data) {
        registerCutoutImage(state.euclid.image.image_key || "restored-cutout", state.euclid.image.label || "Restored cutout", state.euclid.image);
        loadEuclidImagePayload(state.euclid.image);
      } else if (!Object.keys(euclidState.imageOptions).length) {
        clearEuclidImagePreview();
      }
      if (!projectName.value.trim() || /^\d{8}_\d{6}$/.test(projectName.value.trim())) {
        autoProjectNameFromCoordinates({ save: false });
      }
    }
    if (state.image?.cutout_size) cutoutSize.value = String(state.image.cutout_size);
    if (state.image?.input_path) inputImagePath.value = state.image.input_path;
    if (state.image?.rms_input_path) inputRmsPath.value = state.image.rms_input_path;
    if (state.image?.rms_path && projectReferenceIsCurrentOrExternal(state.image.rms_path)) {
      imageView.rmsPath = state.image.rms_path;
      selectedRmsName.textContent = state.image.rms_selected_name || state.image.rms_path;
      if (!inputRmsPath.value.trim()) inputRmsPath.value = state.image.rms_path;
    } else {
      imageView.rmsPath = "";
      selectedRmsName.textContent = "No RMS file selected";
      if (!state.image?.rms_input_path) inputRmsPath.value = "";
    }
    if (state.image?.bg_box_size) bgBoxSize.value = String(state.image.bg_box_size);
    if (state.image?.image_shape || state.image?.pixel_scale_arcsec) {
      setImagePreviewMetadata({
        image_shape: state.image.image_shape,
        source_shape: state.image.source_shape,
        display_stride: state.image.display_stride,
        pixel_scale_arcsec: state.image.pixel_scale_arcsec,
      });
    }
    if (state.image?.mtf) {
      applySharedMtfValues(state.image.mtf, { save: false });
    }

    const savedImageStateValid = !state.image || projectReferenceIsCurrentOrExternal(state.image);
    if (state.image && !savedImageStateValid) {
      appendLog("Project restore: ignored image preprocess state from another project.");
    }
    if (state.image?.source_path && savedImageStateValid) {
      setActivity(`Loading project image ${PathName(state.image.source_path)}`, true);
      await loadImageFromPath(state.image.source_path, {
        activity: false,
        registerArtifacts: !state.image.cutout_artifact?.fits_path,
        registerDefaultCutout: !state.image.cutout_artifact?.fits_path,
      });
      projectState.thumbnailRefreshAfterRestore = true;
      if (state.image.cutout_center) {
        imageView.cutoutCenter = state.image.cutout_center;
      }
      if (state.image.bg_box_center) {
        imageView.bgBoxCenter = state.image.bg_box_center;
      }
      imageView.needsLayoutCenter = true;
      if (state.image.cutout_artifact?.fits_path) {
        cutoutState.fitsPath = state.image.cutout_artifact.fits_path || "";
        cutoutState.previewUrl = state.image.cutout_artifact.preview_url || "";
        cutoutState.shape = state.image.cutout_artifact.shape || null;
        cutoutState.bounds = state.image.cutout_artifact.bounds || null;
        cutoutPreviewLabel.textContent = PathName(cutoutState.fitsPath);
        cutoutPreviewPanel.hidden = false;
      }
      renderMtfPreview();
      centerImagePreviewAfterLayout();
      if (!cutoutPreviewPanel.hidden) {
        renderCutoutPreviewFromCanvas();
      }
    }

    if (state.psf) {
      setActivity("Restoring PSF state", true);
      const savedPsfSciencePath = projectReferenceIsCurrentOrExternal(state.psf.science_path) ? state.psf.science_path || "" : "";
      const savedPsfFullImageValid = projectReferenceIsCurrentOrExternal({
        source_path: state.psf.full_image_source_path || "",
        preview_src: state.psf.full_image_preview_src || "",
        detected_stars: state.psf.detected_stars || [],
      });
      const savedPsfMainPreviewValid = !state.psf.main_preview_data || projectReferenceIsCurrentOrExternal(state.psf.main_preview_data);
      if ((state.psf.science_path || state.psf.full_image_source_path || state.psf.full_image_preview_src) && (!savedPsfSciencePath || !savedPsfFullImageValid)) {
        appendLog("Project restore: ignored PSF full-image state from another project.");
      }
      if (state.psf.main_preview_data && !savedPsfMainPreviewValid) {
        appendLog("Project restore: ignored PSF product preview from another project.");
      }
      psfInputPath.value = state.psf.input_path || "";
      psfSciencePath.value = savedPsfSciencePath;
      psfKernelSize.value = String(state.psf.kernel_size || 31);
      psfFwhm.value = String(state.psf.fwhm || 1.6);
      psfThresholdSigma.value = String(state.psf.threshold_sigma || 5);
      psfSelectedIds.value = state.psf.selected_ids || "";
      const restoredPsfSteps = Number(state.psf.svi_steps ?? 2000);
      psfSviSteps.value = String(Number.isFinite(restoredPsfSteps) && restoredPsfSteps >= 100 ? restoredPsfSteps : 2000);
      psfSsFactor.value = String([1, 3].includes(Number(state.psf.ss_factor)) ? Number(state.psf.ss_factor) : 3);
      psfState.detectedStars = savedPsfFullImageValid ? state.psf.detected_stars || [] : [];
      psfState.fullImageSourcePath = savedPsfFullImageValid ? state.psf.full_image_source_path || "" : "";
      psfState.fullImagePreviewSrc = savedPsfFullImageValid ? state.psf.full_image_preview_src || "" : "";
      psfState.fullImageShape = savedPsfFullImageValid ? state.psf.full_image_shape || null : null;
      psfState.fullImageDisplayStride = savedPsfFullImageValid ? Math.max(1, Number(state.psf.full_image_display_stride || 1)) : 1;
      psfState.mainPreviewData = savedPsfMainPreviewValid ? state.psf.main_preview_data || null : null;
      psfState.jobId = state.psf.job_id || "";
      psfState.jobStatus = state.psf.job_status || null;
      setPsfMode(state.psf.mode || "input");
      syncPsfSelectedIdsFromInput();
      renderPsfCandidates(psfState.detectedStars);
      if (psfState.mainPreviewData) {
        setPsfMainPreview(psfState.mainPreviewData, { persist: false });
      }
      void setPsfPreviewView(state.psf.preview_view || "product");
    }

    if (state.mask) {
      setActivity("Restoring mask state", true);
      maskBrushRadius.value = String(Math.max(0, Math.min(20, Number(state.mask.brush_radius ?? 4))));
      setMaskType(state.mask.type || "mask_1");
      setMaskTool(state.mask.tool || "line");
      setMaskMode(state.mask.mode || "add");
      maskState.savedPath = state.mask.saved_path || "";
      const savedLensLightState = {
        preview_url: state.mask.lens_light_preview_url || "",
        job_status: state.mask.lens_light_job_status || null,
        image: state.mask.lens_light_image || null,
      };
      const hasSavedLensLightState = Boolean(
        savedLensLightState.preview_url || savedLensLightState.job_status || savedLensLightState.image,
      );
      const savedLensLightValid = !hasSavedLensLightState || projectReferenceBelongsHere(savedLensLightState);
      if (hasSavedLensLightState && !savedLensLightValid) {
        appendLog("Project restore: ignored lens-light subtraction from another project.");
      }
      const savedConjugatePointsSource1 = Array.isArray(state.mask.conjugate_points_source1)
        ? state.mask.conjugate_points_source1
        : Array.isArray(state.mask.conjugate_points)
          ? state.mask.conjugate_points
          : [];
      const savedConjugatePointsSource2 = Array.isArray(state.mask.conjugate_points_source2) ? state.mask.conjugate_points_source2 : [];
      maskState.conjugatePoints = filterConjugatePointsForProject(savedConjugatePointsSource1, savedLensLightValid);
      maskState.conjugatePointsSource2 = filterConjugatePointsForProject(savedConjugatePointsSource2, savedLensLightValid);
      setConjugatePlane(state.mask.conjugate_plane || "source1");
      setConjugateMeasureSource(state.mask.conjugate_measure_source || "mask");
      setConjugateClickMode(state.mask.conjugate_click_mode || "brightest");
      updateConjugateStatus();
      maskState.lensLightJobId = savedLensLightValid ? state.mask.lens_light_job_id || "" : "";
      maskState.lensLightPreviewUrl = savedLensLightValid ? state.mask.lens_light_preview_url || "" : "";
      maskState.lensLightJobStatus = savedLensLightValid ? state.mask.lens_light_job_status || null : null;
      maskState.lensLightImage = savedLensLightValid ? state.mask.lens_light_image || null : null;
      if (state.mask.mask_mtf) {
        setMaskCutoutMtfValues(state.mask.mask_mtf);
      } else {
        setMaskCutoutMtfValues(DEFAULT_MASK_MTF_VALUES);
      }
      setMaskPreviewColormap(state.mask.preview_colormap || "twilight", { save: false, refresh: false });
      if (state.mask.lens_light_mtf) {
        setMaskSubtractionMtfValues(state.mask.lens_light_mtf);
      } else if (state.mask.mask_mtf) {
        setMaskSubtractionMtfValues(state.mask.mask_mtf);
      } else {
        setMaskSubtractionMtfValues(DEFAULT_MASK_MTF_VALUES);
      }
      if (state.mask.lens_light_settings) {
        maskLensLightNGauss.value = String(state.mask.lens_light_settings.n_gauss ?? 8);
        maskLensLightSigmaMin.value = String(state.mask.lens_light_settings.sigma_min ?? 0.01);
        maskLensLightSigmaMax.value = String(state.mask.lens_light_settings.sigma_max ?? "auto");
        maskLensLightCenterMaxOffset.value = String(state.mask.lens_light_settings.center_max_offset ?? 0.4);
        if (["", "auto", "half", "half_image", "half image"].includes(maskLensLightSigmaMax.value.trim().toLowerCase())) {
          maskLensLightSigmaMax.dataset.dynamicDefault = "true";
        } else {
          delete maskLensLightSigmaMax.dataset.dynamicDefault;
        }
        maskLensLightSviSteps.value = String(
          state.mask.lens_light_settings.semilinear_mge_steps ??
            state.mask.lens_light_settings.constrained_svi_steps ??
            0,
        );
        maskLensLightUnconstrainedSviSteps.value = String(
          state.mask.lens_light_settings.unconstrained_svi_steps ??
            state.mask.lens_light_settings.svi_steps ??
            state.mask.lens_light_settings.init_svi_steps ??
            2000,
        );
        maskState.lensLightResultChoice = state.mask.lens_light_settings.result_choice || "unconstrained";
        if (maskLensLightResultChoice) maskLensLightResultChoice.value = maskState.lensLightResultChoice;
        maskLensLightBgCorner.value = String(state.mask.lens_light_settings.background_corner ?? "");
      }
      setMaskSubtractionPreview(maskState.lensLightPreviewUrl, maskState.lensLightJobStatus);
      await refreshLensLightSubtractionFromProject({ silent: true, save: false });
      setMaskPreviewSource(state.mask.preview_source || "image", { silent: true, save: false });
      const loadedProjectMask = await loadProjectMask(maskState.type, { silent: true });
      if (!loadedProjectMask) {
        maskStatus.textContent = "Mask Preview requires Data_cutout.fits.";
      }
      if (maskState.savedPath) {
        maskStatus.textContent = `Saved: ${PathName(maskState.savedPath)}`;
      }
    } else {
      await loadProjectMask(maskState.type, { silent: true });
      await refreshLensLightSubtractionFromProject({ silent: true, save: false });
    }

    const savedLensPreviewData = normalizeLensPreviewData(state.lens?.preview_data || null);
    if (savedLensPreviewData && !projectReferenceBelongsHere(savedLensPreviewData)) {
      appendLog("Project restore: ignored lens model preview from another project.");
    } else if (savedLensPreviewData) {
      setActivity("Restoring lens model state", true);
      lensState.selectedChain = state.lens.selected_chain || "";
      updatePreview("lens-model", savedLensPreviewData);
    }
    if (state.lens?.progress) {
      setLensProgress(state.lens.progress);
    }
    if (state.lens?.job_id) {
      lensState.jobId = state.lens.job_id;
    }
    lensState.rating = normalizedProjectRating(state.lens?.rating ?? data.rating ?? 0);
    if (lensResultRating) {
      lensResultRating.value = String(lensState.rating);
    }
    if (state.lens?.form_payload) {
      applyFormPayload(document.querySelector('[data-form="lens-model"]'), state.lens.form_payload);
    } else if (state.lens?.model_config) {
      applyFormPayload(document.querySelector('[data-form="lens-model"]'), lensModelConfigToFormPayload(state.lens.model_config));
    }
    if (state.lens?.generated_code) {
      setLensGeneratedCode(state.lens.generated_code, state.lens.model_config || null);
    } else {
      setLensGeneratedCode("", null);
    }
    await restoreLensModelResultFromProject({ silent: true });
    syncLensLightPriorFromMask(maskState.lensLightJobStatus, { silent: true });
    if (state.lens?.script_stale) {
      markLensScriptStale(state.lens.script_stale_reason || "Lens model settings changed");
    }

    renderSharedMtfPreviews({ save: false });
    if (state.active_page) setPage(state.active_page);
    setActivity(`Project ${projectState.name || projectState.id || ""} loaded`.trim(), false);
  } finally {
    projectState.restoring = false;
    if (state.active_page === "lens-model") {
      window.setTimeout(() => {
        void restoreLensModelResultFromProject({ silent: true });
      }, 0);
    }
    if (projectState.thumbnailRefreshAfterRestore) {
      projectState.thumbnailRefreshAfterRestore = false;
      window.setTimeout(() => {
        void saveCurrentMtfThumbnailNow();
        void saveCurrentMtfCutoutPreviewNow();
      }, 0);
    }
  }
}

async function loadLatestProject() {
  setActivity("Loading latest project metadata", true);
  try {
    const data = await callBackend("/api/project/latest");
    setProjectIdentity(data);
    projectThumbnailState.version = "";
    resetProjectThumbnailCache();
    projectThumbnailState.dirty = false;
    await defaultImageDataFolderToProject({ load: false });
    syncProjectSelect();
    if (data.exists && data.state) {
      setActivity(`Loading project ${projectState.name || projectState.id}`, true);
      await restoreProjectState(data.state);
      appendLog(`Loaded project ${projectState.name || projectState.id}.`);
    } else {
      setActivity("No saved project loaded", false);
    }
  } catch (error) {
    setActivity(`Project load failed: ${error.message}`, false);
    appendLog(`Project load: ${error.message}`);
  }
}

async function startNewModelProject() {
  projectState.restoring = true;
  window.clearTimeout(projectState.saveTimer);
  window.clearTimeout(projectState.thumbnailTimer);
  try {
    const currentProjectLoaded = Boolean(projectState.id || projectState.folder);
    const requestedName = currentProjectLoaded
      ? ""
      : coordinateProjectNameFromInputs("") || projectName.value.trim();
    const data = await callBackend("/api/project/new", requestedName ? { project_name: requestedName } : {});
    projectState.id = data.project_id || "";
    projectState.name = data.project_name || projectState.id;
    projectState.folder = data.project_folder || "";
    projectState.comment = data.comment || "";
    folderPath.value = projectState.folder;
    projectName.value = projectState.name;
    projectThumbnailState.version = "";
    resetProjectThumbnailCache();
    projectThumbnailState.dirty = false;
    window.location.reload();
  } catch (error) {
    projectState.restoring = false;
    appendLog(`Start new project: ${error.message}`);
  }
}

async function loadProjectById(projectId) {
  if (!projectId || projectId === projectState.id) return;
  projectState.restoring = true;
  window.clearTimeout(projectState.saveTimer);
  window.clearTimeout(projectState.thumbnailTimer);
  resetMaskRuntimeState();
  resetLensRuntimeState();
  setActivity(`Loading project ${projectId}`, true);
  try {
    const data = await callBackend("/api/project/load", { project_id: projectId });
    setProjectIdentity(data, projectId);
    projectThumbnailState.version = "";
    resetProjectThumbnailCache();
    projectThumbnailState.dirty = false;
    syncProjectSelect();
    await restoreProjectState(data.state || {});
    appendLog(data.message || `Loaded project ${projectState.name || projectId}.`);
  } catch (error) {
    setActivity(`Project load failed: ${error.message}`, false);
    appendLog(`Project load: ${error.message}`);
  } finally {
    projectState.restoring = false;
  }
}

async function loadSelectedProject() {
  await loadProjectById(projectSelect.value);
}

function closeProjectChooser() {
  projectChooserModal.hidden = true;
  hideProjectContextMenu();
}

function projectDisplayName(project) {
  return canonicalProjectName(project);
}

function hideProjectContextMenu() {
  if (projectContextMenu) {
    projectContextMenu.remove();
    projectContextMenu = null;
  }
}

async function reloadProjectChooser() {
  projectChooserStatus.textContent = "Loading projects...";
  projectChooserGrid.innerHTML = "";
  const data = await callBackend("/api/projects");
  renderProjectChooser(data.projects || []);
}

function setProjectChooserView(view) {
  projectChooserView = view === "list" ? "list" : "thumbnail";
  localStorage.setItem("herculens-project-view", projectChooserView);
  projectChooserViewInputs.forEach((input) => {
    input.checked = input.value === projectChooserView;
  });
  renderProjectChooser(projectChooserProjects, { replace: false });
}

async function renameProjectFromChooser(project) {
  hideProjectContextMenu();
  const currentName = projectDisplayName(project);
  const nextName = window.prompt("Rename project", currentName);
  if (!nextName || nextName.trim() === currentName) return;
  projectChooserStatus.textContent = `Renaming ${currentName}...`;
  try {
    const data = await callBackend("/api/project/rename", {
      project_id: project.project_id,
      project_name: nextName.trim(),
    });
    appendLog(data.message || `Renamed project ${currentName}.`);
    if (project.project_id === projectState.id) {
      projectState.id = data.project_id || projectState.id;
      projectState.name = data.project_name || nextName.trim();
      projectState.folder = data.project_folder || projectState.folder;
      projectName.value = projectState.name;
    }
    await refreshProjectList();
    await reloadProjectChooser();
  } catch (error) {
    projectChooserStatus.textContent = `Rename failed: ${error.message}`;
    appendLog(`Project rename: ${error.message}`);
  }
}

function removeProjectFromChooser(projectId, card = null) {
  const targetCard =
    card || [...projectChooserGrid.querySelectorAll("[data-project-id]")].find((item) => item.dataset.projectId === projectId);
  targetCard?.remove();
  projectChooserProjects = projectChooserProjects.filter((project) => project.project_id !== projectId);
  const option = [...projectSelect.options].find((item) => item.value === projectId);
  option?.remove();
  renderProjectChooser(projectChooserProjects, { replace: false });
}

async function deleteProjectFromChooser(project, card = null) {
  hideProjectContextMenu();
  const name = projectDisplayName(project);
  const confirmed = window.confirm(
    `Move project "${name}" to /mnt/d/lensing/RecycleBin/?\n\n${project.project_folder || ""}`,
  );
  if (!confirmed) return;
  projectChooserStatus.textContent = `Moving ${name} to RecycleBin...`;
  try {
    const data = await callBackend("/api/project/delete", { project_id: project.project_id });
    appendLog(data.message || `Moved project ${name} to RecycleBin.`);
    removeProjectFromChooser(project.project_id, card);
    if (project.project_id === projectState.id) {
      window.clearTimeout(projectState.saveTimer);
      projectState.id = "";
      projectState.name = "";
      projectState.folder = "";
      projectState.comment = "";
      projectName.value = "";
      folderPath.value = "";
      setActivity("Current project moved to RecycleBin. Choose another project or start a new one.", false);
      syncProjectSelect();
    }
  } catch (error) {
    projectChooserStatus.textContent = `Delete failed: ${error.message}`;
    appendLog(`Project delete: ${error.message}`);
  }
}

async function copyProjectFromChooser(project) {
  hideProjectContextMenu();
  const name = projectDisplayName(project);
  projectChooserStatus.textContent = `Copying ${name}...`;
  try {
    const data = await callBackend("/api/project/copy", { project_id: project.project_id });
    appendLog(data.message || `Copied project ${name}.`);
    await refreshProjectList();
    await reloadProjectChooser();
    projectChooserStatus.textContent = `Copied ${name} to ${data.project_name || data.project_id}.`;
  } catch (error) {
    projectChooserStatus.textContent = `Copy failed: ${error.message}`;
    appendLog(`Project copy: ${error.message}`);
  }
}

async function editProjectCommentFromChooser(project) {
  hideProjectContextMenu();
  const name = projectDisplayName(project);
  const currentComment = String(project.comment || "");
  const nextComment = window.prompt(`Comment for ${name}`, currentComment);
  if (nextComment === null) return;
  projectChooserStatus.textContent = `Saving comment for ${name}...`;
  try {
    const data = await callBackend("/api/project/comment", {
      project_id: project.project_id,
      comment: nextComment.trim(),
    });
    appendLog(data.message || `Saved comment for ${name}.`);
    if (project.project_id === projectState.id) {
      projectState.comment = data.comment || "";
    }
    await refreshProjectList();
    await reloadProjectChooser();
    projectChooserStatus.textContent = `Comment saved for ${name}.`;
  } catch (error) {
    projectChooserStatus.textContent = `Comment failed: ${error.message}`;
    appendLog(`Project comment: ${error.message}`);
  }
}

async function rateProjectFromChooser(project, rating) {
  hideProjectContextMenu();
  const name = projectDisplayName(project);
  const nextRating = normalizedProjectRating(rating);
  projectChooserStatus.textContent = `Saving score ${nextRating} for ${name}...`;
  try {
    const data = await callBackend("/api/project/rating", {
      project_id: project.project_id,
      rating: nextRating,
    });
    appendLog(data.message || `Saved score ${nextRating} for ${name}.`);
    if (project.project_id === projectState.id) {
      lensState.rating = nextRating;
      if (lensResultRating) lensResultRating.value = String(nextRating);
    }
    await refreshProjectList();
    await reloadProjectChooser();
    projectChooserStatus.textContent = `Score ${nextRating} saved for ${name}.`;
  } catch (error) {
    projectChooserStatus.textContent = `Score failed: ${error.message}`;
    appendLog(`Project score: ${error.message}`);
  }
}

function showProjectContextMenu(project, x, y, card = null) {
  hideProjectContextMenu();
  const menu = document.createElement("div");
  menu.className = "project-context-menu";
  menu.setAttribute("role", "menu");

  const renameButton = document.createElement("button");
  renameButton.type = "button";
  renameButton.textContent = "Rename Project";
  renameButton.addEventListener("click", () => {
    void renameProjectFromChooser(project);
  });

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.textContent = "Copy Project";
  copyButton.addEventListener("click", () => {
    void copyProjectFromChooser(project);
  });

  const commentButton = document.createElement("button");
  commentButton.type = "button";
  commentButton.textContent = "Edit Comment";
  commentButton.addEventListener("click", () => {
    void editProjectCommentFromChooser(project);
  });

  const scoreGroup = document.createElement("div");
  scoreGroup.className = "project-context-score-group";
  const scoreLabel = document.createElement("div");
  scoreLabel.className = "project-context-score-label";
  scoreLabel.textContent = "Score";
  const scoreButtons = document.createElement("div");
  scoreButtons.className = "project-context-score-buttons";
  const currentRating = normalizedProjectRating(project.rating);
  for (let score = 0; score <= 5; score += 1) {
    const scoreButton = document.createElement("button");
    scoreButton.type = "button";
    scoreButton.textContent = String(score);
    scoreButton.className = score === currentRating ? "active" : "";
    scoreButton.setAttribute("aria-label", `Set score ${score}`);
    scoreButton.addEventListener("click", () => {
      void rateProjectFromChooser(project, score);
    });
    scoreButtons.appendChild(scoreButton);
  }
  scoreGroup.append(scoreLabel, scoreButtons);

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "danger";
  deleteButton.textContent = "Delete Project";
  deleteButton.addEventListener("click", () => {
    void deleteProjectFromChooser(project, card);
  });

  menu.append(renameButton, copyButton, commentButton, scoreGroup, deleteButton);
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  menu.style.left = `${Math.min(x, window.innerWidth - rect.width - 8)}px`;
  menu.style.top = `${Math.min(y, window.innerHeight - rect.height - 8)}px`;
  projectContextMenu = menu;
  window.setTimeout(() => {
    document.addEventListener("click", hideProjectContextMenu, { once: true });
  }, 0);
}

function createProjectPreviewElement(project) {
  if (project.preview_url) {
    const img = document.createElement("img");
    img.src = project.preview_url;
    img.alt = project.preview_label || project.project_name || project.project_id;
    img.loading = "lazy";
    img.decoding = "async";
    img.addEventListener("error", () => {
      const placeholder = document.createElement("div");
      placeholder.className = "project-card-preview-placeholder";
      placeholder.textContent = "Preview unavailable";
      img.replaceWith(placeholder);
    }, { once: true });
    return img;
  }
  const placeholder = document.createElement("div");
  placeholder.className = "project-card-preview-placeholder";
  placeholder.textContent = "No cutout";
  return placeholder;
}

function appendMissingModelBadge(container, project) {
  if (project.in_overleaf_catalog && !project.has_svi_result) {
    const badge = document.createElement("div");
    badge.className = "project-model-empty-badge";
    badge.textContent = "E";
    badge.title = "Overleaf catalogue target without a lens model result";
    container.appendChild(badge);
  }
}

function createProjectRatingChip(project) {
  const rating = normalizedProjectRating(project.rating);
  const ratingBadge = document.createElement("div");
  ratingBadge.className = "project-rating-badge";
  ratingBadge.textContent = `Score ${rating}`;
  ratingBadge.title = `Model result score: ${rating}/5`;
  return ratingBadge;
}

function createProjectActionButton(label, className, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

function createProjectActions(project, item) {
  const actions = document.createElement("div");
  actions.className = "project-card-actions";
  const copyButton = createProjectActionButton("Copy", "project-card-copy", (event) => {
    event.stopPropagation();
    void copyProjectFromChooser(project);
  });
  const commentButton = createProjectActionButton("Comment", "project-card-comment", (event) => {
    event.stopPropagation();
    void editProjectCommentFromChooser(project);
  });
  const ratingChip = createProjectRatingChip(project);
  const deleteButton = createProjectActionButton("Delete", "project-card-delete", (event) => {
    event.stopPropagation();
    void deleteProjectFromChooser(project, item);
  });
  actions.append(copyButton);
  actions.append(commentButton);
  if (ratingChip) actions.append(ratingChip);
  actions.append(deleteButton);
  return actions;
}

function projectCommentText(project) {
  return String(project.comment || "").trim();
}

function applyMissingModelProjectDefaults(project) {
  if (project.has_svi_result) return;
  const applyMaskInput = lensModelForm?.querySelector('input[name="apply_lensed_arc_mask"]');
  if (!applyMaskInput || applyMaskInput.checked) return;
  applyMaskInput.checked = true;
  markLensScriptStale("Apply mask enabled by default for project without a lens model");
  scheduleProjectSave();
}

function attachProjectChooserItemHandlers(item, project) {
  item.addEventListener("click", async (event) => {
    if (event.target.closest(".project-card-actions")) return;
    closeProjectChooser();
    await loadProjectById(project.project_id);
    applyMissingModelProjectDefaults(project);
  });
  item.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    closeProjectChooser();
    await loadProjectById(project.project_id);
    applyMissingModelProjectDefaults(project);
  });
  item.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    showProjectContextMenu(project, event.clientX, event.clientY, item);
  });
}

function createProjectCard(project) {
  const card = document.createElement("div");
  card.className = "project-card";
  card.dataset.projectId = project.project_id;
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.title = `${project.project_name || project.project_id}\n${project.project_folder || ""}`;

  const preview = document.createElement("div");
  preview.className = "project-card-preview";
  preview.appendChild(createProjectPreviewElement(project));
  appendMissingModelBadge(preview, project);
  card.appendChild(preview);

  const title = document.createElement("div");
  title.className = "project-card-title";
  title.textContent = project.project_name || project.project_id;
  const meta = document.createElement("div");
  meta.className = "project-card-meta";
  meta.textContent = project.preview_label || project.saved_at || project.project_id;
  const comment = document.createElement("div");
  comment.className = "project-card-comment-text";
  comment.textContent = projectCommentText(project) || "No comment";
  card.append(title, meta, comment, createProjectActions(project, card));
  attachProjectChooserItemHandlers(card, project);
  return card;
}

function createProjectListRow(project) {
  const row = document.createElement("div");
  row.className = "project-list-row";
  row.dataset.projectId = project.project_id;
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.title = `${project.project_name || project.project_id}\n${project.project_folder || ""}`;

  const preview = document.createElement("div");
  preview.className = "project-list-preview";
  preview.appendChild(createProjectPreviewElement(project));
  appendMissingModelBadge(preview, project);

  const body = document.createElement("div");
  body.className = "project-list-body";
  const title = document.createElement("div");
  title.className = "project-list-title";
  title.textContent = project.project_name || project.project_id;
  const meta = document.createElement("div");
  meta.className = "project-list-meta";
  meta.textContent = project.project_folder || project.preview_label || project.saved_at || project.project_id;
  const comment = document.createElement("div");
  comment.className = "project-list-comment";
  comment.textContent = projectCommentText(project) || "No comment";
  body.append(title, meta, comment);

  row.append(preview, body, createProjectActions(project, row));
  attachProjectChooserItemHandlers(row, project);
  return row;
}

function filteredProjectChooserProjects() {
  const query = projectChooserQuery.trim().toLowerCase();
  const minimumScore = projectChooserMinimumScore === "all" ? null : Number(projectChooserMinimumScore);
  const filtered = projectChooserProjects.filter((project) => {
    const rating = normalizedProjectRating(project.rating);
    if (Number.isFinite(minimumScore) && rating < minimumScore) return false;
    if (!query) return true;
    const searchable = [
      project.project_name,
      project.project_id,
      project.project_folder,
      project.comment,
      project.preview_label,
    ].join(" ").toLowerCase();
    return searchable.includes(query);
  });

  return filtered.sort((left, right) => {
    if (projectChooserSortMode === "name") {
      return projectDisplayName(left).localeCompare(projectDisplayName(right), undefined, { numeric: true });
    }
    if (projectChooserSortMode === "updated") {
      return String(right.saved_at || "").localeCompare(String(left.saved_at || ""));
    }
    const ratingDifference = normalizedProjectRating(right.rating) - normalizedProjectRating(left.rating);
    return ratingDifference || String(right.saved_at || "").localeCompare(String(left.saved_at || ""));
  });
}

function renderProjectChooser(projects, options = {}) {
  if (options.replace !== false) {
    projectChooserProjects = Array.isArray(projects) ? projects : [];
  }
  projectChooserGrid.innerHTML = "";
  projectChooserGrid.classList.toggle("list-mode", projectChooserView === "list");
  if (!projectChooserProjects.length) {
    projectChooserStatus.textContent = "No saved projects.";
    return;
  }
  const visibleProjects = filteredProjectChooserProjects();
  if (!visibleProjects.length) {
    projectChooserStatus.textContent = `No projects match the current filters (${projectChooserProjects.length} total).`;
    projectChooserGrid.innerHTML = '<div class="project-chooser-empty">No matching projects</div>';
    return;
  }
  projectChooserStatus.textContent = visibleProjects.length === projectChooserProjects.length
    ? `${visibleProjects.length} projects`
    : `${visibleProjects.length} of ${projectChooserProjects.length} projects`;
  for (const project of visibleProjects) {
    projectChooserGrid.appendChild(projectChooserView === "list" ? createProjectListRow(project) : createProjectCard(project));
  }
}

async function openProjectChooser() {
  projectChooserModal.hidden = false;
  try {
    await reloadProjectChooser();
    window.setTimeout(() => projectChooserSearch?.focus(), 0);
  } catch (error) {
    const message = /failed to fetch/i.test(error.message)
      ? "Backend unavailable. Start backend_server.py and open the matching host/port."
      : error.message;
    projectChooserStatus.textContent = `Project list failed: ${message}`;
    appendLog(`Project chooser: ${error.message}`);
  }
}

function setPsfMode(mode) {
  mode = mode === "auto" ? "auto" : "input";
  psfState.mode = mode;
  psfModeInputs.forEach((input) => {
    input.checked = input.value === mode;
  });
  psfModePanels.forEach((panel) => {
    panel.hidden = panel.dataset.psfModePanel !== mode;
  });
  psfCandidatesPanel.hidden = mode !== "auto";
  psfPreviewLayout.classList.toggle("input-mode", mode === "input");
  if (mode === "input" && psfState.previewView === "image") {
    setPsfPreviewView("product");
  }
  updatePsfOverlayVisibility();
}

function setPsfRunState(running) {
  psfState.jobRunning = Boolean(running);
  psfRunFit.disabled = psfState.jobRunning;
  psfStopFit.disabled = !psfState.jobRunning;
}

function setPsfMainPreview(data, options = {}) {
  if (!psfMainPreview || !data) return;
  const persist = options.persist !== false;
  if (persist) {
    psfState.mainPreviewData = data;
  }
  const imageUrl = data.preview_url || data.image_url || data.figure_url;
  const stage1Url = data.stage1_all_star_preview_url || data.svi_summary?.stage1_all_star_preview_url;
  if (imageUrl || stage1Url) {
    psfMainPreview.innerHTML = "";
    [
      { url: imageUrl, label: "PSF product" },
      { url: stage1Url, label: "PSF on selected stars" },
    ].forEach((item) => {
      if (!item.url) return;
      const figure = document.createElement("figure");
      figure.className = "psf-preview-figure";
      const caption = document.createElement("figcaption");
      caption.textContent = item.label;
      const img = document.createElement("img");
      img.src = item.url;
      img.alt = item.label;
      figure.appendChild(caption);
      figure.appendChild(img);
      psfMainPreview.appendChild(figure);
    });
    return;
  }
  psfMainPreview.textContent = data.message || "No PSF product loaded";
}

async function clearPsfOutput() {
  const deletePayload = {
    job_id: psfState.jobId,
    main_preview_data: psfState.mainPreviewData,
    job_status: psfState.jobStatus,
    detected_stars: psfState.detectedStars,
  };
  psfClearOutput.disabled = true;
  try {
    const data = await callBackend("/api/psf-fit/clear-output", deletePayload);
    appendLog(data.message || "PSF fit output deleted.");
    if (Array.isArray(data.skipped) && data.skipped.length) {
      appendLog(`PSF clear output skipped ${data.skipped.length} path(s).`);
    }
  } catch (error) {
    appendLog(`PSF clear output delete failed: ${error.message}`);
  } finally {
    psfClearOutput.disabled = false;
  }

  psfState.detectedStars = [];
  psfState.selectedIds = new Set();
  psfState.fullImagePreviewSrc = "";
  psfState.mainPreviewData = null;
  psfState.pollToken += 1;
  psfState.jobId = "";
  psfState.jobStatus = null;
  psfSelectedIds.value = "";
  psfMainPreview.innerHTML = "<span>No PSF product loaded</span>";
  psfFullImageImg.removeAttribute("src");
  psfFullImageFrame.hidden = true;
  psfFullImageEmpty.hidden = false;
  psfFullImageEmpty.textContent = "Run Find stars, then switch here to inspect the full image.";
  psfOverlay.innerHTML = "";
  psfFullImageOverlay.innerHTML = "";
  renderPsfCandidates([]);
  appendLog("PSF fit output cleared.");
  scheduleProjectSave();
}

async function setPsfPreviewView(view) {
  psfState.previewView = view;
  psfPreviewViewInputs.forEach((input) => {
    input.checked = input.value === view;
  });
  psfMainPreview.hidden = view !== "product";
  psfFullImagePreview.hidden = view !== "image";
  psfPreviewLayout.classList.toggle("full-image-mode", view === "image");
  updatePsfOverlayVisibility();
  if (view === "image") {
    await ensurePsfFullImagePreview();
    renderPsfFullImageBoxes();
  }
}

function parsePsfSelectedIds() {
  return psfSelectedIds.value
    .split(/[,\s;]+/)
    .map((value) => Number(value))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function syncPsfSelectedIdsFromInput() {
  psfState.selectedIds = new Set(parsePsfSelectedIds());
  syncPsfCandidateSelection();
}

function setPsfSelectedIds(ids) {
  psfState.selectedIds = new Set(ids);
  psfSelectedIds.value = [...psfState.selectedIds].join(", ");
  syncPsfCandidateSelection();
}

function syncPsfCandidateSelection() {
  psfCandidateList.querySelectorAll(".psf-candidate-card").forEach((card) => {
    card.classList.toggle("selected", psfState.selectedIds.has(Number(card.dataset.starId)));
  });
  psfOverlay.querySelectorAll(".psf-overlay-box").forEach((box) => {
    box.classList.toggle("selected", psfState.selectedIds.has(Number(box.dataset.starId)));
  });
  psfFullImageOverlay.querySelectorAll(".psf-full-image-box").forEach((box) => {
    box.classList.toggle("selected", psfState.selectedIds.has(Number(box.dataset.starId)));
  });
}

function renderPsfCandidates(stars) {
  psfCandidateList.innerHTML = "";
  psfCandidateCount.textContent = String(stars.length);
  if (!stars.length) {
    psfCandidateList.innerHTML = '<span class="empty-list">No candidates found.</span>';
    renderPsfOverlayBoxes();
    renderPsfFullImageBoxes();
    return;
  }

  for (const star of stars) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "psf-candidate-card";
    card.dataset.starId = String(star.id);
    card.innerHTML = `
      <img src="${star.preview_url}" alt="PSF candidate ${star.id}">
      <div class="psf-candidate-meta">
        <span>#${star.id}</span>
        <span>${Number(star.flux).toExponential(2)}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      const next = new Set(parsePsfSelectedIds());
      if (next.has(star.id)) {
        next.delete(star.id);
      } else {
        next.add(star.id);
      }
      setPsfSelectedIds(next);
    });
    psfCandidateList.appendChild(card);
  }
  syncPsfCandidateSelection();
  renderPsfOverlayBoxes();
  renderPsfFullImageBoxes();
}

function updatePsfOverlayTransform() {
  psfOverlay.style.width = `${imagePreviewCanvas.width}px`;
  psfOverlay.style.height = `${imagePreviewCanvas.height}px`;
  psfOverlay.style.transform = imagePreviewCanvas.style.transform;
}

function updatePsfOverlayVisibility() {
  const overlaySource = psfSciencePath.value.trim() || imageView.sourcePath;
  const sourceMatches = !overlaySource || overlaySource === imageView.sourcePath;
  const visible =
    psfState.mode === "auto" &&
    psfState.previewView === "image" &&
    imageView.original &&
    sourceMatches &&
    psfState.detectedStars.length > 0;
  psfOverlay.classList.toggle("visible", Boolean(visible));
}

function renderPsfOverlayBoxes() {
  psfOverlay.innerHTML = "";
  updatePsfOverlayTransform();
  if (!imageView.original || !psfState.detectedStars.length) {
    updatePsfOverlayVisibility();
    return;
  }

  const imageHeight = imagePreviewCanvas.height;
  const fallbackSize = Math.max(3, Number(psfKernelSize.value) || 31);
  for (const star of psfState.detectedStars) {
    const bounds = star.bounds || {};
    const width = Math.max(3, (bounds.x1 ?? 0) - (bounds.x0 ?? 0) || fallbackSize);
    const height = Math.max(3, (bounds.y1 ?? 0) - (bounds.y0 ?? 0) || fallbackSize);
    const xDisplay = Number(star.x);
    const yDisplay = imageHeight - 1 - Number(star.y);
    if (!Number.isFinite(xDisplay) || !Number.isFinite(yDisplay)) continue;

    const box = document.createElement("div");
    box.className = "psf-overlay-box";
    box.dataset.starId = String(star.id);
    box.style.left = `${xDisplay}px`;
    box.style.top = `${yDisplay}px`;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.innerHTML = `<span class="psf-overlay-label">${star.id}</span>`;
    psfOverlay.appendChild(box);
  }
  syncPsfCandidateSelection();
  updatePsfOverlayVisibility();
}

function setPsfFullImagePreview(src, sourcePath, options = {}) {
  if (!src) {
    psfState.fullImagePreviewSrc = "";
    psfState.fullImageSourcePath = "";
    psfState.fullImageShape = null;
    psfState.fullImageDisplayStride = 1;
    psfState.fullImageRenderSeq += 1;
    psfFullImageImg.removeAttribute("src");
    psfFullImageFrame.hidden = true;
    psfFullImageEmpty.hidden = false;
    psfFullImageEmpty.textContent = "Load a science image before using the full-image preview.";
    return;
  }

  psfState.fullImagePreviewSrc = src;
  psfState.fullImageSourcePath = sourcePath || psfState.fullImageSourcePath;
  if (Array.isArray(options.shape)) {
    psfState.fullImageShape = options.shape.map(Number).slice(0, 2);
  }
  if (Number.isFinite(Number(options.displayStride))) {
    psfState.fullImageDisplayStride = Math.max(1, Number(options.displayStride));
  }
  psfFullImageEmpty.hidden = true;
  psfFullImageFrame.hidden = false;
  void renderPsfFullImageMtfPreview();
}

async function renderPsfFullImageMtfPreview() {
  const src = psfState.fullImagePreviewSrc;
  if (!src || psfFullImagePreview.hidden) return;
  const seq = (psfState.fullImageRenderSeq += 1);
  const sourceImg = new Image();
  sourceImg.onload = () => {
    if (seq !== psfState.fullImageRenderSeq) return;
    const canvas = document.createElement("canvas");
    canvas.width = sourceImg.naturalWidth;
    canvas.height = sourceImg.naturalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(sourceImg, 0, 0);
    const raw = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const output = mtfImageData(raw, mtfParameters());
    ctx.putImageData(output, 0, 0);
    const renderedSrc = canvas.toDataURL("image/png");
    psfFullImageImg.onload = () => renderPsfFullImageBoxes();
    psfFullImageImg.src = renderedSrc;
    if (psfFullImageImg.complete) renderPsfFullImageBoxes();
  };
  sourceImg.onerror = () => {
    if (seq !== psfState.fullImageRenderSeq) return;
    psfFullImageEmpty.hidden = false;
    psfFullImageFrame.hidden = true;
    psfFullImageEmpty.textContent = "Full image preview: failed to render MTF preview.";
  };
  sourceImg.src = src;
}

async function ensurePsfFullImagePreview() {
  const sourcePath = psfSciencePath.value.trim() || imageView.sourcePath;
  if (!sourcePath) {
    setPsfFullImagePreview("", "");
    return;
  }

  if (imageView.sourcePath === sourcePath && imageView.previewSrc) {
    setPsfFullImagePreview(imageView.previewSrc, sourcePath, {
      shape: imageView.sourceShape || imageView.shape,
      displayStride: imageView.displayStride,
    });
    return;
  }

  if (psfState.fullImageSourcePath === sourcePath && psfState.fullImagePreviewSrc) {
    setPsfFullImagePreview(psfState.fullImagePreviewSrc, sourcePath, {
      shape: psfState.fullImageShape,
      displayStride: psfState.fullImageDisplayStride,
    });
    return;
  }

  psfFullImageFrame.hidden = true;
  psfFullImageEmpty.hidden = false;
  psfFullImageEmpty.textContent = "Loading full image preview...";
  try {
    const data = await callBackend("/api/image-preprocess/run", {
      path: sourcePath,
      project_id: projectState.id,
      project_folder: projectState.folder,
      register_artifacts: false,
      register_default_cutout: false,
      prefer_project_original: false,
    });
    setPsfFullImagePreview(data.preview_url, data.source_path || sourcePath, {
      shape: data.image_payload?.source_shape || data.image_shape || null,
      displayStride: data.image_payload?.display_stride || data.display_stride || 1,
    });
  } catch (error) {
    psfFullImageEmpty.textContent = `Full image preview: ${error.message}`;
    appendLog(`PSF fit: ${error.message}`);
  }
}

function renderPsfFullImageBoxes() {
  psfFullImageOverlay.innerHTML = "";
  if (!psfState.detectedStars.length || !psfFullImageImg.complete || !psfFullImageImg.naturalWidth) {
    return;
  }

  const fallbackStride = Math.max(1, Number(psfState.fullImageDisplayStride || 1));
  const sourceShape = Array.isArray(psfState.fullImageShape)
    ? psfState.fullImageShape
    : [psfFullImageImg.naturalHeight * fallbackStride, psfFullImageImg.naturalWidth * fallbackStride];
  const imageHeight = Math.max(1, Number(sourceShape[0]) || psfFullImageImg.naturalHeight);
  const imageWidth = Math.max(1, Number(sourceShape[1]) || psfFullImageImg.naturalWidth);
  const fallbackSize = Math.max(3, Number(psfKernelSize.value) || 31);
  for (const star of psfState.detectedStars) {
    const bounds = star.bounds || {};
    const width = Math.max(3, (bounds.x1 ?? 0) - (bounds.x0 ?? 0) || fallbackSize);
    const height = Math.max(3, (bounds.y1 ?? 0) - (bounds.y0 ?? 0) || fallbackSize);
    const xDisplay = Number(star.x);
    const yDisplay = imageHeight - 1 - Number(star.y);
    if (!Number.isFinite(xDisplay) || !Number.isFinite(yDisplay)) continue;

    const box = document.createElement("div");
    box.className = "psf-full-image-box";
    box.dataset.starId = String(star.id);
    box.style.left = `${(100 * xDisplay) / imageWidth}%`;
    box.style.top = `${(100 * yDisplay) / imageHeight}%`;
    box.style.width = `${(100 * width) / imageWidth}%`;
    box.style.height = `${(100 * height) / imageHeight}%`;
    box.innerHTML = `<span class="psf-overlay-label">${star.id}</span>`;
    psfFullImageOverlay.appendChild(box);
  }
  syncPsfCandidateSelection();
}

function clampZoomScale(scale, minScale, maxScale) {
  return Math.min(maxScale, Math.max(minScale, scale));
}

function zoomCanvasAtPointer({ event, canvas, hasImage, getScale, setScale, minScale, maxScale, applyTransform, nudgeView }) {
  if (!hasImage || !canvas) return false;
  const rectBefore = canvas.getBoundingClientRect();
  if (rectBefore.width <= 0 || rectBefore.height <= 0) return false;

  event.preventDefault();
  const anchor = {
    x: (event.clientX - rectBefore.left) / rectBefore.width,
    y: (event.clientY - rectBefore.top) / rectBefore.height,
  };
  const oldScale = Number(getScale());
  const factor = event.deltaY < 0 ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP;
  const newScale = clampZoomScale(oldScale * factor, minScale, maxScale);
  if (!Number.isFinite(newScale) || Math.abs(newScale - oldScale) < 1e-9) return true;

  setScale(newScale);
  applyTransform();

  const rectAfter = canvas.getBoundingClientRect();
  const targetX = rectAfter.left + anchor.x * rectAfter.width;
  const targetY = rectAfter.top + anchor.y * rectAfter.height;
  nudgeView(event.clientX - targetX, event.clientY - targetY);
  applyTransform();
  return true;
}

async function loadInputPsfPreview() {
  psfLoadInput.disabled = true;
  try {
    let data;
    const file = psfInputFile.files[0];
    if (file) {
      appendLog(`PSF fit: loading input PSF file ${file.name}`);
      data = await uploadPsfForPreview(file);
    } else {
      const path = psfInputPath.value.trim();
      if (!path) {
        appendLog("PSF fit: choose a PSF file or enter a PSF path first.");
        return;
      }
      appendLog(`PSF fit: loading input PSF path ${path}`);
      data = await callBackend("/api/psf-fit/input-preview", {
        psf_path: path,
        project_id: projectState.id,
        project_folder: projectState.folder,
      });
    }
    psfState.inputPath = data.source_path || psfInputPath.value.trim();
    if (data.source_path) {
      psfInputPath.value = data.source_path;
    }
    setPsfMainPreview(data);
    appendLog(data.message || "PSF fit: input PSF loaded.");
    scheduleProjectSave();
  } catch (error) {
    appendLog(`PSF fit: ${error.message}`);
  } finally {
    psfLoadInput.disabled = false;
  }
}

async function loadDefaultEuclidPsf() {
  psfDefaultEuclid.disabled = true;
  try {
    if (!projectState.folder) {
      await saveProjectNow();
    }
    appendLog("PSF fit: loading default Euclid PSF.");
    const data = await callBackend("/api/psf-fit/default-euclid", {
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    psfState.inputPath = data.source_path || "";
    psfInputPath.value = data.source_path || "";
    psfState.jobStatus = data;
    setPsfMode("input");
    setPsfMainPreview(data);
    appendLog(data.project_psf_path ? `Default Euclid PSF saved to project: ${data.project_psf_path}` : data.message);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`PSF fit: ${error.message}`);
  } finally {
    psfDefaultEuclid.disabled = false;
  }
}

async function loadDefaultEuclidNispPsf(band, button) {
  button.disabled = true;
  try {
    if (!projectState.folder) {
      await saveProjectNow();
    }
    appendLog(`PSF fit: loading default Euclid ${band}-band PSF.`);
    const data = await callBackend(`/api/psf-fit/default-euclid-${band.toLowerCase()}`, {
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    psfState.inputPath = data.source_path || "";
    psfInputPath.value = data.source_path || "";
    psfState.jobStatus = data;
    setPsfMode("input");
    setPsfMainPreview(data);
    appendLog(data.project_psf_path ? `Default Euclid ${band}-band PSF saved to project: ${data.project_psf_path}` : data.message);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`PSF fit: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function loadDefaultEuclidHPsf() {
  return loadDefaultEuclidNispPsf("H", psfDefaultEuclidH);
}

async function findPsfStars() {
  const imagePath = psfSciencePath.value.trim() || imageView.sourcePath;
  if (!imagePath) {
    appendLog("PSF fit: load a science image first, or enter its path.");
    return;
  }
  psfSciencePath.value = imagePath;
  psfFindStars.disabled = true;
  try {
    appendLog(`PSF fit: finding stars in ${imagePath}`);
    const data = await callBackend("/api/psf-fit/auto-stars", {
      image_path: imagePath,
      kernel_size: Number(psfKernelSize.value),
      fwhm: Number(psfFwhm.value),
      threshold_sigma: Number(psfThresholdSigma.value),
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    psfState.detectedStars = data.stars || [];
    psfState.fullImageSourcePath = data.source_path || imagePath;
    psfState.fullImagePreviewSrc = "";
    psfState.jobId = "";
    psfState.jobStatus = null;
    setPsfSelectedIds([]);
    renderPsfCandidates(psfState.detectedStars);
    if (psfState.previewView === "image") {
      await ensurePsfFullImagePreview();
    }
    appendLog(data.message || "PSF fit: star finder completed.");
    scheduleProjectSave();
  } catch (error) {
    appendLog(`PSF fit: ${error.message}`);
  } finally {
    psfFindStars.disabled = false;
  }
}

async function runPsfFit() {
  if (psfState.jobRunning) {
    appendLog(`PSF fit: job ${psfState.jobId || ""} is already running.`);
    return;
  }
  try {
    const mode = psfState.mode;
    let payload;
    if (mode === "auto") {
      const imagePath = psfSciencePath.value.trim() || imageView.sourcePath;
      if (!imagePath) {
        appendLog("PSF fit: load a science image first, or enter its path.");
        return;
      }
      const selectedIds = parsePsfSelectedIds();
      if (!selectedIds.length) {
        appendLog("PSF fit: select at least one PSF candidate ID.");
        return;
      }
      payload = {
        mode: "auto",
        image_path: imagePath,
        kernel_size: Number(psfKernelSize.value),
        fwhm: Number(psfFwhm.value),
        selected_ids: selectedIds,
        stars: psfState.detectedStars,
        svi_steps: Number(psfSviSteps.value),
        ss_factor: Number(psfSsFactor.value),
        bg_box: currentBgBoxPayload(),
        project_id: projectState.id,
        project_folder: projectState.folder,
      };
    } else {
      const psfPath = psfState.inputPath || psfInputPath.value.trim();
      if (!psfPath) {
        appendLog("PSF fit: load an input PSF first.");
        return;
      }
      payload = {
        mode: "input",
        psf_path: psfPath,
        project_id: projectState.id,
        project_folder: projectState.folder,
      };
    }
    setPsfRunState(true);
    appendLog("PSF fit: running PSF fit.");
    const data = await callBackend("/api/psf-fit/run", payload);
    setPsfMainPreview(data);
    psfState.jobId = data.job_id || "";
    psfState.jobStatus = data;
    appendLog(data.message || "PSF fit: completed.");
    if (data.job_id) {
      const pollToken = psfState.pollToken + 1;
      psfState.pollToken = pollToken;
      scheduleProjectSave();
      pollPsfJob(data.job_id, pollToken);
    } else {
      setPsfRunState(false);
    }
  } catch (error) {
    appendLog(`PSF fit: ${error.message}`);
    setPsfRunState(false);
  }
}

async function pollPsfJob(jobId, pollToken = psfState.pollToken) {
  for (let i = 0; i < 240; i += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    if (psfState.pollToken !== pollToken || psfState.jobId !== jobId) {
      return;
    }
    try {
      const data = await callBackend(`/api/psf-fit/job/${jobId}`);
      updatePsfLogProgress(jobId, data);
      if (data.preview_url) {
        setPsfMainPreview(data);
      }
      psfState.jobId = jobId;
      psfState.jobStatus = data;
      if (data.state === "completed" || data.state === "failed" || data.state === "stopped") {
        setPsfRunState(false);
        scheduleProjectSave();
        appendLog(`PSF job ${jobId}: ${data.state}.`);
        return;
      }
    } catch (error) {
      if (psfState.pollToken !== pollToken || psfState.jobId !== jobId) {
        return;
      }
      appendLog(`PSF job ${jobId}: ${error.message}`);
      setPsfRunState(false);
      return;
    }
  }
  appendLog(`PSF job ${jobId}: still running.`);
  setPsfRunState(false);
}

async function stopPsfFit() {
  const jobId = psfState.jobId;
  if (!jobId) {
    appendLog("PSF fit: no active job id to stop.");
    return;
  }
  psfStopFit.disabled = true;
  try {
    const data = await callBackend("/api/psf-fit/stop", { job_id: jobId });
    psfState.jobStatus = data;
    appendLog(data.message || `PSF job ${jobId}: stop requested.`);
    setPsfRunState(false);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`PSF job ${jobId}: stop failed: ${error.message}`);
    psfStopFit.disabled = false;
  }
}

function applyImageTransform() {
  imagePreviewCanvas.style.transform = `translate(calc(-50% + ${imageView.x}px), calc(-50% + ${imageView.y}px)) scale(${imageView.scale})`;
  updatePsfOverlayTransform();
  syncPanControls();
  updateCutoutMarker();
  updateBgBoxMarker();
  renderCanvasAxes(imagePreviewCanvas, currentImagePixelScaleArcsec());
}

function panLimits() {
  const stageRect = imageZoomStage.getBoundingClientRect();
  const width = imagePreviewCanvas.width * imageView.scale;
  const height = imagePreviewCanvas.height * imageView.scale;
  return {
    x: Math.max(250, Math.ceil((width + stageRect.width) / 2)),
    y: Math.max(250, Math.ceil((height + stageRect.height) / 2)),
  };
}

function syncPanControls() {
  const hasImage = Boolean(imageView.original);
  imagePanX.classList.toggle("disabled", !hasImage);
  imagePanY.classList.toggle("disabled", !hasImage);
  if (!hasImage) return;

  const limits = panLimits();
  imagePanX.setAttribute("aria-valuemin", String(-limits.x));
  imagePanX.setAttribute("aria-valuemax", String(limits.x));
  imagePanX.setAttribute("aria-valuenow", String(Math.round(imageView.x)));
  imagePanY.setAttribute("aria-valuemin", String(-limits.y));
  imagePanY.setAttribute("aria-valuemax", String(limits.y));
  imagePanY.setAttribute("aria-valuenow", String(Math.round(imageView.y)));

  const xPct = 100 * (limits.x - imageView.x) / (2 * limits.x);
  const yPct = 100 * (limits.y - imageView.y) / (2 * limits.y);
  imagePanX.querySelector(".pan-thumb").style.left = `${Math.min(100, Math.max(0, xPct))}%`;
  imagePanY.querySelector(".pan-thumb").style.top = `${Math.min(100, Math.max(0, yPct))}%`;
}

function setPanFromPointer(event, axis) {
  if (!imageView.original) return;
  const limits = panLimits();
  const control = axis === "x" ? imagePanX : imagePanY;
  const rect = control.getBoundingClientRect();
  if (axis === "x") {
    const pct = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    imageView.x = Math.round((1 - pct * 2) * limits.x);
  } else {
    const pct = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    imageView.y = Math.round((1 - pct * 2) * limits.y);
  }
  applyImageTransform();
}

function attachPanControl(control, axis) {
  control.addEventListener("pointerdown", (event) => {
    if (!imageView.original) return;
    event.preventDefault();
    event.stopPropagation();
    imageView.panDragAxis = axis;
    control.classList.add("active");
    control.setPointerCapture(event.pointerId);
    setPanFromPointer(event, axis);
  });

  control.addEventListener("pointermove", (event) => {
    if (imageView.panDragAxis !== axis) return;
    event.preventDefault();
    event.stopPropagation();
    setPanFromPointer(event, axis);
  });

  control.addEventListener("pointerup", (event) => {
    if (imageView.panDragAxis !== axis) return;
    event.stopPropagation();
    imageView.panDragAxis = null;
    control.classList.remove("active");
    control.releasePointerCapture(event.pointerId);
  });

  control.addEventListener("pointercancel", () => {
    imageView.panDragAxis = null;
    control.classList.remove("active");
  });
}

function resetImageTransform() {
  imageView.scale = fittedImageScale();
  centerImageOnCentroid();
}

function imagePreviewStageHasLayout() {
  const stageRect = imageZoomStage.getBoundingClientRect();
  return stageRect.width > 1 && stageRect.height > 1;
}

function centerImagePreviewAfterLayout() {
  if (!imageView.original) return;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (!imageView.original) return;
      if (!imagePreviewStageHasLayout()) {
        imageView.needsLayoutCenter = true;
        return;
      }
      resetImageTransform();
      if (!cutoutPreviewPanel.hidden) {
        renderCutoutPreviewFromCanvas();
      }
    });
  });
}

function imageBrightnessCentroid(imageData) {
  if (!imageData) return null;
  const { data, width, height } = imageData;
  const pixelCount = width * height;
  const stride = Math.max(1, Math.floor(Math.sqrt(pixelCount / 250000)));
  const samples = [];
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const offset = 4 * (y * width + x);
      samples.push(0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2]);
    }
  }
  if (!samples.length) return { x: width / 2, y: height / 2 };
  samples.sort((a, b) => a - b);
  const background = samples[Math.floor(0.5 * (samples.length - 1))] || 0;
  let weightSum = 0;
  let xSum = 0;
  let ySum = 0;
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const offset = 4 * (y * width + x);
      const value = 0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2];
      const weight = Math.max(0, value - background);
      if (weight <= 0) continue;
      weightSum += weight;
      xSum += weight * (x + 0.5);
      ySum += weight * (y + 0.5);
    }
  }
  if (weightSum <= 0) return { x: width / 2, y: height / 2 };
  return { x: xSum / weightSum, y: ySum / weightSum };
}

function imageRowsBrightnessCentroid(rows, width, height) {
  if (!rows?.length || !width || !height) return null;
  const pixelCount = width * height;
  const stride = Math.max(1, Math.floor(Math.sqrt(pixelCount / 250000)));
  const samples = [];
  for (let y = 0; y < height; y += stride) {
    const row = rows[y] || [];
    for (let x = 0; x < width; x += stride) {
      const value = Number(row[x]);
      if (Number.isFinite(value)) samples.push(value);
    }
  }
  if (!samples.length) return { x: width / 2, y: height / 2 };
  samples.sort((a, b) => a - b);
  const background = samples[Math.floor(0.5 * (samples.length - 1))] || 0;
  let weightSum = 0;
  let xSum = 0;
  let ySum = 0;
  for (let y = 0; y < height; y += stride) {
    const row = rows[y] || [];
    for (let x = 0; x < width; x += stride) {
      const value = Number(row[x]);
      if (!Number.isFinite(value)) continue;
      const weight = Math.max(0, value - background);
      if (weight <= 0) continue;
      weightSum += weight;
      xSum += weight * (x + 0.5);
      ySum += weight * (y + 0.5);
    }
  }
  if (weightSum <= 0) return { x: width / 2, y: height / 2 };
  return { x: xSum / weightSum, y: ySum / weightSum };
}

function centerImageOnCentroid() {
  if (!imageView.original) return;
  if (!imagePreviewStageHasLayout()) {
    imageView.x = 0;
    imageView.y = 0;
    applyImageTransform();
    imageView.needsLayoutCenter = true;
    return;
  }
  const point = imageView.centroid || {
    x: imagePreviewCanvas.width / 2,
    y: imagePreviewCanvas.height / 2,
  };
  imageView.x = 0;
  imageView.y = 0;
  applyImageTransform();
  const rect = imagePreviewCanvas.getBoundingClientRect();
  const stageRect = imageZoomStage.getBoundingClientRect();
  const pointLeft = rect.left - stageRect.left + (point.x / imagePreviewCanvas.width) * rect.width;
  const pointTop = rect.top - stageRect.top + (point.y / imagePreviewCanvas.height) * rect.height;
  imageView.x += stageRect.width / 2 - pointLeft;
  imageView.y += stageRect.height / 2 - pointTop;
  applyImageTransform();
}

function fittedImageScale() {
  if (!imageView.original) return 1;
  const stageRect = imageZoomStage.getBoundingClientRect();
  const width = imagePreviewCanvas.width || imageView.original.width || 1;
  const height = imagePreviewCanvas.height || imageView.original.height || 1;
  const stageWidth = stageRect.width || imageZoomStage.clientWidth || Math.max(320, window.innerWidth - 360);
  const stageHeight = stageRect.height || imageZoomStage.clientHeight || 620;
  const availableWidth = Math.max(1, stageWidth - 36);
  const availableHeight = Math.max(1, stageHeight - 36);
  const scale = Math.min(availableWidth / width, availableHeight / height);
  return Math.max(0.05, scale);
}

function setImagePreview(src, options = {}) {
  const resetArtifacts = options.resetArtifacts !== false;
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = img.naturalWidth;
    sourceCanvas.height = img.naturalHeight;
    const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    sourceCtx.drawImage(img, 0, 0);
    imageView.original = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    imageView.rawPayload = null;
    imageView.centroid = imageBrightnessCentroid(imageView.original);
    imageView.previewSrc = src;
    imageView.shape = [sourceCanvas.height, sourceCanvas.width];
    imageView.sourceShape = [sourceCanvas.height, sourceCanvas.width];
    imageView.displayStride = 1;
    updateLensSigmaMaxAutoHint();

    imagePreviewCanvas.width = sourceCanvas.width;
    imagePreviewCanvas.height = sourceCanvas.height;
    psfOverlay.style.width = `${sourceCanvas.width}px`;
    psfOverlay.style.height = `${sourceCanvas.height}px`;
    imagePreviewCanvas.style.imageRendering = "pixelated";
    imagePreviewCanvas.style.display = "block";
    imagePreviewEmpty.style.display = "none";
    if (options.autoMtf !== false) {
      applySharedAutoMidtones(autoMidtonesFromImageData(imageView.original, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET), { resetBounds: true });
    }
    if (resetArtifacts) {
      cutoutPreviewPanel.hidden = true;
      cutoutState.fitsPath = "";
      cutoutState.previewUrl = "";
      cutoutState.shape = null;
      cutoutState.bounds = null;
      psfState.detectedStars = [];
      psfState.mainPreviewData = null;
      psfState.jobId = "";
      psfState.jobStatus = null;
      setPsfSelectedIds([]);
      renderPsfCandidates(psfState.detectedStars);
    }
    if (psfState.fullImageSourcePath === imageView.sourcePath || !psfState.fullImageSourcePath) {
      setPsfFullImagePreview(src, imageView.sourcePath, {
        shape: imageView.sourceShape,
        displayStride: imageView.displayStride,
      });
    }
    if (resetArtifacts) {
      clearCutoutCenter();
      imageView.bgBoxCenter = null;
      updateBgBoxMarker();
    }
    renderMtfPreview();
    resetImageTransform();
    resolve();
    };
    img.onerror = () => {
    appendLog("Image preprocess: failed to load preview image.");
      reject(new Error("Failed to load preview image."));
    };
    img.src = src;
  });
}

function setImagePreviewFromPayload(image, src = "", options = {}) {
  const resetArtifacts = options.resetArtifacts !== false;
  const height = Number(image?.shape?.[0] || image?.data?.length || 0);
  const width = Number(image?.shape?.[1] || image?.data?.[0]?.length || 0);
  if (!image?.data?.length || !width || !height) return false;

  imageView.rawPayload = image;
  imageView.original = new ImageData(width, height);
  imageView.centroid = imageRowsBrightnessCentroid(image.data, width, height);
  imageView.previewSrc = src || "";
  imageView.shape = [height, width];
  imageView.sourceShape = Array.isArray(image.source_shape) ? image.source_shape.map(Number).slice(0, 2) : [height, width];
  imageView.displayStride = Math.max(1, Number(image.display_stride || 1));
  const payloadPixelScale = Number(image.pixel_scale_arcsec);
  if (Number.isFinite(payloadPixelScale) && payloadPixelScale > 0) {
    imageView.pixelScaleArcsec = payloadPixelScale;
  }
  updateLensSigmaMaxAutoHint();

  imagePreviewCanvas.width = width;
  imagePreviewCanvas.height = height;
  psfOverlay.style.width = `${width}px`;
  psfOverlay.style.height = `${height}px`;
  imagePreviewCanvas.style.imageRendering = "pixelated";
  imagePreviewCanvas.style.display = "block";
  imagePreviewEmpty.style.display = "none";
  if (options.autoMtf !== false) {
    applySharedAutoMidtones(autoMidtonesFromRows(image.data, width, height, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET), { resetBounds: true });
  }
  if (resetArtifacts) {
    cutoutPreviewPanel.hidden = true;
    cutoutState.fitsPath = "";
    cutoutState.previewUrl = "";
    cutoutState.shape = null;
    cutoutState.bounds = null;
    psfState.detectedStars = [];
    psfState.mainPreviewData = null;
    psfState.jobId = "";
    psfState.jobStatus = null;
    setPsfSelectedIds([]);
    renderPsfCandidates(psfState.detectedStars);
    clearCutoutCenter();
    imageView.bgBoxCenter = null;
    updateBgBoxMarker();
  }
  if (psfState.fullImageSourcePath === imageView.sourcePath || !psfState.fullImageSourcePath) {
    setPsfFullImagePreview(src || imageView.previewSrc, imageView.sourcePath, {
      shape: imageView.sourceShape,
      displayStride: imageView.displayStride,
    });
  }
  renderMtfPreview();
  resetImageTransform();
  return true;
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

function imageDisplayStride() {
  return Math.max(1, Number(imageView.displayStride || 1));
}

function imageSourceHeight() {
  return Math.max(
    1,
    Number(imageView.sourceShape?.[0]) || imagePreviewCanvas.height * imageDisplayStride() || imagePreviewCanvas.height || 1,
  );
}

function imageDisplayYOffset() {
  const stride = imageDisplayStride();
  return Math.max(0, imageSourceHeight() - 1 - (imagePreviewCanvas.height - 1) * stride);
}

function sourceDisplayPointFromCanvas(point) {
  const stride = imageDisplayStride();
  return {
    x: Number(point?.x || 0) * stride,
    y: imageDisplayYOffset() + Number(point?.y || 0) * stride,
  };
}

function canvasPointFromSourceDisplay(point) {
  const stride = imageDisplayStride();
  return {
    x: Number(point?.x || 0) / stride,
    y: (Number(point?.y || 0) - imageDisplayYOffset()) / stride,
  };
}

function sourcePixelsToCanvasPixels(value) {
  return Number(value || 0) / imageDisplayStride();
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
  const displaySide = Math.max(1, sourcePixelsToCanvasPixels(Number(cutoutSize.value) || 0));
  const side = Math.max(4, displaySide * (rect.width / imagePreviewCanvas.width));
  cutoutMarker.style.left = `${left}px`;
  cutoutMarker.style.top = `${top}px`;
  cutoutMarker.style.width = `${side}px`;
  cutoutMarker.style.height = `${side}px`;
  cutoutMarker.style.display = "block";
  const sourcePoint = sourceDisplayPointFromCanvas(imageView.cutoutCenter);
  cutoutCenterLabel.textContent = `x=${Math.round(sourcePoint.x)}, y=${Math.round(sourcePoint.y)}, s=${Math.round(Number(cutoutSize.value) || 0)}`;
}

function setCutoutCenterFromEvent(event) {
  const point = canvasPixelFromEvent(event);
  if (!point) return;
  imageView.cutoutCenter = point;
  updateCutoutMarker();
  const sourcePoint = sourceDisplayPointFromCanvas(point);
  appendLog(`Cutout center set: x=${Math.round(sourcePoint.x)}, y=${Math.round(sourcePoint.y)}.`);
}

function clearCutoutCenter() {
  imageView.cutoutCenter = null;
  updateCutoutMarker();
}

function updateBgBoxMarker() {
  if (!imageView.bgBoxCenter) {
    bgBoxMarker.style.display = "none";
    bgBoxLabel.textContent = "Right click image";
    return;
  }
  const rect = imagePreviewCanvas.getBoundingClientRect();
  const stageRect = imageZoomStage.getBoundingClientRect();
  const left = rect.left - stageRect.left + (imageView.bgBoxCenter.x / imagePreviewCanvas.width) * rect.width;
  const top = rect.top - stageRect.top + (imageView.bgBoxCenter.y / imagePreviewCanvas.height) * rect.height;
  const side = Math.max(4, Number(bgBoxSize.value) || 0) * (rect.width / imagePreviewCanvas.width);
  bgBoxMarker.style.left = `${left}px`;
  bgBoxMarker.style.top = `${top}px`;
  bgBoxMarker.style.width = `${side}px`;
  bgBoxMarker.style.height = `${side}px`;
  bgBoxMarker.style.display = "block";
  bgBoxLabel.textContent = `x=${Math.round(imageView.bgBoxCenter.x)}, y=${Math.round(imageView.bgBoxCenter.y)}, s=${Math.round(Number(bgBoxSize.value) || 0)}`;
}

function setBgBoxFromEvent(event) {
  const point = canvasPixelFromEvent(event);
  if (!point) return;
  imageView.bgBoxCenter = point;
  updateBgBoxMarker();
  appendLog(`Background box set: x=${Math.round(point.x)}, y=${Math.round(point.y)}, s=${Math.round(Number(bgBoxSize.value) || 0)}.`);
  scheduleProjectSave();
}

function clearBackgroundBox() {
  imageView.bgBoxCenter = null;
  updateBgBoxMarker();
  appendLog("Background box cleared.");
  scheduleProjectSave();
}

function currentBgBoxPayload() {
  if (!imageView.bgBoxCenter) return null;
  return {
    x: Math.round(imageView.bgBoxCenter.x),
    y: Math.round(imageView.bgBoxCenter.y),
    size: Math.max(4, Math.round(Number(bgBoxSize.value) || 4)),
  };
}

function setMaskTool(tool) {
  cancelMaskInteraction();
  maskState.tool = tool;
  if (maskState.conjugateMode) {
    setConjugateMode(false);
  }
  maskToolInputs.forEach((input) => {
    input.checked = input.value === tool;
  });
  maskState.lastPoint = null;
  maskState.polygonPoints = [];
  updateMaskCursor();
  renderMaskOverlay();
  scheduleProjectSave();
}

function setMaskMode(mode) {
  cancelMaskInteraction();
  maskState.mode = mode;
  if (maskState.conjugateMode) {
    setConjugateMode(false);
  }
  maskModeInputs.forEach((input) => {
    input.checked = input.value === mode;
  });
  maskState.polygonPoints = [];
  updateMaskCursor();
  renderMaskOverlay();
  scheduleProjectSave();
}

function checkedMaskTool() {
  const checked = Array.from(maskToolInputs).find((input) => input.checked);
  const value = checked?.value || maskState.tool;
  maskState.tool = value === "brush" ? "brush" : "line";
  return maskState.tool;
}

function checkedMaskMode() {
  const checked = Array.from(maskModeInputs).find((input) => input.checked);
  const value = checked?.value || maskState.mode;
  maskState.mode = value === "subtract" ? "subtract" : "add";
  return maskState.mode;
}

function cancelMaskInteraction() {
  maskState.drawing = false;
  maskState.lastPoint = null;
  maskState.polygonPoints = [];
}

function normalizeMaskData() {
  if (!maskState.data) return;
  for (let i = 0; i < maskState.data.length; i += 1) {
    maskState.data[i] = maskState.data[i] > 0 ? 1 : 0;
  }
}

function maskPixelCount() {
  if (!maskState.data) return 0;
  let count = 0;
  for (let i = 0; i < maskState.data.length; i += 1) {
    if (maskState.data[i] > 0) count += 1;
  }
  return count;
}

function maskRowsForSave() {
  normalizeMaskData();
  const rows = [];
  for (let y = 0; y < maskState.height; y += 1) {
    const row = [];
    const offset = y * maskState.width;
    for (let x = 0; x < maskState.width; x += 1) {
      row.push(maskState.data[offset + x] > 0 ? 1 : 0);
    }
    rows.push(row);
  }
  return rows;
}

function setMaskType(type) {
  maskState.type = ["mask_1", "mask_2", "mask_out"].includes(type) ? type : "mask_1";
  maskType.value = maskState.type;
  if (maskState.conjugateMode) {
    setConjugateMode(false);
  }
  maskState.lastPoint = null;
  maskState.polygonPoints = [];
  scheduleProjectSave();
}

function setConjugateMode(enabled) {
  maskState.conjugateMode = Boolean(enabled);
  maskConjugatePoint.classList.toggle("active", maskState.conjugateMode);
  const label = maskState.conjugatePlane === "source2" ? "Plane 2" : "Plane 1";
  maskConjugatePoint.textContent = maskState.conjugateMode ? `${label} conj on` : "Conjugate point";
  updateMaskCursor();
  scheduleProjectSave();
}

function setConjugatePlane(plane) {
  maskState.conjugatePlane = plane === "source2" ? "source2" : "source1";
  if (maskConjugatePlane) maskConjugatePlane.value = maskState.conjugatePlane;
  if (maskState.conjugateMode) setConjugateMode(true);
  updateConjugateStatus();
}

function setConjugateMeasureSource(source, options = {}) {
  maskState.conjugateMeasureSource = source === "subtracted" ? "subtracted" : "mask";
  if (maskConjugateSource) maskConjugateSource.value = maskState.conjugateMeasureSource;
  updateMaskCursor();
  if (options.save !== false) scheduleProjectSave();
}

function setConjugateClickMode(mode) {
  const normalized = ["click", "gaussian", "brightest"].includes(mode) ? mode : "brightest";
  maskState.conjugateClickMode = normalized;
  if (maskConjugateClickMode) maskConjugateClickMode.value = normalized;
  scheduleProjectSave();
}

function conjugateClickModeLabel(mode = maskState.conjugateClickMode) {
  if (mode === "click") return "click";
  if (mode === "gaussian") return "Gaussian fit";
  return "brightest pixel";
}

function activeConjugatePoints() {
  return maskState.conjugatePlane === "source2" ? maskState.conjugatePointsSource2 : maskState.conjugatePoints;
}

function updateConjugateStatus() {
  const count1 = maskState.conjugatePoints.length;
  const count2 = maskState.conjugatePointsSource2.length;
  maskConjugateStatus.textContent = `P1 ${count1}, P2 ${count2}`;
  renderConjugateLayer();
}

function clearConjugatePoints() {
  const count1 = maskState.conjugatePoints.length;
  const count2 = maskState.conjugatePointsSource2.length;
  maskState.conjugatePoints = [];
  maskState.conjugatePointsSource2 = [];
  updateConjugateStatus();
  renderMaskOverlay();
  if (lensLightSubtractedAvailable()) {
    scheduleMaskSubtractionMatplotlibPreview(20);
  }
  appendLog(`Mask: cleared conjugate points (P1 ${count1}, P2 ${count2}).`);
  syncLensScriptConjugatesFromMask();
  scheduleLensScriptCutoutPreviewRefresh(20);
  scheduleProjectSave();
}

function updateMaskCursor() {
  const radius = Math.max(0, Math.min(20, Math.round(Number(maskBrushRadius.value) || 0)));
  maskBrushRadiusValue.textContent = `${radius} px`;
  if (maskSubtractionPreview) {
    maskSubtractionPreview.style.cursor =
      maskState.conjugateMode && activeMaskConjugateSource() === "subtracted" ? "crosshair" : "default";
  }
  if (maskState.conjugateMode) {
    maskOverlayCanvas.style.cursor = "crosshair";
    return;
  }
  if (maskState.tool !== "brush") {
    maskOverlayCanvas.style.cursor = "crosshair";
    return;
  }
  const rect = maskOverlayCanvas.getBoundingClientRect();
  const cssScale = maskState.width && rect.width ? rect.width / maskState.width : 1;
  const cursorRadius = Math.max(1, radius * cssScale);
  const size = Math.ceil(cursorRadius * 2 + 6);
  const center = size / 2;
  const color = maskState.mode === "add" ? "#31d0aa" : "#ff7285";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><circle cx="${center}" cy="${center}" r="${cursorRadius}" fill="none" stroke="${color}" stroke-width="2"/></svg>`;
  maskOverlayCanvas.style.cursor = `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${center} ${center}, crosshair`;
}

function maskPointFromEvent(event) {
  if (!maskState.data) return null;
  const rect = maskOverlayCanvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const x = ((event.clientX - rect.left) / rect.width) * maskState.width;
  const y = ((event.clientY - rect.top) / rect.height) * maskState.height;
  if (x < 0 || y < 0 || x >= maskState.width || y >= maskState.height) return null;
  return { x, y };
}

function displayedCanvasPointFromEvent(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0 || canvas.width <= 0 || canvas.height <= 0) return null;
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
  return { x, y };
}

function nativeMaskPointFromCanvasPoint(point, canvas) {
  if (!point || !canvas || !maskState.width || !maskState.height || !canvas.width || !canvas.height) return null;
  return {
    x: (point.x / canvas.width) * maskState.width,
    y: (point.y / canvas.height) * maskState.height,
  };
}

function fitMaskSubtractionCanvasToStage() {
  if (!maskSubtractionPreview || maskSubtractionPreview.hidden || !maskSubtractionPreview.width || !maskSubtractionPreview.height) return;
  maskSubtractionPreview.style.width = "100%";
  maskSubtractionPreview.style.height = "auto";
}

function applyMaskBrush(point) {
  if (!maskState.data) return;
  const radius = Math.max(0, Math.min(20, Math.round(Number(maskBrushRadius.value) || 0)));
  const value = checkedMaskMode() === "add" ? 1 : 0;
  const cx = Math.round(point.x);
  const cy = Math.round(point.y);
  const r2 = radius * radius;
  const x0 = Math.max(0, cx - radius);
  const x1 = Math.min(maskState.width - 1, cx + radius);
  const y0 = Math.max(0, cy - radius);
  const y1 = Math.min(maskState.height - 1, cy + radius);
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        maskState.data[y * maskState.width + x] = value;
      }
    }
  }
}

function drawMaskLine(fromPoint, toPoint) {
  const dx = toPoint.x - fromPoint.x;
  const dy = toPoint.y - fromPoint.y;
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    applyMaskBrush({
      x: fromPoint.x + t * dx,
      y: fromPoint.y + t * dy,
    });
  }
}

function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i].x;
    const yi = points[i].y;
    const xj = points[j].x;
    const yj = points[j].y;
    const crosses = yi > y !== yj > y;
    if (crosses) {
      const xCross = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (x < xCross) inside = !inside;
    }
  }
  return inside;
}

function fillMaskPolygon(points) {
  if (!maskState.data || points.length < 3) return;
  const value = checkedMaskMode() === "add" ? 1 : 0;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const x1 = Math.min(maskState.width - 1, Math.ceil(Math.max(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const y1 = Math.min(maskState.height - 1, Math.ceil(Math.max(...ys)));
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      if (pointInPolygon(x + 0.5, y + 0.5, points)) {
        maskState.data[y * maskState.width + x] = value;
      }
    }
  }
}

function addMaskPolygonPoint(point) {
  if (!maskState.data) return;
  maskState.polygonPoints.push({ x: point.x, y: point.y });
  if (maskState.polygonPoints.length >= 3) {
    fillMaskPolygon(maskState.polygonPoints);
  }
}

function drawMaskPolygonGuide(ctx) {
  if (!maskState.polygonPoints.length) return;
  const color = maskState.mode === "add" ? "#31d0aa" : "#ff7285";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  maskState.polygonPoints.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  if (maskState.polygonPoints.length >= 3) ctx.closePath();
  ctx.stroke();
  for (const point of maskState.polygonPoints) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}

function conjugateMarkerLocalPosition(point) {
  const arcsec = point?.arcsec || point?.center_arcsec;
  const scale = currentImagePixelScaleArcsec();
  const arcsecX = Number(arcsec?.x);
  const arcsecY = Number(arcsec?.y);
  if (Number.isFinite(arcsecX) && Number.isFinite(arcsecY) && Number.isFinite(scale) && scale > 0) {
    return {
      x: Math.max(0, Math.min(maskState.width, maskState.width / 2 + arcsecX / scale)),
      y: Math.max(0, Math.min(maskState.height, maskState.height / 2 - arcsecY / scale)),
    };
  }
  const local = point.local || point.local_display;
  if (!local) return null;
  return {
    x: Math.max(0, Math.min(maskState.width, Number(local.x))),
    y: Math.max(0, Math.min(maskState.height, Number(local.y))),
  };
}

function renderConjugateLayer() {
  maskConjugateLayer.innerHTML = "";
  if (maskState.useMatplotlibPreview) {
    maskConjugateLayer.style.display = "none";
    return;
  }
  const groups = [
    { plane: "source1", label: "S1", points: maskState.conjugatePoints },
    { plane: "source2", label: "S2", points: maskState.conjugatePointsSource2 },
  ];
  const totalPoints = groups.reduce((sum, group) => sum + group.points.length, 0);
  if (!maskState.data || !totalPoints) {
    maskConjugateLayer.style.display = "none";
    return;
  }
  const displayWidth = parseFloat(maskOverlayCanvas.style.width) || maskOverlayCanvas.getBoundingClientRect().width;
  const displayHeight = parseFloat(maskOverlayCanvas.style.height) || maskOverlayCanvas.getBoundingClientRect().height;
  if (!displayWidth || !displayHeight || !maskState.width || !maskState.height) {
    maskConjugateLayer.style.display = "none";
    return;
  }
  maskConjugateLayer.style.display = "block";
  maskConjugateLayer.style.width = `${displayWidth}px`;
  maskConjugateLayer.style.height = `${displayHeight}px`;
  groups.forEach((group) => {
    group.points.forEach((point, index) => {
      const local = conjugateMarkerLocalPosition(point);
      if (!local) return;
      const marker = document.createElement("div");
      marker.className = `conjugate-marker ${group.plane}`;
      marker.style.left = `${(local.x / maskState.width) * displayWidth}px`;
      marker.style.top = `${(local.y / maskState.height) * displayHeight}px`;
      marker.dataset.label = `${group.label}-${index + 1}`;
      marker.title = `${group.label} conjugate point ${index + 1}`;
      maskConjugateLayer.appendChild(marker);
    });
  });
}

function renderMaskOverlay() {
  if (!maskState.data) {
    maskImageCanvas.style.display = "none";
    maskOverlayCanvas.style.display = "none";
    maskConjugateLayer.style.display = "none";
    maskEmpty.style.display = "grid";
    removePreviewAxes(maskImageCanvas);
    return;
  }
  normalizeMaskData();
  maskEmpty.style.display = "none";
  maskImageCanvas.style.display = "block";
  maskOverlayCanvas.style.display = "block";

  const ctx = maskOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, maskState.width, maskState.height);
  drawMaskContour(ctx);
  drawMaskPolygonGuide(ctx);
  const nMask = maskPixelCount();
  maskStatus.textContent = `${maskState.type}: ${maskState.width}x${maskState.height}, mask pixels ${nMask}`;
  updateConjugateStatus();
  if (maskState.useMatplotlibPreview) {
    scheduleMaskMatplotlibPreview(maskState.drawing ? 300 : 80);
    if (lensLightSubtractedAvailable()) {
      scheduleMaskSubtractionMatplotlibPreview(maskState.drawing ? 300 : 100);
    }
  }
}

function maskContourColor() {
  if (maskState.type === "mask_2") return "#ff3300";
  if (maskState.type === "mask_out") return "#ff4f5f";
  return "#9933ff";
}

function drawMaskContour(ctx) {
  if (!maskState.data || !maskState.width || !maskState.height) return;
  const valueAt = (x, y) =>
    x >= 0 && y >= 0 && x < maskState.width && y < maskState.height && maskState.data[y * maskState.width + x] > 0;

  ctx.save();
  ctx.strokeStyle = maskContourColor();
  ctx.globalAlpha = 0.95;
  ctx.lineWidth = 0.6;
  ctx.setLineDash([2, 1.5]);
  ctx.beginPath();
  for (let y = 0; y < maskState.height; y += 1) {
    for (let x = 0; x < maskState.width - 1; x += 1) {
      if (valueAt(x, y) !== valueAt(x + 1, y)) {
        ctx.moveTo(x + 1, y);
        ctx.lineTo(x + 1, y + 1);
      }
    }
  }
  for (let y = 0; y < maskState.height - 1; y += 1) {
    for (let x = 0; x < maskState.width; x += 1) {
      if (valueAt(x, y) !== valueAt(x, y + 1)) {
        ctx.moveTo(x, y + 1);
        ctx.lineTo(x + 1, y + 1);
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}

function fitMaskCanvasToStage() {
  if (!maskState.width || !maskState.height) return;
  const stage = document.querySelector(".mask-stage");
  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width <= 0 || stageRect.height <= 0) return;
  const style = window.getComputedStyle(stage);
  const paddingX = Number.parseFloat(style.paddingLeft) + Number.parseFloat(style.paddingRight);
  const paddingY = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
  const availableWidth = Math.max(1, stageRect.width - paddingX - 2);
  const availableHeight = Math.max(1, stageRect.height - paddingY - 2);
  const scale = Math.min(availableWidth / maskState.width, availableHeight / maskState.height);
  const displayWidth = Math.max(1, Math.floor(maskState.width * scale));
  const displayHeight = Math.max(1, Math.floor(maskState.height * scale));
  for (const canvas of [maskImageCanvas, maskOverlayCanvas]) {
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
  }
  maskConjugateLayer.style.width = `${displayWidth}px`;
  maskConjugateLayer.style.height = `${displayHeight}px`;
  renderConjugateLayer();
  updateMaskCursor();
}

function refreshVisibleMaskPreview() {
  if (!maskState.data) return;
  refreshMaskImageFromCurrentMtf();
  renderMaskOverlay();
}

async function loadCurrentCutoutIntoMask() {
  if (!currentMaskSourcePath()) {
    appendLog("Mask: Data_cutout.fits is required for Mask Preview.");
    return;
  }
  const loaded = await loadProjectMask(maskState.type, { silent: false });
  if (loaded) {
    await refreshLensLightSubtractionFromProject({ silent: false });
    if (maskPreviewSourceKind() === "subtracted") refreshVisibleMaskPreview();
  }
}

function loadMaskFromBounds(bounds, maskRows = null, sourcePath = imageView.sourcePath) {
  maskState.width = bounds.width;
  maskState.height = bounds.height;
  maskState.data = new Uint8Array(bounds.width * bounds.height);
  if (maskRows) {
    for (let y = 0; y < Math.min(bounds.height, maskRows.length); y += 1) {
      const row = maskRows[y] || [];
      for (let x = 0; x < Math.min(bounds.width, row.length); x += 1) {
        maskState.data[y * bounds.width + x] = row[x] ? 1 : 0;
      }
    }
  }
  maskState.bounds = bounds;
  maskState.sourcePath = sourcePath;
  updateMaskLensLightSigmaMaxDefault();
  maskState.lastPoint = null;
  maskState.polygonPoints = [];

  maskImageCanvas.width = bounds.width;
  maskImageCanvas.height = bounds.height;
  maskOverlayCanvas.width = bounds.width;
  maskOverlayCanvas.height = bounds.height;
  fitMaskCanvasToStage();
  refreshMaskImageFromCurrentMtf();
  renderMaskOverlay();
}

function currentMaskSourcePath() {
  if (cutoutState.fitsPath) return cutoutState.fitsPath;
  if (!projectState.folder) return "";
  return `${projectState.folder.replace(/[\\/]$/, "")}/Data_cutout.fits`;
}

async function loadProjectMask(type = maskState.type, options = {}) {
  if (!projectState.folder) return false;
  const sourcePath = currentMaskSourcePath();
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestToken = maskState.loadToken;
  try {
    const data = await callBackend("/api/mask/load", {
      project_id: requestProjectId,
      project_folder: requestProjectFolder,
      mask_type: type,
      source_path: sourcePath,
    });
    if (
      requestToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder
    ) {
      return false;
    }
    if (data.source_path) {
      cutoutState.fitsPath = data.source_path;
      cutoutState.previewUrl = data.preview_url || cutoutState.previewUrl;
      cutoutState.shape = data.shape || cutoutState.shape;
      cutoutState.bounds = data.bounds || cutoutState.bounds;
      updateLensSigmaMaxAutoHint();
    }
    setMaskType(data.mask_type || type);
    maskState.previewSrc = data.preview_url || "";
    maskState.sourceImage = data.image || null;
    loadMaskFromBounds(data.bounds, data.mask, data.source_path || sourcePath || imageView.sourcePath);
    maskState.savedPath = data.fits_path || "";
    maskStatus.textContent = `${maskState.type}: ${maskState.width}x${maskState.height}, mask pixels ${data.mask_pixels ?? 0}`;
    renderMaskSubtractionPreview();
    if (!options.silent) {
      appendLog(data.message || `Loaded ${type} from project folder.`);
    }
    return true;
  } catch (error) {
    if (!options.silent) appendLog(`Mask load: ${error.message}`);
    return false;
  }
}

function lensLightSubtractedFitsPath() {
  const variants = Array.isArray(maskState.lensLightJobStatus?.result_variants)
    ? maskState.lensLightJobStatus.result_variants
    : [];
  const selected = selectedLensLightResultVariant();
  if (selected?.subtracted_fits) return selected.subtracted_fits;
  if (!variants.length && maskState.lensLightJobStatus?.subtracted_fits) return maskState.lensLightJobStatus.subtracted_fits;
  if (maskState.lensLightImage?.path) return maskState.lensLightImage.path;
  if (!projectState.folder) return "";
  if (maskState.lensLightResultChoice === "constrained") {
    return `${projectState.folder.replace(/[\\/]$/, "")}/lens_light_subtraction_result/lens_light_subtracted_constrained.fits`;
  }
  return `${projectState.folder.replace(/[\\/]$/, "")}/lens_light_subtraction_result/lens_light_subtracted.fits`;
}

function lensLightSubtractedAvailable() {
  if (maskState.lensLightJobStatus?.subtracted_stale) return false;
  const variants = Array.isArray(maskState.lensLightJobStatus?.result_variants)
    ? maskState.lensLightJobStatus.result_variants
    : [];
  const selected = selectedLensLightResultVariant();
  if (variants.length) return Boolean(selected?.subtracted_fits || selected?.preview_url || selected?.preview_path);
  return Boolean(maskState.lensLightJobStatus?.subtracted_fits || maskState.lensLightImage?.data || maskState.lensLightPreviewUrl);
}

function maskPreviewSourceKind() {
  return maskState.previewSource === "subtracted" ? "subtracted" : "image";
}

function normalizeMaskPreviewColormap(value) {
  return value === "gray" ? "gray" : "twilight";
}

function setMaskPreviewColormap(value, options = {}) {
  maskState.previewColormap = normalizeMaskPreviewColormap(value);
  if (maskPreviewColormap) maskPreviewColormap.value = maskState.previewColormap;
  if (options.refresh !== false) {
    refreshVisibleMaskPreview();
  }
  if (options.save !== false) scheduleProjectSave();
}

function activeMaskPreviewSourcePath() {
  return maskPreviewSourceKind() === "subtracted" ? lensLightSubtractedFitsPath() : maskState.sourcePath;
}

function activeMaskConjugateSource() {
  return maskPreviewSourceKind() === "subtracted" ? "subtracted" : "mask";
}

function setMaskPreviewSource(source, options = {}) {
  const normalized = source === "subtracted" ? "subtracted" : "image";
  if (normalized === "subtracted" && !lensLightSubtractedAvailable()) {
    if (!options.silent) appendLog("Mask Preview: run lens-light subtraction before switching to the subtracted image.");
    maskState.previewSource = "image";
  } else {
    maskState.previewSource = normalized;
  }
  if (maskPreviewSource) maskPreviewSource.value = maskState.previewSource;
  setConjugateMeasureSource(maskState.previewSource === "subtracted" ? "subtracted" : "mask", { save: false });
  refreshVisibleMaskPreview();
  if (options.save !== false) scheduleProjectSave();
}

async function addConjugatePointFromMask(point, sourceKind = activeMaskConjugateSource()) {
  if (!maskState.bounds || !maskState.sourcePath) {
    appendLog("Conjugate point: load an image or cutout first.");
    return;
  }
  const useSubtracted = sourceKind === "subtracted";
  const sourcePath = useSubtracted ? lensLightSubtractedFitsPath() : maskState.sourcePath;
  if (useSubtracted && !sourcePath) {
    appendLog("Conjugate point: run lens-light subtraction before measuring on the subtracted image.");
    return;
  }
  const xDisplay = useSubtracted ? point.x : maskState.bounds.x0 + point.x;
  const yDisplay = useSubtracted ? point.y : maskState.bounds.y0 + point.y;
  const requestX = maskState.conjugateClickMode === "brightest" ? Math.floor(xDisplay) : xDisplay;
  const requestY = maskState.conjugateClickMode === "brightest" ? Math.floor(yDisplay) : yDisplay;
  try {
    const data = await callBackend("/api/mask/conjugate-point", {
      path: sourcePath,
      x: requestX,
      y: requestY,
      method: maskState.conjugateClickMode,
      fwhm: 1.5,
      size: maskState.conjugateClickMode === "gaussian" ? 9 : 4,
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    const local = useSubtracted
      ? { x: data.center_display.x, y: data.center_display.y }
      : {
          x: data.center_display.x - maskState.bounds.x0,
          y: data.center_display.y - maskState.bounds.y0,
        };
    const record = {
      local,
      display: data.center_display,
      data: data.center_data,
      arcsec: data.center_arcsec,
      fwhm: data.fwhm ?? null,
      params: data.params,
      measurement_method: data.method || maskState.conjugateClickMode,
      measurement_source: useSubtracted ? "lens_light_subtracted" : "image_preview",
      project_id: projectState.id || "",
      project_folder: projectState.folder || "",
      source_path: sourcePath,
    };
    activeConjugatePoints().push(record);
    renderMaskOverlay();
    updateConjugateStatus();
    const planeLabel = maskState.conjugatePlane === "source2" ? "P2" : "P1";
    const sourceLabel = useSubtracted ? "lens-light subtracted" : "image preview";
    const methodLabel = conjugateClickModeLabel(data.method || maskState.conjugateClickMode);
    const pointCount = activeConjugatePoints().length;
    appendLog(
      `Conjugate point ${planeLabel}-${pointCount} (${sourceLabel}, ${methodLabel}): x=${record.arcsec.x.toFixed(4)}, y=${record.arcsec.y.toFixed(4)} arcsec.`
    );
    syncLensScriptConjugatesFromMask();
    scheduleProjectSave();
  } catch (error) {
    appendLog(`Conjugate point: ${error.message}`);
  }
}

function drawMaskImageFromPreviewUrl(src) {
  if (!src || !maskState.width || !maskState.height) return;
  const versionedSrc = `${src}${src.includes("?") ? "&" : "?"}mask_v=${Date.now()}`;
  const token = `base:${versionedSrc}:${maskState.width}x${maskState.height}`;
  maskState.previewToken = token;
  const img = new Image();
  img.onload = () => {
    if (maskState.previewToken !== token) return;
    const ctx = maskImageCanvas.getContext("2d");
    maskImageCanvas.width = maskState.width;
    maskImageCanvas.height = maskState.height;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, maskState.width, maskState.height);
    ctx.drawImage(img, 0, 0, maskState.width, maskState.height);
    maskImageCanvas.classList.add("visible");
    renderCanvasAxes(maskImageCanvas, currentSourcePixelScaleArcsec());
  };
  img.src = versionedSrc;
}

function drawMaskMatplotlibPreview(src) {
  if (!src || !maskState.width || !maskState.height) return;
  const url = cacheBustUrl(src);
  const token = `matplotlib:${url}:${maskState.width}x${maskState.height}`;
  maskState.previewToken = token;
  const img = new Image();
  img.onload = () => {
    if (maskState.previewToken !== token) return;
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return;
    maskImageCanvas.width = width;
    maskImageCanvas.height = height;
    const ctx = maskImageCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    maskImageCanvas.classList.add("visible");
    fitMaskCanvasToStage();
    removePreviewAxes(maskImageCanvas);
  };
  img.src = url;
}

function scheduleMaskMatplotlibPreview(delay = 120) {
  if (!maskState.data || !projectState.folder || !maskState.sourcePath) return;
  if (maskState.matplotlibPreviewTimer) {
    window.clearTimeout(maskState.matplotlibPreviewTimer);
  }
  maskState.matplotlibPreviewTimer = window.setTimeout(() => {
    maskState.matplotlibPreviewTimer = null;
    void refreshMaskMatplotlibPreview();
  }, delay);
}

async function refreshMaskMatplotlibPreview() {
  if (!maskState.data || !projectState.folder || !maskState.sourcePath) return;
  const previewSourcePath = activeMaskPreviewSourcePath();
  if (!previewSourcePath) return;
  const token = maskState.matplotlibPreviewToken + 1;
  maskState.matplotlibPreviewToken = token;
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestSourcePath = previewSourcePath;
  const requestPreviewSource = maskPreviewSourceKind();
  const requestMaskType = maskState.type;
  const requestLoadToken = maskState.loadToken;
  const previewMtf = requestPreviewSource === "subtracted" ? maskSubtractionMtfParameters() : maskCutoutMtfParameters();
  try {
    const data = await callBackend("/api/mask/preview", {
      project_id: requestProjectId,
      project_folder: requestProjectFolder,
      source_path: requestSourcePath,
      source_kind: requestPreviewSource,
      display_mtf: previewMtf,
      display_cmap: maskState.previewColormap,
      mask_type: requestMaskType,
      mask: maskRowsForSave(),
      conjugate_points_source1: maskState.conjugatePoints,
      conjugate_points_source2: maskState.conjugatePointsSource2,
    });
    if (
      token !== maskState.matplotlibPreviewToken ||
      requestLoadToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder ||
      requestPreviewSource !== maskPreviewSourceKind() ||
      requestMaskType !== maskState.type
    ) {
      return;
    }
    drawMaskMatplotlibPreview(data.preview_url || "");
  } catch (error) {
    if (token === maskState.matplotlibPreviewToken) {
      maskStatus.textContent = `Mask preview failed: ${error.message}`;
    }
  }
}

function dataBoundsToPreviewCanvasBounds(bounds) {
  const stride = imageDisplayStride();
  const sourceHeight = Math.max(1, Number(bounds.source_shape?.[0]) || imageSourceHeight());
  const yOffset = Math.max(0, sourceHeight - 1 - (imagePreviewCanvas.height - 1) * stride);
  const x0Source = Number(bounds.x0) || 0;
  const x1Source = Number(bounds.x1 ?? x0Source + Number(bounds.width || 0));
  const y1Source = Number(bounds.y1 ?? Number(bounds.y0 || 0) + Number(bounds.height || 0));
  const width = (x1Source - x0Source) / stride;
  const height = Number(bounds.height ?? bounds.y1 - bounds.y0) / stride;
  return {
    x0: x0Source / stride,
    y0: (sourceHeight - y1Source - yOffset) / stride,
    width,
    height,
    targetWidth: Math.max(1, Math.round(x1Source - x0Source)),
    targetHeight: Math.max(1, Math.round(Number(bounds.height ?? bounds.y1 - bounds.y0))),
  };
}

function drawCurrentImagePreviewRegion(canvas, bounds, targetWidth, targetHeight) {
  if (!canvas || !bounds || !targetWidth || !targetHeight || !imagePreviewCanvas.width || !imagePreviewCanvas.height) return false;
  const x0 = Math.round(Number(bounds.x0) || 0);
  const y0 = Math.round(Number(bounds.y0) || 0);
  const width = Math.round(Number(bounds.width ?? bounds.x1 - bounds.x0) || targetWidth);
  const height = Math.round(Number(bounds.height ?? bounds.y1 - bounds.y0) || targetHeight);
  const canCrop =
    x0 >= 0 &&
    y0 >= 0 &&
    width > 0 &&
    height > 0 &&
    x0 + width <= imagePreviewCanvas.width &&
    y0 + height <= imagePreviewCanvas.height;
  if (!canCrop) return false;
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, targetWidth, targetHeight);
  ctx.drawImage(imagePreviewCanvas, x0, y0, width, height, 0, 0, targetWidth, targetHeight);
  canvas.classList.add("visible");
  renderCanvasAxes(canvas, currentSourcePixelScaleArcsec());
  return true;
}

function normalizedImagePath(path) {
  return String(path || "").replace(/\\/g, "/");
}

function sameImagePath(left, right) {
  const leftPath = normalizedImagePath(left);
  const rightPath = normalizedImagePath(right);
  return Boolean(leftPath && rightPath && leftPath === rightPath);
}

function refreshMaskImageFromCurrentMtf() {
  if (!maskState.bounds || !maskState.width || !maskState.height) return;
  if (maskState.useMatplotlibPreview) {
    scheduleMaskMatplotlibPreview(40);
    return;
  }
  drawMaskImageFromPreviewUrl(maskState.previewSrc);
}

function clearMaskData() {
  if (!maskState.data) return;
  maskState.data.fill(0);
  maskState.lastPoint = null;
  maskState.polygonPoints = [];
  renderMaskOverlay();
  appendLog("Mask: cleared.");
  scheduleMaskAutoSave();
  scheduleProjectSave();
}

function scheduleMaskAutoSave() {
  if (!maskState.data || (!projectState.folder && !projectState.id)) return;
  if (maskState.autosaveTimer) {
    window.clearTimeout(maskState.autosaveTimer);
  }
  maskState.autosaveTimer = window.setTimeout(() => {
    maskState.autosaveTimer = null;
    void saveMaskFits({ autosave: true });
  }, 250);
}

async function saveMaskFits(options = {}) {
  const autosave = Boolean(options.autosave);
  if (!maskState.data) {
    if (!autosave) appendLog("Mask: load a cutout first.");
    return;
  }
  const selectedPixels = maskPixelCount();
  if (!autosave && (maskState.type === "mask_1" || maskState.type === "mask_2") && selectedPixels === 0) {
    const message = `${maskState.type} has 0 selected pixels. Select Add, then paint with Brush or define a polygon with at least three Line clicks.`;
    maskStatus.textContent = message;
    appendLog(`Mask: ${message}`);
    return;
  }
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestSourcePath = maskState.sourcePath;
  const requestMaskType = maskState.type;
  const requestBounds = maskState.bounds;
  const requestToken = maskState.loadToken;
  if (autosave && maskState.autosaveInFlight) {
    maskState.autosaveQueued = true;
    return;
  }
  if (autosave) {
    maskState.autosaveInFlight = true;
  } else {
    maskSave.disabled = true;
  }
  try {
    const rows = maskRowsForSave();
    const data = await callBackend("/api/mask/save", {
      source_path: requestSourcePath,
      mask_type: requestMaskType,
      bounds: requestBounds,
      mask: rows,
      project_id: requestProjectId,
      project_folder: requestProjectFolder,
      autosave,
    });
    if (
      requestToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder
    ) {
      return;
    }
    maskState.savedPath = data.project_mask_path || data.fits_path || "";
    const savedPixels = Number(data.mask_pixels ?? selectedPixels);
    maskStatus.textContent = `Saved: ${PathName(maskState.savedPath)} (${savedPixels} pixels)`;
    if (!autosave) {
      appendLog(`${data.message} (${savedPixels} pixels) -> ${data.fits_path}`);
    }
    void loadLensScriptCutoutPreview({ silent: true });
    scheduleProjectSave();
  } catch (error) {
    if (!autosave) {
      appendLog(`Mask: ${error.message}`);
    } else {
      maskStatus.textContent = `Autosave failed: ${error.message}`;
    }
  } finally {
    if (autosave) {
      maskState.autosaveInFlight = false;
      if (maskState.autosaveQueued) {
        maskState.autosaveQueued = false;
        scheduleMaskAutoSave();
      }
    } else {
      maskSave.disabled = false;
    }
  }
}

function maskSubtractionMtfParameters() {
  return sliderMtfParameters(maskSubtractionMtfShadows, maskSubtractionMtfMidtones, maskSubtractionMtfHighlights);
}

function maskCutoutMtfParameters() {
  return sliderMtfParameters(maskCutoutMtfShadows, maskCutoutMtfMidtones, maskCutoutMtfHighlights);
}

function updateMtfValueInput(input, value, slider) {
  if (!input) return;
  input.value = formatSliderNumber(slider, value);
  if (slider) {
    input.min = slider.min;
    input.max = slider.max;
    input.step = slider.step;
    input.title = `max ${Number(slider.max).toFixed(3)}`;
  }
}

function updateMaskCutoutMtfLabels() {
  const { shadows, midtones, highlights } = maskCutoutMtfParameters();
  updateMtfValueInput(maskCutoutMtfShadowsValue, shadows, maskCutoutMtfShadows);
  updateMtfValueInput(maskCutoutMtfMidtonesValue, midtones, maskCutoutMtfMidtones);
  updateMtfValueInput(maskCutoutMtfHighlightsValue, highlights, maskCutoutMtfHighlights);
}

function updateMaskSubtractionMtfLabels() {
  const { shadows, midtones, highlights } = maskSubtractionMtfParameters();
  updateMtfValueInput(maskSubtractionMtfShadowsValue, shadows, maskSubtractionMtfShadows);
  updateMtfValueInput(maskSubtractionMtfMidtonesValue, midtones, maskSubtractionMtfMidtones);
  updateMtfValueInput(maskSubtractionMtfHighlightsValue, highlights, maskSubtractionMtfHighlights);
}

function renderMaskSubtractionPreview() {
  updateMaskSubtractionMtfLabels();
  const image = maskState.lensLightImage;
  const sourcePath = lensLightSubtractedFitsPath();
  if (maskSubtractionStage) maskSubtractionStage.hidden = true;
  if (maskSubtractionPreview) maskSubtractionPreview.hidden = true;
  if (maskSubtractionEmpty) maskSubtractionEmpty.hidden = true;
  if (!sourcePath || !lensLightSubtractedAvailable()) {
    if (maskSubtractionEmpty) {
      maskSubtractionEmpty.textContent = maskState.lensLightJobStatus?.subtracted_stale_message || "Run lens light subtraction.";
    }
    if (maskSubtractionPreview) removePreviewAxes(maskSubtractionPreview);
    return;
  }
  if (image?.shape) {
    const [height, width] = image.shape;
    if (maskState.width && maskState.height && (Number(width) !== maskState.width || Number(height) !== maskState.height)) {
      if (maskSubtractionEmpty) {
        maskSubtractionEmpty.textContent =
          `Lens-light subtraction is ${width}x${height}; current mask is ${maskState.width}x${maskState.height}. Run lens light subtraction again.`;
      }
      maskSubtractionStatus.textContent = "stale";
      if (maskSubtractionPreview) removePreviewAxes(maskSubtractionPreview);
      return;
    }
  }
  if (maskPreviewSourceKind() === "subtracted") {
    scheduleMaskMatplotlibPreview(40);
  }
}

function scheduleMaskSubtractionMatplotlibPreview(delay = 120) {
  if (maskPreviewSourceKind() === "subtracted") {
    scheduleMaskMatplotlibPreview(delay);
  }
}

async function refreshMaskSubtractionMatplotlibPreview() {
  const sourcePath = lensLightSubtractedFitsPath();
  if (!maskState.data || !projectState.folder || !sourcePath || !lensLightSubtractedAvailable()) return;
  const token = maskState.lensLightPreviewSeq + 1;
  maskState.lensLightPreviewSeq = token;
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestMaskType = maskState.type;
  const requestLoadToken = maskState.loadToken;
  try {
    const data = await callBackend("/api/mask/preview", {
      project_id: requestProjectId,
      project_folder: requestProjectFolder,
      source_path: sourcePath,
      source_kind: "subtracted",
      display_mtf: maskSubtractionMtfParameters(),
      display_cmap: maskState.previewColormap,
      mask_type: requestMaskType,
      mask: maskRowsForSave(),
      conjugate_points_source1: maskState.conjugatePoints,
      conjugate_points_source2: maskState.conjugatePointsSource2,
    });
    if (
      token !== maskState.lensLightPreviewSeq ||
      requestLoadToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder ||
      requestMaskType !== maskState.type
    ) {
      return;
    }
    drawMaskSubtractionMatplotlibPreview(data.preview_url || "", token);
  } catch (error) {
    if (token === maskState.lensLightPreviewSeq) {
      maskSubtractionPreview.hidden = true;
      maskSubtractionEmpty.hidden = false;
      maskSubtractionEmpty.textContent = `Preview failed: ${error.message}`;
    }
  }
}

function drawMaskSubtractionMatplotlibPreview(src, requestToken = null) {
  if (!src) return;
  if (requestToken !== null && requestToken !== maskState.lensLightPreviewSeq) return;
  if (maskSubtractionStage) maskSubtractionStage.hidden = false;
  const url = cacheBustUrl(src);
  const token = `lens-light-matplotlib:${url}`;
  maskState.lensLightPreviewToken = token;
  const img = new Image();
  img.onload = () => {
    if (maskState.lensLightPreviewToken !== token) return;
    if (requestToken !== null && requestToken !== maskState.lensLightPreviewSeq) return;
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return;
    maskSubtractionPreview.width = width;
    maskSubtractionPreview.height = height;
    const ctx = maskSubtractionPreview.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    maskSubtractionPreview.hidden = false;
    fitMaskSubtractionCanvasToStage();
    maskSubtractionEmpty.hidden = true;
    maskSubtractionEmpty.textContent = "Run lens light subtraction.";
    removePreviewAxes(maskSubtractionPreview);
  };
  img.onerror = () => {
    if (maskState.lensLightPreviewToken !== token) return;
    if (requestToken !== null && requestToken !== maskState.lensLightPreviewSeq) return;
    maskSubtractionPreview.hidden = true;
    maskSubtractionEmpty.hidden = false;
    maskSubtractionEmpty.textContent = "Preview failed.";
  };
  img.src = url;
  maskSubtractionPreview.hidden = true;
  maskSubtractionEmpty.hidden = false;
  maskSubtractionEmpty.textContent = "Loading matplotlib preview...";
}

function updateLensLightResultChoices(status = null) {
  if (!maskLensLightResultChoice) return;
  const variants = Array.isArray(status?.result_variants) ? status.result_variants : [];
  const current = maskState.lensLightResultChoice || "unconstrained";
  maskLensLightResultChoice.innerHTML = "";
  const byKey = new Map();
  [
    { key: "unconstrained", label: "SVI" },
    { key: "constrained", label: "Semilinear MGE (no SVI)" },
  ].forEach((variant) => byKey.set(variant.key, variant));
  variants.forEach((variant) => {
    if (!variant?.key) return;
    byKey.set(variant.key, { ...byKey.get(variant.key), ...variant });
  });
  const optionKeys = variants.length
    ? ["unconstrained", "constrained"].filter((key) => variants.some((variant) => variant?.key === key))
    : ["unconstrained", "constrained"];
  const options = optionKeys.map((key) => byKey.get(key)).filter(Boolean);
  options.forEach((variant) => {
    const option = document.createElement("option");
    option.value = variant.key || "";
    option.textContent = variant.label || variant.key || "Result";
    maskLensLightResultChoice.appendChild(option);
  });
  const hasCurrent = options.some((variant) => variant.key === current);
  maskState.lensLightResultChoice = hasCurrent ? current : (options[0]?.key || "unconstrained");
  maskLensLightResultChoice.value = maskState.lensLightResultChoice;
}

function selectedLensLightResultVariant(status = maskState.lensLightJobStatus) {
  const variants = Array.isArray(status?.result_variants) ? status.result_variants : [];
  const selected = variants.find((item) => item?.key === maskState.lensLightResultChoice);
  if (selected) return selected;
  if (!projectState.folder) return null;
  const root = `${projectState.folder.replace(/[\\/]$/, "")}/lens_light_subtraction_result`;
  if (maskState.lensLightResultChoice === "constrained") {
    return {
      key: "constrained",
      label: "Semilinear MGE (no SVI)",
      subtracted_fits: `${root}/lens_light_subtracted_constrained.fits`,
      preview_path: `${root}/lens_light_subtracted_constrained_preview.png`,
      kwargs_lens_light: `${root}/kwargs_lens_light_constrained.pkl`,
    };
  }
  return {
    key: "unconstrained",
    label: "SVI",
    subtracted_fits: `${root}/lens_light_subtracted.fits`,
    preview_path: `${root}/lens_light_subtracted_preview.png`,
    kwargs_lens_light: `${root}/kwargs_lens_light.pkl`,
  };
}

function projectRelativeArtifactPath(value) {
  const normalized = String(value || "").trim().replace(/\\/g, "/");
  if (!normalized) return "";
  const projectFolder = String(projectState.folder || "").replace(/\\/g, "/").replace(/\/+$/, "");
  if (projectFolder && normalized.startsWith(`${projectFolder}/`)) {
    return normalized.slice(projectFolder.length + 1);
  }
  const runMatch = normalized.match(/\/runs\/([^/]+)\/(.+)$/);
  if (runMatch && decodeURIComponent(runMatch[1]) === currentProjectFolderName()) {
    return runMatch[2];
  }
  return normalized;
}

function setLensLightHandoffState(state, message) {
  if (!lensLightHandoffStatus) return;
  lensLightHandoffStatus.dataset.state = state;
  const label = lensLightHandoffStatus.querySelector("span:last-child");
  if (label) label.textContent = message;
}

function syncLensLightPriorFromMask(status = maskState.lensLightJobStatus, options = {}) {
  if (!lensLightExternalMode || !lensLightExternalPath) return false;
  if (!status || status.state !== "completed") {
    const state = status?.state === "failed" ? "error" : status?.state === "running" ? "pending" : "idle";
    const message = status?.state === "failed"
      ? "Step 4 prior unavailable: lens-light fit failed"
      : status?.state === "running"
        ? "Step 4 prior: lens-light fit running"
        : "Step 4 prior: free lens-light fit";
    setLensLightHandoffState(state, message);
    return false;
  }
  if (status.subtracted_stale) {
    setLensLightHandoffState("error", "Step 4 prior is stale for the current cutout");
    return false;
  }

  const selected = selectedLensLightResultVariant(status);
  if (!selected) {
    setLensLightHandoffState("error", "Step 4 prior has no reusable MGE parameters");
    return false;
  }
  const variantKey = selected.key === "constrained" ? "constrained" : "unconstrained";
  const fallbackPath = variantKey === "constrained"
    ? "lens_light_subtraction_result/kwargs_lens_light_constrained.pkl"
    : "lens_light_subtraction_result/kwargs_lens_light.pkl";
  const kwargsPath = projectRelativeArtifactPath(
    selected.kwargs_lens_light || (variantKey === "unconstrained" ? status.kwargs_lens_light : "") || fallbackPath,
  );
  const previousMode = lensLightExternalMode.value || "free";
  const previousPath = lensLightExternalPath.value.trim();
  const previousNGauss = lensLightGaussianCount?.value || "";
  const lensSigmaMinControl = lensModelForm?.querySelector('[name="lens_sigma_min"]');
  const lensSigmaMaxControl = lensSigmaMaxInput();
  const previousSigmaMin = lensSigmaMinControl?.value || "";
  const previousSigmaMax = lensSigmaMaxControl?.value || "";
  const previousCenterMaxOffset = lensLightCenterMaxOffset?.value || "";
  if (lensState.lensLightPriorAuto && previousMode !== "fixed") {
    lensLightExternalMode.value = "start";
  }
  if (lensState.lensLightPriorAuto || previousPath === "" || previousPath.startsWith("lens_light_subtraction_result/")) {
    lensLightExternalPath.value = kwargsPath;
  }

  const nGauss = Number(status.summary?.n_gauss ?? maskLensLightNGauss?.value);
  if (lensState.lensLightPriorAuto && lensLightGaussianCount && Number.isFinite(nGauss) && nGauss > 0) {
    lensLightGaussianCount.value = String(Math.round(nGauss));
  }

  const sigmaMin = Number(maskLensLightSigmaMin?.value);
  const sigmaMaxText = maskLensLightSigmaMax?.value.trim() || "";
  if (lensState.lensLightPriorAuto && lensSigmaMinControl && Number.isFinite(sigmaMin) && sigmaMin > 0) {
    lensSigmaMinControl.value = String(sigmaMin);
  }
  if (lensState.lensLightPriorAuto && lensSigmaMaxControl && sigmaMaxText) {
    lensSigmaMaxControl.value = sigmaMaxText;
  }
  const centerMaxOffset = Number(maskLensLightCenterMaxOffset?.value);
  if (lensState.lensLightPriorAuto && lensLightCenterMaxOffset && Number.isFinite(centerMaxOffset) && centerMaxOffset > 0) {
    lensLightCenterMaxOffset.value = String(centerMaxOffset);
  }
  updateLensSigmaMaxAutoHint();

  const activeMode = lensLightExternalMode.value;
  const generatedMode = String(
    lensState.modelConfig?.lens_light_external_mode ?? lensState.modelConfig?.light?.lens?.external_mode ?? "free",
  );
  const generatedPath = projectRelativeArtifactPath(
    lensState.modelConfig?.lens_light_external_path ?? lensState.modelConfig?.light?.lens?.external_kwargs_path ?? "",
  );
  const usesExternalPrior = activeMode === "start" || activeMode === "fixed";
  const activePath = lensLightExternalPath.value.trim();
  const activeNGauss = lensLightGaussianCount?.value || "";
  const activeSigmaMin = lensSigmaMinControl?.value || "";
  const activeSigmaMax = lensSigmaMaxControl?.value || "";
  const activeCenterMaxOffset = lensLightCenterMaxOffset?.value || "";
  const handoffSettingsChanged = (
    previousNGauss !== activeNGauss ||
    previousSigmaMin !== activeSigmaMin ||
    previousSigmaMax !== activeSigmaMax ||
    previousCenterMaxOffset !== activeCenterMaxOffset
  );
  const priorChanged = (
    previousMode !== activeMode ||
    (usesExternalPrior && previousPath !== activePath) ||
    handoffSettingsChanged
  );
  const generatedNGauss = Number(
    lensState.modelConfig?.n_gauss_lens ?? lensState.modelConfig?.light?.lens?.n_gauss,
  );
  const generatedSigma = (
    lensState.modelConfig?.lens_sigma_lims ?? lensState.modelConfig?.light?.lens?.sigma_lims ?? []
  );
  const generatedSigmaMax = generatedSigma?.[1];
  const generatedCenterMaxOffset = Number(
    lensState.modelConfig?.lens_light_center_max_offset ??
      lensState.modelConfig?.light?.lens?.center_max_offset ??
      0.4,
  );
  const sigmaMaxMatches = activeSigmaMax.trim().toLowerCase() === "auto"
    ? generatedSigmaMax === null || generatedSigmaMax === undefined
    : Number(generatedSigmaMax) === Number(activeSigmaMax);
  const generatedSettingsMismatch = Boolean(lensState.generatedCode.trim()) && (
    (Number.isFinite(generatedNGauss) && generatedNGauss !== Number(activeNGauss)) ||
    Number(generatedSigma?.[0]) !== Number(activeSigmaMin) ||
    !sigmaMaxMatches ||
    generatedCenterMaxOffset !== Number(activeCenterMaxOffset)
  );
  const generatedPriorMismatch = Boolean(lensState.generatedCode.trim()) && (
    generatedMode !== activeMode || (usesExternalPrior && generatedPath !== activePath)
  );
  if (priorChanged || generatedPriorMismatch || generatedSettingsMismatch) {
    markLensScriptStale("Step 4 lens-light prior changed");
  }

  const variantLabel = selected.label || (variantKey === "constrained" ? "Semilinear MGE" : "SVI");
  const modeLabel = activeMode === "fixed"
    ? "fixed model"
    : activeMode === "start"
      ? "initialization prior"
      : "available; free fit selected";
  const gaussianLabel = Number.isFinite(nGauss) && nGauss > 0 ? `, ${Math.round(nGauss)} Gaussian` : "";
  setLensLightHandoffState(
    lensState.scriptStale || !usesExternalPrior ? "pending" : "ready",
    `Step 4 prior: ${variantLabel} ${modeLabel}${gaussianLabel}${lensState.scriptStale ? "; regenerate code" : ""}`,
  );

  const handoffKey = `${projectState.id}:${variantKey}:${activePath}:${activeMode}`;
  if (!options.silent && handoffKey !== lensLightHandoffKey) {
    appendLog(`Lens model: Step 4 ${variantLabel} selected as lens-light ${modeLabel}.`);
  }
  lensLightHandoffKey = handoffKey;
  updateWorkflowStatus();
  return true;
}

function setMaskSubtractionPreview(previewUrl, status = null, options = {}) {
  maskState.lensLightJobStatus = status || maskState.lensLightJobStatus;
  updateLensLightResultChoices(maskState.lensLightJobStatus);
  const selected = selectedLensLightResultVariant(maskState.lensLightJobStatus);
  maskState.lensLightPreviewUrl = normalizeProjectAssetUrl(
    selected?.preview_url || previewUrl || selected?.preview_path || "",
  );
  if (status?.subtracted_stale) {
    maskState.lensLightImage = null;
  }
  if (!selected && status?.subtracted_image) {
    maskState.lensLightImage = status.subtracted_image;
  } else if (selected) {
    maskState.lensLightImage = null;
  }
  if (maskState.lensLightPreviewUrl) {
    drawMaskSubtractionMatplotlibPreview(maskState.lensLightPreviewUrl);
  } else if (maskState.lensLightImage) {
    renderMaskSubtractionPreview();
  } else {
    if (maskSubtractionStage) maskSubtractionStage.hidden = true;
    if (maskSubtractionPreview) maskSubtractionPreview.hidden = true;
    if (maskSubtractionEmpty) {
      maskSubtractionEmpty.hidden = true;
      maskSubtractionEmpty.textContent = status?.subtracted_stale_message || "Run lens light subtraction.";
    }
  }
  if (status?.state) {
    const loss = Number(status.summary?.final_loss);
    maskSubtractionStatus.textContent = status.subtracted_stale
      ? "stale"
      : Number.isFinite(loss) ? `loss ${loss.toPrecision(6)}` : status.state;
  } else if (maskState.lensLightPreviewUrl) {
    maskSubtractionStatus.textContent = "completed";
  } else {
    maskSubtractionStatus.textContent = "Not run";
  }
  if (maskPreviewSourceKind() === "subtracted") {
    refreshVisibleMaskPreview();
  }
  syncLensLightPriorFromMask(maskState.lensLightJobStatus, { silent: Boolean(options.silent) });
  scheduleLensScriptCutoutPreviewRefresh();
}

async function refreshLensLightSubtractionFromProject(options = {}) {
  if (!projectState.folder) return false;
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestToken = maskState.loadToken;
  const query = new URLSearchParams();
  if (requestProjectId) query.set("project_id", requestProjectId);
  query.set("project_folder", requestProjectFolder);
  try {
    const data = await callBackend(`/api/mask/lens-light-subtraction-latest?${query.toString()}`);
    if (
      requestToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder
    ) {
      return false;
    }
    maskState.lensLightJobId = data.job_id || "";
    maskState.lensLightJobStatus = data;
    updateLensLightSubtractionProgress(maskState.lensLightJobId || "latest", data);
    if (data.preview_url || data.subtracted_image || Array.isArray(data.result_variants)) {
      setMaskSubtractionPreview(data.preview_url || "", data);
    }
    if (data.state && !["completed", "failed", "stopped"].includes(data.state) && data.job_id) {
      pollLensLightSubtractionJob(data.job_id);
    } else {
      setLensLightSubtractionRunning(false);
    }
    if (!options.silent) {
      appendLog(data.message || "Loaded lens-light subtraction result from project folder.");
    }
    if (options.save !== false) scheduleProjectSave();
    return true;
  } catch (error) {
    if (!options.silent) appendLog(`Lens light subtraction load: ${error.message}`);
    return false;
  }
}

function updateLensLightSubtractionProgress(jobId, data) {
  const progress = {
    ...(data.progress || {}),
    fraction: data.progress?.fraction ?? (data.state === "completed" ? 1 : 0),
    message: Number.isFinite(Number(data.summary?.final_loss))
      ? `${data.message || "Lens-light subtraction completed."} loss ${Number(data.summary.final_loss).toPrecision(6)}`
      : data.message || `Lens-light subtraction ${data.state || "running"}`,
  };
  updateLogProgress(`Lens light subtraction ${jobId}`, progress, data.state || "running");
  syncLensLightPriorFromMask(data, { silent: true });
}

function setLensLightSubtractionRunning(running) {
  maskState.lensLightJobRunning = Boolean(running);
  if (maskLensLightSemilinear) maskLensLightSemilinear.disabled = Boolean(running);
  maskLensLightSubtraction.disabled = Boolean(running);
  if (maskLensLightSubtractionGpu) maskLensLightSubtractionGpu.disabled = Boolean(running);
  maskLensLightStop.disabled = !running;
}

async function pollLensLightSubtractionJob(jobId) {
  maskState.lensLightJobId = jobId;
  setLensLightSubtractionRunning(true);
  const requestProjectId = projectState.id;
  const requestProjectFolder = projectState.folder;
  const requestToken = maskState.loadToken;
  let transientErrors = 0;
  for (let i = 0; i < 7200; i += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    if (
      requestToken !== maskState.loadToken ||
      requestProjectId !== projectState.id ||
      requestProjectFolder !== projectState.folder
    ) {
      return;
    }
    try {
      const query = requestProjectId ? `?project_id=${encodeURIComponent(requestProjectId)}` : "";
      const data = await callBackend(`/api/mask/lens-light-subtraction-job/${jobId}${query}`);
      transientErrors = 0;
      if (
        requestToken !== maskState.loadToken ||
        requestProjectId !== projectState.id ||
        requestProjectFolder !== projectState.folder
      ) {
        return;
      }
      maskState.lensLightJobStatus = data;
      updateLensLightSubtractionProgress(jobId, data);
      maskSubtractionStatus.textContent = data.state || "running";
      if (data.preview_url) {
        setMaskSubtractionPreview(data.preview_url, data);
      }
      if (data.state === "completed" || data.state === "failed" || data.state === "stopped") {
        appendLog(`Lens light subtraction ${jobId}: ${data.state}.`);
        setActivity(`Lens light subtraction ${jobId}: ${data.state}`, false);
        setLensLightSubtractionRunning(false);
        scheduleProjectSave();
        return;
      }
    } catch (error) {
      transientErrors += 1;
      if (transientErrors < 4) {
        appendLog(`Lens light subtraction ${jobId}: waiting for status (${error.message})`);
        continue;
      }
      appendLog(`Lens light subtraction ${jobId}: ${error.message}`);
      setActivity(`Lens light subtraction failed: ${error.message}`, false);
      setLensLightSubtractionRunning(false);
      return;
    }
  }
  appendLog(`Lens light subtraction ${jobId}: still running.`);
  setLensLightSubtractionRunning(false);
}

async function generateLensLightSubtractionScript(useGpu = false, options = {}) {
  const actionLabel = options.actionLabel || (useGpu ? "Sciama GPU lens-light subtraction" : "lens light subtraction");
  const preferredResultChoice = options.preferredResultChoice || "";
  const semilinearMgeSteps = options.semilinearMgeSteps ?? Number(maskLensLightSviSteps.value);
  const fullSviSteps = options.sviSteps ?? Number(maskLensLightUnconstrainedSviSteps.value);
  if (!projectState.id || !projectState.folder) {
    await saveProjectNow();
  }
  if (!projectState.id || !projectState.folder) {
    appendLog("Lens light subtraction: save or load a project first.");
    return;
  }
  if (useGpu) {
    const approved = window.confirm(
      "Upload a slim project to Sciama and request GPU for lens-light subtraction? Priority: full A100, then full L40, then A100 fallback partitions.",
    );
    if (!approved) return;
  }

  setLensLightSubtractionRunning(true);
  setActivity(useGpu ? "Submitting Sciama GPU lens-light subtraction..." : `Starting ${actionLabel}...`, true);
  try {
    if (maskState.data) {
      await saveMaskFits();
    }
    await saveProjectNow();
    if (preferredResultChoice) {
      maskState.lensLightResultChoice = preferredResultChoice;
      if (maskLensLightResultChoice) maskLensLightResultChoice.value = preferredResultChoice;
    }
    if (options.updateStepControls) {
      maskLensLightSviSteps.value = String(semilinearMgeSteps);
      maskLensLightUnconstrainedSviSteps.value = String(fullSviSteps);
    }
    setMaskSubtractionPreview("", { state: "running" });
    const endpoint = useGpu ? "/api/mask/lens-light-subtraction-run-gpu" : "/api/mask/lens-light-subtraction-run";
    const data = await callBackend(endpoint, {
      project_id: projectState.id,
      project_folder: projectState.folder,
      n_gauss: Number(maskLensLightNGauss.value),
      sigma_min: Number(maskLensLightSigmaMin.value),
      sigma_max: maskLensLightSigmaMax.value.trim(),
      center_max_offset: Number(maskLensLightCenterMaxOffset.value),
      semilinear_mge_steps: semilinearMgeSteps,
      constrained_svi_steps: semilinearMgeSteps,
      unconstrained_svi_steps: fullSviSteps,
      svi_steps: fullSviSteps,
      background_corner: maskLensLightBgCorner.value.trim(),
    });
    maskState.lensLightJobId = data.job_id || "";
    maskState.lensLightJobStatus = data;
    updateLensLightSubtractionProgress(maskState.lensLightJobId, data);
    appendLog(data.message || (useGpu ? "Started Sciama GPU lens-light subtraction." : "Started lens light subtraction."));
    if (data.preview_url) {
      setMaskSubtractionPreview(data.preview_url, data);
    }
    if (data.job_id && data.state !== "completed" && data.state !== "failed") {
      pollLensLightSubtractionJob(data.job_id);
    } else {
      setActivity(data.message || "Lens light subtraction finished.", false);
      setLensLightSubtractionRunning(false);
      scheduleProjectSave();
    }
  } catch (error) {
    appendLog(`Lens light subtraction: ${error.message}`);
    setActivity("Lens light subtraction failed.", false);
    setLensLightSubtractionRunning(false);
  }
}

async function stopLensLightSubtractionJob() {
  const jobId = maskState.lensLightJobId || maskState.lensLightJobStatus?.job_id || "";
  if (!jobId && !projectState.folder) {
    appendLog("Lens light subtraction: no running job to stop.");
    return;
  }
  maskLensLightStop.disabled = true;
  setActivity("Stopping lens light subtraction...", true);
  try {
    const data = await callBackend("/api/mask/lens-light-subtraction-stop", {
      job_id: jobId,
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    maskState.lensLightJobStatus = data;
    updateLensLightSubtractionProgress(data.job_id || jobId, data);
    maskSubtractionStatus.textContent = data.state || "stopped";
    appendLog(data.message || "Lens light subtraction stopped.");
    setActivity(data.message || "Lens light subtraction stopped.", false);
    setLensLightSubtractionRunning(false);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`Lens light subtraction stop: ${error.message}`);
    setActivity("Lens light subtraction stop failed.", false);
    maskLensLightStop.disabled = !maskState.lensLightJobRunning;
  }
}

function cutoutDisplayBounds() {
  if (!imageView.cutoutCenter || !imageView.original) return null;
  const size = Math.max(1, sourcePixelsToCanvasPixels(Number(cutoutSize.value) || 1));
  const half = Math.floor(size / 2);
  let x0 = Math.max(0, Math.round(imageView.cutoutCenter.x) - half);
  let x1 = Math.min(imagePreviewCanvas.width, x0 + size);
  x0 = Math.max(0, x1 - size);
  let y0 = Math.max(0, Math.round(imageView.cutoutCenter.y) - half);
  let y1 = Math.min(imagePreviewCanvas.height, y0 + size);
  y0 = Math.max(0, y1 - size);
  const targetSize = Math.max(1, Math.round(Number(cutoutSize.value) || 1));
  return { x0, x1, y0, y1, width: x1 - x0, height: y1 - y0, targetWidth: targetSize, targetHeight: targetSize };
}

function fullImageDisplayBounds() {
  return {
    x0: 0,
    x1: imagePreviewCanvas.width,
    y0: 0,
    y1: imagePreviewCanvas.height,
    width: imagePreviewCanvas.width,
    height: imagePreviewCanvas.height,
    full_image: true,
  };
}

function currentSavedCutoutDisplayBounds() {
  const activeBounds = cutoutDisplayBounds();
  if (activeBounds) return activeBounds;

  const savedHeight = Number(cutoutState.shape?.[0]);
  const savedWidth = Number(cutoutState.shape?.[1]);
  if (
    Number.isFinite(savedWidth) &&
    Number.isFinite(savedHeight) &&
    savedWidth > 0 &&
    savedHeight > 0 &&
    imagePreviewCanvas.width === savedWidth &&
    imagePreviewCanvas.height === savedHeight
  ) {
    return { x0: 0, y0: 0, width: savedWidth, height: savedHeight };
  }

  const bounds = cutoutState.bounds;
  const sourceShape = bounds?.source_shape;
  if (
    bounds &&
    Array.isArray(sourceShape) &&
    imageSourceHeight() === Number(sourceShape[0]) &&
    (Number(imageView.sourceShape?.[1]) || imagePreviewCanvas.width * imageDisplayStride()) === Number(sourceShape[1])
  ) {
    return dataBoundsToPreviewCanvasBounds(bounds);
  }

  return null;
}

function renderCurrentMtfCutoutToCanvas(canvas) {
  const bounds = currentSavedCutoutDisplayBounds();
  if (!bounds || bounds.width <= 0 || bounds.height <= 0) return false;
  if (
    bounds.x0 < 0 ||
    bounds.y0 < 0 ||
    bounds.x0 + bounds.width > imagePreviewCanvas.width ||
    bounds.y0 + bounds.height > imagePreviewCanvas.height
  ) {
    return false;
  }
  const targetWidth = Math.max(1, Math.round(Number(bounds.targetWidth || bounds.width)));
  const targetHeight = Math.max(1, Math.round(Number(bounds.targetHeight || bounds.height)));
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, targetWidth, targetHeight);
  ctx.drawImage(
    imagePreviewCanvas,
    bounds.x0,
    bounds.y0,
    bounds.width,
    bounds.height,
    0,
    0,
    targetWidth,
    targetHeight,
  );
  renderCanvasAxes(canvas, currentSourcePixelScaleArcsec());
  return true;
}

function renderCutoutPreviewFromCanvas() {
  if (cutoutPreviewPanel.hidden) return;
  renderCurrentMtfCutoutToCanvas(cutoutPreviewCanvas);
}

function renderSavedCutoutPreview(url) {
  if (!url) {
    renderCutoutPreviewFromCanvas();
    return;
  }
  const img = new Image();
  img.onload = () => {
    cutoutPreviewCanvas.width = img.naturalWidth;
    cutoutPreviewCanvas.height = img.naturalHeight;
    const ctx = cutoutPreviewCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cutoutPreviewCanvas.width, cutoutPreviewCanvas.height);
    ctx.drawImage(img, 0, 0);
    renderCanvasAxes(cutoutPreviewCanvas, currentSourcePixelScaleArcsec());
  };
  img.onerror = () => {
    appendLog("Cutout preview: failed to load saved preview; using current canvas crop.");
    renderCutoutPreviewFromCanvas();
  };
  img.src = cacheBustUrl(url);
}

async function fitGaussianAtCutoutCenter() {
  if (!imageView.sourcePath) {
    appendLog("Gaussian fit: load an image first.");
    return;
  }
  if (!imageView.cutoutCenter) {
    appendLog("Gaussian fit: click the image to set a center first.");
    return;
  }

  try {
    const sourcePoint = sourceDisplayPointFromCanvas(imageView.cutoutCenter);
    const data = await callBackend("/api/image-preprocess/gaussian-fit", {
      path: imageView.sourcePath,
      x: Math.round(sourcePoint.x),
      y: Math.round(sourcePoint.y),
      size: Number(cutoutSize.value),
    });
    imageView.cutoutCenter = canvasPointFromSourceDisplay(data.center_display);
    updateCutoutMarker();
    appendLog(
      `Gaussian fit center: x=${Math.round(data.center_display.x)}, y=${Math.round(data.center_display.y)}.`
    );
  } catch (error) {
    appendLog(`Gaussian fit: ${error.message}`);
  }
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

function mtfImageData(imageData, mtfValues = mtfParameters()) {
  const output = new ImageData(
    new Uint8ClampedArray(imageData.data),
    imageData.width,
    imageData.height,
  );
  const { shadows, midtones, highlights } = mtfValues;
  const range = imageDataFiniteRange(imageData);
  for (let i = 0; i < output.data.length; i += 4) {
    output.data[i] = Math.round(255 * histogramTransformValue(normalizePixelValue(output.data[i], range), shadows, midtones, highlights));
    output.data[i + 1] = Math.round(255 * histogramTransformValue(normalizePixelValue(output.data[i + 1], range), shadows, midtones, highlights));
    output.data[i + 2] = Math.round(255 * histogramTransformValue(normalizePixelValue(output.data[i + 2], range), shadows, midtones, highlights));
  }
  return output;
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
  mtfMidtonesValue.textContent = `${midtones.toFixed(3)} / ${Number(mtfMidtones.max).toFixed(3)}`;
  mtfHighlightsValue.textContent = highlights.toFixed(3);
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
  if (imageView.rawPayload) {
    drawImagePayloadToCanvas(imagePreviewCanvas, imageView.rawPayload, { mtf: mtfParameters() });
    renderCutoutPreviewFromCanvas();
    renderProjectCanvasAxes();
    return;
  }
  if (!imageView.original) {
    return;
  }

  const ctx = imagePreviewCanvas.getContext("2d");
  ctx.putImageData(mtfImageData(imageView.original, mtfParameters()), 0, 0);
  renderCutoutPreviewFromCanvas();
  renderProjectCanvasAxes();
}

function resetMtf() {
  resetSharedMtf();
}

function sliderMtfParameters(shadowsInput, midtonesInput, highlightsInput) {
  let shadows = Number(shadowsInput.value);
  const midtones = Number(midtonesInput.value);
  let highlights = Number(highlightsInput.value);
  if (highlights <= shadows + 0.005) {
    highlights = Math.min(1, shadows + 0.005);
    highlightsInput.value = String(highlights);
  }
  if (highlights <= shadows) {
    shadows = Math.max(0, highlights - 0.005);
    shadowsInput.value = String(shadows);
  }
  return { shadows, midtones, highlights };
}

function euclidMtfParameters() {
  return sliderMtfParameters(euclidMtfShadows, euclidMtfMidtones, euclidMtfHighlights);
}

function sharedMtfValues(source) {
  if (source === "euclid") return euclidMtfParameters();
  return mtfParameters();
}

function normalizeMtfValues(values = {}) {
  return {
    shadows: Number.isFinite(Number(values.shadows)) ? Number(values.shadows) : 0,
    midtones: Number.isFinite(Number(values.midtones)) ? Number(values.midtones) : 0.5,
    highlights: Number.isFinite(Number(values.highlights)) ? Number(values.highlights) : 1,
  };
}

function setSharedMtfValues(values) {
  const normalized = normalizeMtfValues(values);
  const shadows = String(normalized.shadows);
  const midtones = String(normalized.midtones);
  const highlights = String(normalized.highlights);
  ensureMidtonesSliderCanRepresent(mtfMidtones, normalized.midtones);
  ensureMidtonesSliderCanRepresent(euclidMtfMidtones, normalized.midtones);
  mtfShadows.value = shadows;
  mtfMidtones.value = midtones;
  mtfHighlights.value = highlights;
  euclidMtfShadows.value = shadows;
  euclidMtfMidtones.value = midtones;
  euclidMtfHighlights.value = highlights;
}

function setSharedAutoMtfValues(values) {
  const normalized = normalizeMtfValues(values);
  const shadows = String(normalized.shadows);
  const highlights = String(normalized.highlights);
  mtfShadows.value = shadows;
  mtfHighlights.value = highlights;
  euclidMtfShadows.value = shadows;
  euclidMtfHighlights.value = highlights;
  setMidtonesSliderAuto(mtfMidtones, normalized.midtones);
  setMidtonesSliderAuto(euclidMtfMidtones, normalized.midtones);
}

function setMaskSubtractionMtfValues(values) {
  const normalized = normalizeMtfValues(values);
  ensureMidtonesSliderCanRepresent(maskSubtractionMtfMidtones, normalized.midtones);
  maskSubtractionMtfShadows.value = String(normalized.shadows);
  maskSubtractionMtfMidtones.value = String(normalized.midtones);
  maskSubtractionMtfHighlights.value = String(normalized.highlights);
  updateMaskSubtractionMtfLabels();
}

function setMaskCutoutMtfValues(values) {
  const normalized = normalizeMtfValues(values);
  ensureMidtonesSliderCanRepresent(maskCutoutMtfMidtones, normalized.midtones);
  maskCutoutMtfShadows.value = String(normalized.shadows);
  maskCutoutMtfMidtones.value = String(normalized.midtones);
  maskCutoutMtfHighlights.value = String(normalized.highlights);
  updateMaskCutoutMtfLabels();
}

function applySharedMtfValues(values, options = {}) {
  setSharedMtfValues(values);
  renderSharedMtfPreviews(options);
}

function renderSharedMtfPreviews(options = {}) {
  renderMtfPreview();
  renderEuclidPreview();
  void renderPsfFullImageMtfPreview();
  if (options.save !== false) {
    scheduleProjectSave();
    scheduleCurrentMtfThumbnailSave();
    scheduleCurrentMtfCutoutPreviewSave();
  }
}

function syncSharedMtfFrom(source) {
  setSharedMtfValues(sharedMtfValues(source));
  markProjectThumbnailDirty();
  renderSharedMtfPreviews();
}

function defaultSharedMtfValues() {
  let midtones = null;
  if (imageView.rawPayload?.data?.length) {
    const height = Number(imageView.rawPayload.shape?.[0] || imageView.rawPayload.data.length || 0);
    const width = Number(imageView.rawPayload.shape?.[1] || imageView.rawPayload.data[0]?.length || 0);
    midtones = autoMidtonesFromRows(
      imageView.rawPayload.data,
      width,
      height,
      DEFAULT_MTF_VALUES,
      MTF_DATA_DISPLAY_TARGET,
    );
  } else if (imageView.original) {
    midtones = autoMidtonesFromImageData(imageView.original, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET);
  } else if (euclidState.data?.length) {
    midtones = autoMidtonesFromRows(
      euclidState.data,
      euclidState.width,
      euclidState.height,
      DEFAULT_MTF_VALUES,
      MTF_DATA_DISPLAY_TARGET,
    );
  }
  return {
    ...DEFAULT_MTF_VALUES,
    midtones: Number.isFinite(midtones) ? midtones : DEFAULT_MTF_VALUES.midtones,
  };
}

function resetSharedMtf() {
  setSharedAutoMtfValues(defaultSharedMtfValues());
  markProjectThumbnailDirty();
  renderSharedMtfPreviews();
}

function updateEuclidMtfLabels() {
  const { shadows, midtones, highlights } = euclidMtfParameters();
  euclidMtfShadowsValue.textContent = shadows.toFixed(3);
  euclidMtfMidtonesValue.textContent = `${midtones.toFixed(3)} / ${Number(euclidMtfMidtones.max).toFixed(3)}`;
  euclidMtfHighlightsValue.textContent = highlights.toFixed(3);
}

function currentEuclidZeropoint() {
  const system = euclidMagSystem.value || "ab";
  const raw = euclidState.zeropoints?.[system];
  if (raw === null || raw === undefined || raw === "") return null;
  const zp = Number(raw);
  return Number.isFinite(zp) ? zp : null;
}

function updateEuclidZeropointLabel() {
  const system = euclidMagSystem.value || "ab";
  const zp = currentEuclidZeropoint();
  const source = euclidState.zeropointSource?.[system] || "none";
  euclidZeropoint.textContent = zp === null ? `${system.toUpperCase()} ZP: none` : `${system.toUpperCase()} ZP: ${zp.toFixed(4)} (${source})`;
}

function updateEuclidDisplayModeButton() {
  const surfaceMode = euclidState.displayMode === "surface_brightness";
  euclidDisplayMode.classList.toggle("active", surfaceMode);
  euclidDisplayMode.textContent = surfaceMode ? "Show flux image" : "Show mag/arcsec^2";
}

function euclidPixelAreaArcsec2() {
  const scale = Number(euclidState.pixelScaleArcsec);
  return Number.isFinite(scale) && scale > 0 ? scale * scale : null;
}

function euclidSurfaceBrightness(value) {
  const zp = currentEuclidZeropoint();
  const area = euclidPixelAreaArcsec2();
  if (zp === null || area === null || value <= 0) return null;
  return zp - 2.5 * Math.log10(value / area);
}

function percentileSorted(sortedValues, fraction) {
  if (!sortedValues.length) return null;
  const index = Math.min(sortedValues.length - 1, Math.max(0, fraction * (sortedValues.length - 1)));
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (index - lo);
}

function jetColor(value) {
  const t = Math.min(1, Math.max(0, value));
  const clip = (x) => Math.min(1, Math.max(0, x));
  return [
    Math.round(255 * clip(1.5 - Math.abs(4 * t - 3))),
    Math.round(255 * clip(1.5 - Math.abs(4 * t - 2))),
    Math.round(255 * clip(1.5 - Math.abs(4 * t - 1))),
  ];
}

function updateEuclidColorbar(visible, low = null, high = null) {
  euclidColorbar.style.display = visible ? "grid" : "none";
  euclidColorbarTicks.innerHTML = "";
  if (!visible) return;
  if (low === null || high === null || high <= low) return;
  const tickCount = 6;
  for (let i = 0; i < tickCount; i += 1) {
    const fraction = i / (tickCount - 1);
    const value = high - fraction * (high - low);
    const tick = document.createElement("div");
    tick.className = "euclid-colorbar-tick";
    tick.style.top = `${100 * fraction}%`;
    tick.innerHTML = `<span class="euclid-colorbar-mark"></span><span class="euclid-colorbar-label">${value.toFixed(2)}</span>`;
    euclidColorbarTicks.append(tick);
  }
}

function euclidManualSbBounds(autoLow, autoHigh) {
  let low = autoLow;
  let high = autoHigh;
  const lowerText = euclidSbLower.value.trim();
  const upperText = euclidSbUpper.value.trim();
  const manualLow = lowerText === "" ? NaN : Number(lowerText);
  const manualHigh = upperText === "" ? NaN : Number(upperText);
  if (Number.isFinite(manualLow)) low = manualLow;
  if (Number.isFinite(manualHigh)) high = manualHigh;
  return { low, high };
}

function refreshCutoutImageSelect(activeKey = euclidState.currentImageKey) {
  cutoutImageSelect.innerHTML = "";
  const entries = Object.entries(euclidState.imageOptions);
  if (!entries.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No cutout loaded";
    cutoutImageSelect.append(option);
    cutoutImageSelect.disabled = true;
    return;
  }
  entries.forEach(([key, entry]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = entry.label || key;
    if (key === activeKey) option.selected = true;
    cutoutImageSelect.append(option);
  });
  cutoutImageSelect.disabled = false;
}

function registerCutoutImage(key, label, image) {
  if (!key || !image) return;
  euclidState.imageOptions[key] = {
    label,
    image: {
      ...image,
      image_key: key,
      label,
    },
  };
  refreshCutoutImageSelect(key);
}

const EUCLID_BAND_OPTIONS = [
  { band: "VIS", key: "euclid-vis", label: "Euclid VIS" },
  { band: "NIR_Y", key: "euclid-y", label: "Euclid Y" },
  { band: "NIR_J", key: "euclid-j", label: "Euclid J" },
  { band: "NIR_H", key: "euclid-h", label: "Euclid H" },
];

function euclidSiblingFitsPath(path, band) {
  const text = String(path || "");
  if (!text) return "";
  if (/_(VIS|NIR_Y|NIR_J|NIR_H)\.fits$/i.test(text)) {
    return text.replace(/_(VIS|NIR_Y|NIR_J|NIR_H)\.fits$/i, `_${band}.fits`);
  }
  return "";
}

async function loadEuclidFitsFamily(seedPath, preferredBand = "VIS") {
  const seedText = String(seedPath || "");
  if (!seedText) return null;
  const loaded = {};
  for (const option of EUCLID_BAND_OPTIONS) {
    const path = euclidSiblingFitsPath(seedText, option.band);
    if (!path) continue;
    try {
      const data = await callBackend("/api/euclid-cutout/load-fits", { path });
      const image = {
        ...data.image,
        image_key: option.key,
        label: option.label,
        band: option.band,
        source: "euclid",
      };
      registerCutoutImage(option.key, option.label, image);
      loaded[option.band] = image;
    } catch (error) {
      appendLog(`Euclid ${option.band} restore skipped: ${error.message}`);
    }
  }
  return loaded[preferredBand] || loaded.VIS || Object.values(loaded)[0] || null;
}

function clearEuclidImagePreview(message = "No cutout loaded.") {
  euclidState.data = null;
  euclidState.width = 0;
  euclidState.height = 0;
  euclidState.vmin = 0;
  euclidState.vmax = 1;
  euclidState.zeropoint = null;
  euclidState.zeropoints = { ab: null, vega: null };
  euclidState.zeropointSource = {};
  euclidState.fluxUnit = "";
  euclidState.pixelScaleArcsec = null;
  euclidState.band = "";
  euclidState.source = "";
  euclidState.currentImageKey = "";
  euclidState.path = "";
  euclidState.imageOptions = {};
  euclidState.lastPoint = null;
  euclidPreviewCanvas.width = 1;
  euclidPreviewCanvas.height = 1;
  euclidPreviewCanvas.style.display = "none";
  euclidPreviewEmpty.style.display = "block";
  euclidPreviewEmpty.textContent = message;
  euclidPreviewMeta.textContent = message;
  euclidApertureOverlay.style.display = "none";
  euclidAnnulusOverlay.style.display = "none";
  updateEuclidColorbar(false);
  renderEuclidAxes();
  refreshCutoutImageSelect("");
}

function applyEuclidPreviewScale() {
  const scale = Math.max(25, Math.min(1000, Number(euclidPreviewScale.value) || 100));
  euclidPreviewScale.value = String(scale);
  euclidPreviewScaleValue.textContent = `${scale}%`;
  if (!euclidState.width || !euclidState.height) return;
  const width = Math.round(euclidState.width * scale / 100);
  const height = Math.round(euclidState.height * scale / 100);
  euclidPreviewCanvas.style.width = `${width}px`;
  euclidPreviewCanvas.style.height = `${height}px`;
  euclidPlotFrame.style.width = `${width}px`;
  euclidPlotFrame.style.height = `${height}px`;
  renderEuclidAxes();
  if (euclidState.lastPoint) updateEuclidApertureOverlay(euclidState.lastPoint);
}

function currentEuclidPreviewScale() {
  return (Math.max(25, Math.min(1000, Number(euclidPreviewScale.value) || 100))) / 100;
}

function setEuclidPreviewScale(scale) {
  const percent = Math.round(100 * clampZoomScale(scale, 0.25, 10));
  euclidPreviewScale.value = String(percent);
}

function niceAxisStep(range) {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const target = range / 5;
  const exponent = Math.floor(Math.log10(target));
  const base = target / (10 ** exponent);
  const niceBase = base <= 1 ? 1 : base <= 2 ? 2 : base <= 5 ? 5 : 10;
  return niceBase * (10 ** exponent);
}

function formatAxisValue(value, step) {
  if (Math.abs(value) < 1e-9) return "0";
  if (Math.abs(step) < 0.1) return value.toFixed(2);
  if (Math.abs(step) < 1) return value.toFixed(1);
  return value.toFixed(0);
}

function removePreviewAxes(canvas) {
  if (canvas?._previewAxisFrame) {
    canvas._previewAxisFrame.remove();
    canvas._previewAxisFrame = null;
  }
}

function canvasUsesInsetPreviewAxes(canvas) {
  return canvas === maskSubtractionPreview
    || canvas === lensScriptCutoutCanvas
    || canvas === lensScriptSubtractedCanvas;
}

function renderCanvasAxes(canvas, pixelScaleArcsec) {
  const scale = Number(pixelScaleArcsec);
  if (!canvas || !Number.isFinite(scale) || scale <= 0 || !canvas.width || !canvas.height || canvas.hidden) {
    removePreviewAxes(canvas);
    return;
  }
  const parent = canvas.parentElement;
  if (!parent) return;
  const canvasRect = canvas.getBoundingClientRect();
  const parentRect = parent.getBoundingClientRect();
  if (canvasRect.width <= 1 || canvasRect.height <= 1 || parentRect.width <= 1 || parentRect.height <= 1) {
    removePreviewAxes(canvas);
    return;
  }

  if (!canvas._previewAxisFrame) {
    const frame = document.createElement("div");
    frame.className = "preview-axis-frame";
    frame.setAttribute("aria-hidden", "true");
    frame.innerHTML = `
      <div class="preview-axis preview-axis-x"></div>
      <div class="preview-axis preview-axis-y"></div>
      <div class="preview-axis-title preview-axis-title-x">arcsec</div>
      <div class="preview-axis-title preview-axis-title-y">arcsec</div>
    `;
    parent.appendChild(frame);
    canvas._previewAxisFrame = frame;
  }

  const frame = canvas._previewAxisFrame;
  frame.classList.toggle("preview-axis-inset", canvasUsesInsetPreviewAxes(canvas));
  frame.style.left = `${canvasRect.left - parentRect.left}px`;
  frame.style.top = `${canvasRect.top - parentRect.top}px`;
  frame.style.width = `${canvasRect.width}px`;
  frame.style.height = `${canvasRect.height}px`;

  const axisX = frame.querySelector(".preview-axis-x");
  const axisY = frame.querySelector(".preview-axis-y");
  axisX.innerHTML = "";
  axisY.innerHTML = "";

  const xHalfRange = 0.5 * canvas.width * scale;
  const yHalfRange = 0.5 * canvas.height * scale;
  const xStep = niceAxisStep(2 * xHalfRange);
  const yStep = niceAxisStep(2 * yHalfRange);

  for (let value = Math.ceil(-xHalfRange / xStep) * xStep; value <= xHalfRange + 1e-9; value += xStep) {
    const fraction = (value + xHalfRange) / (2 * xHalfRange);
    const tick = document.createElement("div");
    tick.className = "preview-axis-tick preview-axis-tick-x";
    tick.style.left = `${100 * fraction}%`;
    tick.innerHTML = `<span class="preview-axis-mark"></span><span class="preview-axis-label">${formatAxisValue(value, xStep)}</span>`;
    axisX.append(tick);
  }

  for (let value = Math.ceil(-yHalfRange / yStep) * yStep; value <= yHalfRange + 1e-9; value += yStep) {
    const fraction = 1 - ((value + yHalfRange) / (2 * yHalfRange));
    const tick = document.createElement("div");
    tick.className = "preview-axis-tick preview-axis-tick-y";
    tick.style.top = `${100 * fraction}%`;
    tick.innerHTML = `<span class="preview-axis-label">${formatAxisValue(value, yStep)}</span><span class="preview-axis-mark"></span>`;
    axisY.append(tick);
  }
}

function renderProjectCanvasAxes() {
  const scale = currentImagePixelScaleArcsec();
  renderCanvasAxes(imagePreviewCanvas, imageView.original ? scale : null);
  renderCanvasAxes(cutoutPreviewCanvas, cutoutPreviewPanel.hidden ? null : scale);
  renderCanvasAxes(maskImageCanvas, maskState.data && !maskState.useMatplotlibPreview ? scale : null);
  renderCanvasAxes(maskSubtractionPreview, null);
  renderCanvasAxes(lensScriptCutoutCanvas, null);
  renderCanvasAxes(lensScriptSubtractedCanvas, null);
}

function renderEuclidAxes() {
  euclidAxisX.innerHTML = "";
  euclidAxisY.innerHTML = "";
  const scale = Number(euclidState.pixelScaleArcsec);
  if (!euclidState.width || !euclidState.height || !Number.isFinite(scale) || scale <= 0) return;

  const xHalfRange = 0.5 * euclidState.width * scale;
  const yHalfRange = 0.5 * euclidState.height * scale;
  const xStep = niceAxisStep(2 * xHalfRange);
  const yStep = niceAxisStep(2 * yHalfRange);

  for (let value = Math.ceil(-xHalfRange / xStep) * xStep; value <= xHalfRange + 1e-9; value += xStep) {
    const fraction = (value + xHalfRange) / (2 * xHalfRange);
    const tick = document.createElement("div");
    tick.className = "euclid-axis-tick euclid-axis-tick-x";
    tick.style.left = `${100 * fraction}%`;
    tick.innerHTML = `<span class="euclid-axis-mark"></span><span class="euclid-axis-label">${Math.abs(value) < 1e-9 ? "0" : value.toFixed(Math.abs(xStep) < 1 ? 1 : 0)}</span>`;
    euclidAxisX.append(tick);
  }

  for (let value = Math.ceil(-yHalfRange / yStep) * yStep; value <= yHalfRange + 1e-9; value += yStep) {
    const fraction = 1 - ((value + yHalfRange) / (2 * yHalfRange));
    const tick = document.createElement("div");
    tick.className = "euclid-axis-tick euclid-axis-tick-y";
    tick.style.top = `${100 * fraction}%`;
    tick.innerHTML = `<span class="euclid-axis-label">${Math.abs(value) < 1e-9 ? "0" : value.toFixed(Math.abs(yStep) < 1 ? 1 : 0)}</span><span class="euclid-axis-mark"></span>`;
    euclidAxisY.append(tick);
  }
}

function loadEuclidImagePayload(image, options = {}) {
  const rows = image.data || [];
  euclidState.height = Number(image.shape?.[0] || rows.length || 0);
  euclidState.width = Number(image.shape?.[1] || rows[0]?.length || 0);
  euclidState.data = rows;
  euclidState.vmin = Number(image.vmin ?? 0);
  euclidState.vmax = Number(image.vmax ?? 1);
  euclidState.zeropoints = {
    ab: image.zeropoints?.ab ?? image.zeropoint ?? null,
    vega: image.zeropoints?.vega ?? null,
  };
  euclidState.zeropointSource = image.zeropoint_source || (image.zeropoint === undefined ? {} : { ab: "legacy project" });
  euclidState.fluxUnit = image.flux_unit || "";
  euclidState.pixelScaleArcsec = Number(image.pixel_scale_arcsec) || null;
  euclidState.band = image.band || "";
  euclidState.source = image.source || "";
  euclidState.currentImageKey = image.image_key || euclidState.currentImageKey;
  euclidState.zeropoint = currentEuclidZeropoint();
  euclidState.path = image.path || "";
  euclidPreviewCanvas.width = euclidState.width;
  euclidPreviewCanvas.height = euclidState.height;
  euclidPreviewCanvas.style.display = "block";
  euclidPreviewEmpty.style.display = "none";
  const bandText = euclidState.band ? `, band ${euclidState.band}` : "";
  const scaleText = euclidState.pixelScaleArcsec ? `, ${euclidState.pixelScaleArcsec.toFixed(3)} arcsec/pix` : "";
  const imageLabel = image.label ? `${image.label}: ` : "";
  euclidPreviewMeta.textContent = `${imageLabel}${PathName(euclidState.path)}  ${euclidState.width}x${euclidState.height}${bandText}${euclidState.fluxUnit ? `, ${euclidState.fluxUnit}` : ""}${scaleText}`;
  updateEuclidZeropointLabel();
  updateEuclidDisplayModeButton();
  refreshCutoutImageSelect(euclidState.currentImageKey);
  if (options.autoMtf !== false) {
    applySharedAutoMidtones(autoMidtonesFromRows(euclidState.data, euclidState.width, euclidState.height, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET), { resetBounds: true });
  }
  applyEuclidPreviewScale();
  renderEuclidPreview();
}

function renderEuclidPreview() {
  updateEuclidMtfLabels();
  if (!euclidState.data || !euclidState.width || !euclidState.height) return;
  const output = new ImageData(euclidState.width, euclidState.height);
  if (euclidState.displayMode === "surface_brightness") {
    const sbValues = [];
    for (let y = 0; y < euclidState.height; y += 1) {
      const row = euclidState.data[y] || [];
      for (let x = 0; x < euclidState.width; x += 1) {
        const sb = euclidSurfaceBrightness(Number(row[x]));
        if (sb !== null && Number.isFinite(sb)) sbValues.push(sb);
      }
    }
    sbValues.sort((a, b) => a - b);
    const autoLow = sbValues.length ? sbValues[0] : null;
    const autoHigh = percentileSorted(sbValues, 0.98);
    const { low, high } = euclidManualSbBounds(autoLow, autoHigh);
    if (low === null || high === null || high <= low) {
      updateEuclidColorbar(false);
      for (let i = 0; i < output.data.length; i += 4) output.data[i + 3] = 255;
    } else {
      for (let y = 0; y < euclidState.height; y += 1) {
        const row = euclidState.data[y] || [];
        for (let x = 0; x < euclidState.width; x += 1) {
          const sb = euclidSurfaceBrightness(Number(row[x]));
          const i = (y * euclidState.width + x) * 4;
          if (sb === null || !Number.isFinite(sb)) {
            output.data[i] = 0;
            output.data[i + 1] = 0;
            output.data[i + 2] = 0;
            output.data[i + 3] = 255;
            continue;
          }
          const normalized = 1 - Math.min(1, Math.max(0, (sb - low) / (high - low)));
          const [r, g, b] = jetColor(normalized);
          output.data[i] = r;
          output.data[i + 1] = g;
          output.data[i + 2] = b;
          output.data[i + 3] = 255;
        }
      }
      updateEuclidColorbar(true, low, high);
    }
    const ctx = euclidPreviewCanvas.getContext("2d");
    ctx.putImageData(output, 0, 0);
    return;
  }
  updateEuclidColorbar(false);
  const { shadows, midtones, highlights } = euclidMtfParameters();
  const range = imageRowsFiniteRange(euclidState.data, euclidState.width, euclidState.height);
  for (let y = 0; y < euclidState.height; y += 1) {
    const row = euclidState.data[y] || [];
    for (let x = 0; x < euclidState.width; x += 1) {
      const value = Number(row[x]);
      const normalized = normalizePixelValue(value, range);
      const gray = Math.round(255 * histogramTransformValue(normalized, shadows, midtones, highlights));
      const i = (y * euclidState.width + x) * 4;
      output.data[i] = gray;
      output.data[i + 1] = gray;
      output.data[i + 2] = gray;
      output.data[i + 3] = 255;
    }
  }
  const ctx = euclidPreviewCanvas.getContext("2d");
  ctx.putImageData(output, 0, 0);
}

function resetEuclidMtf() {
  resetSharedMtf();
}

function euclidPixelFromEvent(event) {
  const rect = euclidPreviewCanvas.getBoundingClientRect();
  if (!euclidState.data || rect.width <= 0 || rect.height <= 0) return null;
  const x = ((event.clientX - rect.left) / rect.width) * euclidState.width;
  const y = ((event.clientY - rect.top) / rect.height) * euclidState.height;
  if (x < 0 || y < 0 || x >= euclidState.width || y >= euclidState.height) return null;
  return { x, y };
}

function updateEuclidApertureOverlay(point) {
  if (!point || !euclidState.photometryEnabled) {
    euclidApertureOverlay.style.display = "none";
    euclidAnnulusOverlay.style.display = "none";
    return;
  }
  const rect = euclidPreviewCanvas.getBoundingClientRect();
  const stageRect = document.querySelector(".euclid-stage").getBoundingClientRect();
  const radius = Math.max(0.5, Number(euclidApertureRadius.value) || 2);
  const annulusWidth = Math.max(0, Number(euclidAnnulusWidth.value) || 0);
  const outerRadius = radius + annulusWidth;
  const left = rect.left - stageRect.left + (point.x / euclidState.width) * rect.width;
  const top = rect.top - stageRect.top + (point.y / euclidState.height) * rect.height;
  const pixelScale = rect.width / euclidState.width;
  const apertureDiameter = 2 * radius * pixelScale;
  const annulusDiameter = 2 * outerRadius * pixelScale;
  euclidApertureOverlay.style.left = `${left}px`;
  euclidApertureOverlay.style.top = `${top}px`;
  euclidApertureOverlay.style.width = `${apertureDiameter}px`;
  euclidApertureOverlay.style.height = `${apertureDiameter}px`;
  euclidApertureOverlay.style.display = "block";
  euclidAnnulusOverlay.style.left = `${left}px`;
  euclidAnnulusOverlay.style.top = `${top}px`;
  euclidAnnulusOverlay.style.width = `${annulusDiameter}px`;
  euclidAnnulusOverlay.style.height = `${annulusDiameter}px`;
  euclidAnnulusOverlay.style.display = annulusWidth > 0 ? "block" : "none";
}

function euclidPhotometryAt(point) {
  const apertureRadius = Math.max(0.5, Number(euclidApertureRadius.value) || 2);
  const annulusWidth = Math.max(0, Number(euclidAnnulusWidth.value) || 0);
  const outerRadius = apertureRadius + annulusWidth;
  const cx = point.x;
  const cy = point.y;
  const x0 = Math.max(0, Math.floor(cx - outerRadius));
  const x1 = Math.min(euclidState.width - 1, Math.ceil(cx + outerRadius));
  const y0 = Math.max(0, Math.floor(cy - outerRadius));
  const y1 = Math.min(euclidState.height - 1, Math.ceil(cy + outerRadius));
  let apertureFlux = 0;
  let aperturePixels = 0;
  const bgValues = [];

  for (let y = y0; y <= y1; y += 1) {
    const row = euclidState.data[y] || [];
    for (let x = x0; x <= x1; x += 1) {
      const value = Number(row[x]);
      if (!Number.isFinite(value)) continue;
      const dist = Math.hypot(x - cx, y - cy);
      if (dist <= apertureRadius) {
        apertureFlux += value;
        aperturePixels += 1;
      } else if (annulusWidth > 0 && dist <= outerRadius) {
        bgValues.push(value);
      }
    }
  }

  const sortedBgValues = [...bgValues].sort((a, b) => a - b);
  const mid = Math.floor(sortedBgValues.length / 2);
  const bgMedian = annulusWidth > 0 && sortedBgValues.length
    ? sortedBgValues.length % 2
      ? sortedBgValues[mid]
      : 0.5 * (sortedBgValues[mid - 1] + sortedBgValues[mid])
    : 0;
  const flux = apertureFlux - bgMedian * aperturePixels;
  const zp = currentEuclidZeropoint();
  const magnitude = zp !== null && flux > 0 ? zp - 2.5 * Math.log10(flux) : null;
  const apertureAreaArcsec2 = euclidPixelAreaArcsec2() === null ? null : aperturePixels * euclidPixelAreaArcsec2();
  const surfaceBrightness = magnitude !== null && apertureAreaArcsec2 !== null && apertureAreaArcsec2 > 0
    ? magnitude + 2.5 * Math.log10(apertureAreaArcsec2)
    : null;
  return { aperturePixels, bgPixels: bgValues.length, bgMedian, apertureFlux, flux, magnitude, apertureAreaArcsec2, surfaceBrightness, backgroundSubtracted: annulusWidth > 0 };
}

function updateEuclidPhotometry(event) {
  const point = euclidPixelFromEvent(event);
  if (point) euclidState.lastPoint = point;
  updateEuclidApertureOverlay(point);
  if (!point) {
    euclidPhotometryReadout.textContent = "Move mouse over the cutout.";
    return;
  }
  if (!euclidState.photometryEnabled) {
    euclidPhotometryReadout.textContent = `x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}`;
    return;
  }
  const result = euclidPhotometryAt(point);
  const magSystem = (euclidMagSystem.value || "ab").toUpperCase();
  const magText =
    result.flux <= 0
      ? `${magSystem} mag: flux<=0`
      : result.magnitude === null
        ? `${magSystem} mag: no valid ZP`
        : `${magSystem} mag=${result.magnitude.toFixed(4)}`;
  const sbText = result.surfaceBrightness === null ? "" : `, ${magSystem} SB=${result.surfaceBrightness.toFixed(4)} mag/arcsec^2`;
  euclidPhotometryReadout.textContent = `x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}, ${magText}${sbText}, flux=${result.flux.toExponential(3)}, aperture_sum=${result.apertureFlux.toExponential(3)}, annulus_median=${result.bgMedian.toExponential(3)}, annulus_n=${result.bgPixels}`;
}

function selectedPhotozBands() {
  const selected = euclidPhotozBandInputs
    .filter((input) => input.checked)
    .map((input) => input.dataset.photozBand || "")
    .filter((band) => PHOTOZ_SUPPORTED_BANDS.has(band));
  euclidState.photozBands = new Set(selected);
  return euclidState.photozBands;
}

function applyPhotozBandSelection(bands, options = {}) {
  const requested = Array.isArray(bands) && bands.length ? new Set(bands.filter((band) => PHOTOZ_SUPPORTED_BANDS.has(band))) : new Set(PHOTOZ_BAND_LIST);
  euclidPhotozBandInputs.forEach((input) => {
    input.checked = requested.has(input.dataset.photozBand || "");
  });
  selectedPhotozBands();
  if (options.save !== false) scheduleProjectSave();
}

function photozImagePayloads() {
  const seen = new Set();
  selectedPhotozBands();
  return Object.values(euclidState.imageOptions)
    .map((entry) => entry.image)
    .filter((image) => image?.path && PHOTOZ_SUPPORTED_BANDS.has(image.band || ""))
    .filter((image) => {
      const key = `${image.band}:${image.path}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .map((image) => ({
      path: image.path,
      band: image.band || "",
      label: image.label || image.band || PathName(image.path),
      source: image.source || "",
    }));
}

function photozSummaryText(data) {
  const photometry = data.photometry || [];
  const fluxParts = photometry.map((item) => {
    if (item.error) return `${item.label || item.band_key}: ${item.error}`;
    const flux = Number(item.flux);
    return `${item.label || item.band_key}: ${Number.isFinite(flux) ? flux.toExponential(3) : "nan"} nJy`;
  });
  const formatRun = (key, fallbackLabel) => {
    const run = data.photoz_results?.[key] || null;
    const label = run?.label || fallbackLabel;
    const photoz = run?.photoz || null;
    const zKeys = photoz ? Object.keys(photoz).filter((name) => /^Z/i.test(name) && Number.isFinite(Number(photoz[name]))) : [];
    const preferredKey = zKeys.find((name) => /MEDIAN|MODE|MEAN|Q50|BEST/i.test(name)) || zKeys[0];
    const bands = run?.used_bands?.length ? ` [${run.used_bands.join(",")}]` : "";
    if (photoz && preferredKey) return `${label}${bands}: ${preferredKey}=${Number(photoz[preferredKey]).toFixed(4)}`;
    return `${label}${bands}: failed (${run?.error || "not enough valid bands"})`;
  };
  const zText = [
    formatRun("euclid_only", "Euclid only"),
    formatRun("euclid_plus_ground", "Euclid + DESI/DECam"),
  ].join("; ");
  return `${zText}; fluxes: ${fluxParts.join(", ")}`;
}

async function runPhotozAtClick(event) {
  if (!euclidState.photozEnabled) return;
  const point = euclidPixelFromEvent(event);
  if (!point || !euclidState.path) return;
  euclidState.lastPoint = point;
  const images = photozImagePayloads();
  if (!images.length) {
    euclidPhotometryReadout.textContent = "Photo-z needs at least one supported FITS image.";
    return;
  }
  updateEuclidApertureOverlay(point);
  try {
    setActivity("Running pixel photo-z", true);
    euclidPhotometryReadout.textContent = `Running photo-z at x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}...`;
    const data = await callBackend("/api/photoz/measure", {
      reference_path: euclidState.path,
      x: point.x,
      y: point.y,
      aperture_radius: Number(euclidApertureRadius.value),
      annulus_width: Number(euclidAnnulusWidth.value),
      images,
      photoz_bands: Array.from(selectedPhotozBands()),
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    const summary = photozSummaryText(data);
    euclidPhotometryReadout.textContent = `x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}, RA=${data.skycoord_deg.ra.toFixed(6)}, Dec=${data.skycoord_deg.dec.toFixed(6)}, ${summary}`;
    appendLog(`Photo-z: ${summary}`);
    setActivity("Photo-z completed", false);
  } catch (error) {
    euclidPhotometryReadout.textContent = `Photo-z failed: ${error.message}`;
    appendLog(`Photo-z: ${error.message}`);
    setActivity(`Photo-z failed: ${error.message}`, false);
  }
}

async function pollEuclidJob(jobId) {
  euclidState.jobId = jobId;
  for (let i = 0; i < 1800; i += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    try {
      const projectFolderQuery = projectState.folder ? `?project_folder=${encodeURIComponent(projectState.folder)}` : "";
      const data = await callBackend(`/api/euclid-cutout/job/${jobId}${projectFolderQuery}`);
      updateLogProgress(`Euclid job ${jobId}`, data.progress, data.state);
      setActivity(data.progress?.message || data.message || `Euclid job ${data.state}`, data.state !== "completed" && data.state !== "failed");
      if (data.rms_map_path) {
        euclidState.rmsMapPath = data.rms_map_path;
      }
      if (data.rms_bundle_path) {
        euclidState.rmsBundlePath = data.rms_bundle_path;
      }
      if (data.images) {
        Object.entries(data.images).forEach(([band, image]) => {
          const label = image.label || `Euclid ${band}`;
          const key = `euclid-${String(band).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
          registerCutoutImage(key, label, { ...image, image_key: key, label });
        });
        const activeBand = data.active_band || "VIS";
        const activeKey = `euclid-${String(activeBand).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
        const activeImage = euclidState.imageOptions[activeKey]?.image || Object.values(euclidState.imageOptions).find((entry) => entry.image?.source === "euclid")?.image;
        if (activeImage) {
          loadEuclidImagePayload(activeImage);
          applyRegisteredImageArtifacts(data, { showCutoutPanel: true });
          setImagePreviewFromPayload(activeImage, data.preview_url || "", { resetArtifacts: false });
          const registeredPath = data.original_path || activeImage.path;
          if (registeredPath) {
            inputImagePath.value = registeredPath;
            selectedImageName.textContent = registeredPath;
          }
          scheduleCurrentMtfThumbnailSave();
          scheduleCurrentMtfCutoutPreviewSave();
        }
        scheduleProjectSave();
      } else if (data.image) {
        const euclidImage = { ...data.image, image_key: "euclid-vis", label: "Euclid VIS" };
        registerCutoutImage("euclid-vis", "Euclid VIS", euclidImage);
        loadEuclidImagePayload(euclidImage);
        applyRegisteredImageArtifacts(data, { showCutoutPanel: true });
        setImagePreviewFromPayload(euclidImage, data.preview_url || "", { resetArtifacts: false });
        const registeredPath = data.original_path || euclidImage.path;
        if (registeredPath) {
          inputImagePath.value = registeredPath;
          selectedImageName.textContent = registeredPath;
        }
        scheduleCurrentMtfThumbnailSave();
        scheduleCurrentMtfCutoutPreviewSave();
        scheduleProjectSave();
      }
      if (data.state === "completed" || data.state === "failed") {
        appendLog(`Euclid job ${jobId}: ${data.state} - ${data.message || ""}`);
        if (data.rms_map_path) {
          appendLog(`Euclid VIS RMS saved: ${data.rms_map_path}`);
        }
        setActivity(`Euclid job ${jobId}: ${data.state}`, false);
        return;
      }
    } catch (error) {
      appendLog(`Euclid job ${jobId}: ${error.message}`);
      setActivity(`Euclid job failed: ${error.message}`, false);
      return;
    }
  }
  appendLog(`Euclid job ${jobId}: still running.`);
}

async function runEuclidCutout() {
  euclidRun.disabled = true;
  try {
    autoProjectNameFromCoordinates({ save: false, prefix: "Euclid" });
    if (!projectState.folder) {
      await saveProjectNow();
    }
    const payload = {
      ra: Number(euclidRa.value),
      dec: Number(euclidDec.value),
      cutout_size: Number(euclidCutoutSize.value),
      username: euclidUsername.value.trim(),
      password: euclidPassword.value,
      project_id: projectState.id,
      project_name: canonicalProjectName(projectState),
      project_folder: projectState.folder,
    };
    appendLog(`Euclid cutout: requesting RA=${payload.ra}, Dec=${payload.dec}, radius=${payload.cutout_size} arcsec.`);
    setActivity("Submitting Euclid cutout job", true);
    const data = await callBackend("/api/euclid-cutout/run", payload);
    appendLog(data.message || "Euclid cutout job started.");
    pollEuclidJob(data.job_id);
  } catch (error) {
    appendLog(`Euclid cutout: ${error.message}`);
    setActivity(`Euclid cutout failed: ${error.message}`, false);
  } finally {
    euclidRun.disabled = false;
  }
}

async function runLegacyCutout() {
  legacyRun.disabled = true;
  try {
    autoProjectNameFromCoordinates({ save: false, prefix: "", keepEuclidPrefix: true });
    if (!projectState.folder) {
      await saveProjectNow();
    }
    const payload = {
      ra: Number(euclidRa.value),
      dec: Number(euclidDec.value),
      cutout_size: Number(euclidCutoutSize.value),
      bands: "griz",
      project_id: projectState.id,
      project_name: canonicalProjectName(projectState),
      project_folder: projectState.folder,
    };
    appendLog(`DESI cutout: requesting RA=${payload.ra}, Dec=${payload.dec}, radius=${payload.cutout_size} arcsec.`);
    setActivity("Downloading DESI Legacy Survey cutout", true);
    const data = await callBackend("/api/legacy-cutout/run", payload);
    const legacyImages = data.images || {};
    Object.entries(legacyImages).forEach(([band, image]) => {
      const key = `desi-${band}`;
      registerCutoutImage(key, `DESI ${band}`, { ...image, image_key: key, label: `DESI ${band}` });
    });
    const activeKey = data.active_band ? `desi-${data.active_band}` : Object.keys(euclidState.imageOptions).find((key) => key.startsWith("desi-"));
    const activeImage = euclidState.imageOptions[activeKey]?.image;
    if (activeImage) loadEuclidImagePayload(activeImage);
    appendLog(data.message || "Loaded DESI Legacy Survey cutout.");
    setActivity("Loaded DESI Legacy Survey cutout", false);
    scheduleProjectSave();
  } catch (error) {
    appendLog(`DESI cutout: ${error.message}`);
    setActivity(`DESI cutout failed: ${error.message}`, false);
  } finally {
    legacyRun.disabled = false;
  }
}

document.querySelectorAll(".nav-tab").forEach((button) => {
  button.addEventListener("click", () => {
    setPage(button.dataset.page);
    scheduleProjectSave();
  });
});

document.querySelector("#clear-log").addEventListener("click", () => {
  runLog.textContent = "Ready.";
});

euclidRun.addEventListener("click", runEuclidCutout);
legacyRun.addEventListener("click", runLegacyCutout);
euclidGenerateProjectName.addEventListener("click", generateEuclidProjectName);
euclidRa.addEventListener("input", () => autoProjectNameFromCoordinates());
euclidDec.addEventListener("input", () => autoProjectNameFromCoordinates());
euclidResetMtf.addEventListener("click", resetEuclidMtf);
[euclidMtfShadows, euclidMtfMidtones, euclidMtfHighlights].forEach((slider) => {
  slider.addEventListener("input", () => syncSharedMtfFrom("euclid"));
});
function handleMaskCutoutMtfChanged() {
  updateMaskCutoutMtfLabels();
  scheduleProjectSave();
}

function applyMaskCutoutMtf(options = {}) {
  updateMaskCutoutMtfLabels();
  if (maskState.useMatplotlibPreview) {
    maskStatus.textContent = "Applying Mask Preview display...";
    void refreshMaskMatplotlibPreview();
  } else if (maskPreviewSourceKind() === "image") {
    refreshVisibleMaskPreview();
  }
  renderLensScriptCutoutPreview();
  scheduleLensScriptCutoutPreviewRefresh(20);
  scheduleProjectSave();
  if (!options.silent) appendLog(`Mask: applied Cutout MTF (${maskState.previewColormap}).`);
}

function applyMaskCutoutAutoMtf() {
  let autoMidtones = autoMidtonesFromImagePayload(maskState.sourceImage, MTF_DATA_DISPLAY_TARGET);
  if (!Number.isFinite(autoMidtones)) {
    autoMidtones = autoMidtonesFromImagePayload(imageView.rawPayload, MTF_DATA_DISPLAY_TARGET);
  }
  if (!Number.isFinite(autoMidtones) && imageView.original) {
    autoMidtones = autoMidtonesFromImageData(imageView.original, DEFAULT_MTF_VALUES, MTF_DATA_DISPLAY_TARGET);
  }
  if (!Number.isFinite(autoMidtones)) {
    appendLog("Mask: cannot compute Cutout Auto MTF because no image data is loaded.");
    return;
  }
  applyMaskCutoutAutoMidtones(autoMidtones);
  applyMaskCutoutMtf({ silent: true });
  appendLog(`Mask: applied Cutout Auto MTF, midtones ${formatSliderNumber(maskCutoutMtfMidtones, maskCutoutMtfMidtones.value)}.`);
}

function handleMaskSubtractionMtfChanged() {
  updateMaskSubtractionMtfLabels();
  scheduleProjectSave();
}

function applyMaskSubtractionMtf(options = {}) {
  updateMaskSubtractionMtfLabels();
  if (maskState.useMatplotlibPreview) {
    maskStatus.textContent = "Applying Mask Preview display...";
    void refreshMaskMatplotlibPreview();
  } else if (maskPreviewSourceKind() === "subtracted") {
    refreshVisibleMaskPreview();
  }
  renderLensScriptCutoutPreview();
  scheduleLensScriptCutoutPreviewRefresh(20);
  scheduleProjectSave();
  if (!options.silent) appendLog(`Mask: applied Lens light MTF (${maskState.previewColormap}).`);
}

function applyMaskSubtractionAutoMtf() {
  const autoMidtones = autoMidtonesFromImagePayload(maskState.lensLightImage, MTF_DATA_DISPLAY_TARGET);
  if (!Number.isFinite(autoMidtones)) {
    appendLog("Mask: cannot compute Lens light Auto MTF because no lens-light image data is loaded.");
    return;
  }
  applyMaskSubtractionAutoMidtones(autoMidtones);
  applyMaskSubtractionMtf({ silent: true });
  appendLog(`Mask: applied Lens light Auto MTF, midtones ${formatSliderNumber(maskSubtractionMtfMidtones, maskSubtractionMtfMidtones.value)}.`);
}

function bindMaskMtfValueInput(input, slider, onChange, onApply) {
  if (!input || !slider) return;
  const applyValue = ({ clamp = false } = {}) => {
    const raw = String(input.value || "").trim();
    if (!raw) return;
    let value = Number(raw);
    if (!Number.isFinite(value)) return;
    const min = Number(slider.min);
    const max = Number(slider.max);
    if (clamp) {
      value = Math.min(max, Math.max(min, value));
    } else if (value < min || value > max) {
      return;
    }
    slider.value = String(value);
    onChange();
  };
  input.addEventListener("input", () => applyValue());
  input.addEventListener("change", () => applyValue({ clamp: true }));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      applyValue({ clamp: true });
      if (onApply) onApply();
      input.blur();
    }
  });
}

[maskCutoutMtfShadows, maskCutoutMtfMidtones, maskCutoutMtfHighlights].forEach((slider) => {
  slider.addEventListener("input", handleMaskCutoutMtfChanged);
});
[
  [maskCutoutMtfShadowsValue, maskCutoutMtfShadows],
  [maskCutoutMtfMidtonesValue, maskCutoutMtfMidtones],
  [maskCutoutMtfHighlightsValue, maskCutoutMtfHighlights],
].forEach(([input, slider]) => bindMaskMtfValueInput(input, slider, handleMaskCutoutMtfChanged, applyMaskCutoutMtf));
maskCutoutMtfAuto?.addEventListener("click", applyMaskCutoutAutoMtf);
maskCutoutMtfApply?.addEventListener("click", () => applyMaskCutoutMtf());

[maskSubtractionMtfShadows, maskSubtractionMtfMidtones, maskSubtractionMtfHighlights].forEach((slider) => {
  slider.addEventListener("input", handleMaskSubtractionMtfChanged);
});
[
  [maskSubtractionMtfShadowsValue, maskSubtractionMtfShadows],
  [maskSubtractionMtfMidtonesValue, maskSubtractionMtfMidtones],
  [maskSubtractionMtfHighlightsValue, maskSubtractionMtfHighlights],
].forEach(([input, slider]) => bindMaskMtfValueInput(input, slider, handleMaskSubtractionMtfChanged, applyMaskSubtractionMtf));
maskSubtractionMtfAuto?.addEventListener("click", applyMaskSubtractionAutoMtf);
maskSubtractionMtfApply?.addEventListener("click", () => applyMaskSubtractionMtf());
[maskLensLightNGauss, maskLensLightSigmaMin, maskLensLightCenterMaxOffset, maskLensLightSviSteps, maskLensLightBgCorner].forEach((input) => {
  input.addEventListener("input", scheduleProjectSave);
});
maskLensLightSigmaMax.addEventListener("input", () => {
  delete maskLensLightSigmaMax.dataset.dynamicDefault;
  scheduleProjectSave();
});
euclidMagSystem.addEventListener("change", () => {
  euclidState.zeropoint = currentEuclidZeropoint();
  updateEuclidZeropointLabel();
  renderEuclidPreview();
  scheduleProjectSave();
});
cutoutImageSelect.addEventListener("change", () => {
  const image = euclidState.imageOptions[cutoutImageSelect.value]?.image;
  if (!image) return;
  loadEuclidImagePayload(image);
  scheduleProjectSave();
});
euclidDisplayMode.addEventListener("click", () => {
  euclidState.displayMode = euclidState.displayMode === "surface_brightness" ? "flux" : "surface_brightness";
  updateEuclidDisplayModeButton();
  renderEuclidPreview();
  scheduleProjectSave();
});
[euclidSbLower, euclidSbUpper].forEach((input) => {
  input.addEventListener("input", () => {
    renderEuclidPreview();
    scheduleProjectSave();
  });
});
euclidPreviewScale.addEventListener("input", () => {
  applyEuclidPreviewScale();
  scheduleProjectSave();
});
euclidStage.addEventListener("wheel", (event) => {
  const zoomed = zoomCanvasAtPointer({
    event,
    canvas: euclidPreviewCanvas,
    hasImage: Boolean(euclidState.data),
    getScale: currentEuclidPreviewScale,
    setScale: setEuclidPreviewScale,
    minScale: 0.25,
    maxScale: 10,
    applyTransform: applyEuclidPreviewScale,
    nudgeView: (dx, dy) => {
      euclidStage.scrollLeft -= dx;
      euclidStage.scrollTop -= dy;
    },
  });
  if (zoomed) scheduleProjectSave();
}, { passive: false });
euclidPhotometryToggle.addEventListener("click", () => {
  euclidState.photometryEnabled = !euclidState.photometryEnabled;
  euclidPhotometryToggle.classList.toggle("active", euclidState.photometryEnabled);
  euclidPhotometryToggle.textContent = euclidState.photometryEnabled ? "测光 On" : "测光";
  if (!euclidState.photometryEnabled) {
    euclidApertureOverlay.style.display = "none";
    euclidAnnulusOverlay.style.display = "none";
  }
});
euclidPhotozToggle.addEventListener("click", () => {
  euclidState.photozEnabled = !euclidState.photozEnabled;
  euclidPhotozToggle.classList.toggle("active", euclidState.photozEnabled);
  euclidPhotozToggle.textContent = euclidState.photozEnabled ? "测光红移 On" : "测光红移";
  if (euclidState.photozEnabled) {
    euclidState.photometryEnabled = true;
    euclidPhotometryToggle.classList.add("active");
    euclidPhotometryToggle.textContent = "测光 On";
    euclidPhotometryReadout.textContent = "Photo-z mode: click a pixel in the cutout preview.";
  }
  scheduleProjectSave();
});
euclidPhotozBandInputs.forEach((input) => {
  input.addEventListener("change", () => {
    const selected = Array.from(selectedPhotozBands());
    euclidPhotometryReadout.textContent = selected.length
      ? `Photo-z bands: ${selected.join(", ")}`
      : "Photo-z needs at least one selected band.";
    scheduleProjectSave();
  });
});
euclidPreviewCanvas.addEventListener("mousemove", updateEuclidPhotometry);
euclidPreviewCanvas.addEventListener("click", runPhotozAtClick);
euclidPreviewCanvas.addEventListener("mouseleave", () => {
  euclidApertureOverlay.style.display = "none";
  euclidAnnulusOverlay.style.display = "none";
  euclidPhotometryReadout.textContent = "Move mouse over the cutout.";
});

imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files[0];
  if (file) inputImagePath.value = "";
  selectedImageName.textContent = file ? `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)` : "No file selected";
});

rmsFileInput.addEventListener("change", () => {
  const file = rmsFileInput.files[0];
  if (file) inputRmsPath.value = "";
  selectedRmsName.textContent = file ? `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)` : "No RMS file selected";
});

loadRmsFile.addEventListener("click", async () => {
  const file = rmsFileInput.files[0];
  const directPath = inputRmsPath.value.trim();
  if (!directPath && !file) {
    appendLog("Image preprocess: enter an RMS path or choose an RMS FITS file first.");
    return;
  }

  loadRmsFile.disabled = true;
  appendLog(`Image preprocess: registering RMS ${directPath || file.name}`);
  try {
    const data = directPath ? await loadRmsFromPath(directPath) : await uploadRmsForLensModel(file);
    applyRegisteredRms(data);
  } catch (error) {
    appendLog(`Image preprocess RMS: ${error.message}`);
  } finally {
    loadRmsFile.disabled = false;
  }
});

loadImagePreview.addEventListener("click", async () => {
  const file = imageFileInput.files[0];
  const directPath = inputImagePath.value.trim();
  if (!directPath && !file) {
    appendLog("Image preprocess: enter an image path or choose an image file first.");
    return;
  }

  loadImagePreview.disabled = true;
  appendLog(`Image preprocess: loading ${directPath || file.name}`);
  try {
    if (directPath) {
      await loadImageFromPath(directPath, {
        registerDefaultCutout: !(cutoutState.fitsPath && isProjectImageArtifactPath(directPath)),
      });
    } else {
      const data = await uploadImageForPreview(file);
      imageView.sourcePath = data.uploaded_path || "";
      inputImagePath.value = imageView.sourcePath;
      if (!psfSciencePath.value.trim()) {
        psfSciencePath.value = imageView.sourcePath;
      }
      setImagePreviewMetadata(data);
      if (data.image_payload) {
        setImagePreviewFromPayload(data.image_payload, data.preview_url || "");
      } else {
        await setImagePreview(data.preview_url);
      }
      applyRegisteredImageArtifacts(data);
      if (imageInputGroup && imageCutoutGroup) {
        imageInputGroup.open = false;
        imageCutoutGroup.open = true;
      }
      scheduleCurrentMtfThumbnailSave();
      scheduleCurrentMtfCutoutPreviewSave();
      appendLog(data.message || "Image preprocess: backend preview loaded.");
      scheduleProjectSave();
    }
  } catch (error) {
    appendLog(`Image preprocess: ${error.message}`);
  } finally {
    loadImagePreview.disabled = false;
  }
});

psfModeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) {
      setPsfMode(input.value);
    }
  });
});

psfPreviewViewInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) {
      setPsfPreviewView(input.value);
    }
  });
});

psfLoadInput.addEventListener("click", loadInputPsfPreview);
psfDefaultEuclid.addEventListener("click", loadDefaultEuclidPsf);
psfDefaultEuclidY.addEventListener("click", () => loadDefaultEuclidNispPsf("Y", psfDefaultEuclidY));
psfDefaultEuclidJ.addEventListener("click", () => loadDefaultEuclidNispPsf("J", psfDefaultEuclidJ));
psfDefaultEuclidH.addEventListener("click", loadDefaultEuclidHPsf);
psfInputFile.addEventListener("change", () => {
  if (psfInputFile.files[0]) {
    loadInputPsfPreview();
  }
});
psfFindStars.addEventListener("click", findPsfStars);
psfRunFit.addEventListener("click", runPsfFit);
psfStopFit.addEventListener("click", stopPsfFit);
psfClearOutput.addEventListener("click", clearPsfOutput);
psfSelectedIds.addEventListener("input", syncPsfSelectedIdsFromInput);
maskToolInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) setMaskTool(input.value);
  });
  input.addEventListener("click", () => {
    if (input.checked) setMaskTool(input.value);
  });
});
maskModeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) setMaskMode(input.value);
  });
  input.addEventListener("click", () => {
    if (input.checked) setMaskMode(input.value);
  });
});
maskType.addEventListener("change", async () => {
  setMaskType(maskType.value);
  const loaded = await loadProjectMask(maskState.type, { silent: true });
  if (!loaded) {
    maskStatus.textContent = "Mask Preview requires Data_cutout.fits.";
  }
  renderMaskOverlay();
});
maskBrushRadius.addEventListener("input", () => {
  updateMaskCursor();
  scheduleProjectSave();
});
maskLoadImage.addEventListener("click", loadCurrentCutoutIntoMask);
maskClear.addEventListener("click", clearMaskData);
maskSave.addEventListener("click", saveMaskFits);
maskLensLightSemilinear?.addEventListener("click", () =>
  generateLensLightSubtractionScript(false, {
    actionLabel: "semilinear MGE lens-light fit",
    preferredResultChoice: "constrained",
    semilinearMgeSteps: 0,
    sviSteps: 0,
    updateStepControls: true,
  }),
);
maskLensLightSubtraction.addEventListener("click", () => generateLensLightSubtractionScript(false));
maskLensLightSubtractionGpu?.addEventListener("click", () => generateLensLightSubtractionScript(true));
maskLensLightStop.addEventListener("click", stopLensLightSubtractionJob);
maskLensLightResultChoice?.addEventListener("change", () => {
  maskState.lensLightResultChoice = maskLensLightResultChoice.value || "unconstrained";
  setMaskSubtractionPreview(maskState.lensLightPreviewUrl, maskState.lensLightJobStatus);
  void refreshMaskSubtractionMatplotlibPreview();
  if (maskPreviewSourceKind() === "subtracted") refreshVisibleMaskPreview();
  scheduleLensScriptCutoutPreviewRefresh();
  scheduleProjectSave();
});
maskConjugatePlane.addEventListener("change", () => {
  setConjugatePlane(maskConjugatePlane.value);
  scheduleProjectSave();
});
maskConjugateSource?.addEventListener("change", () => {
  setMaskPreviewSource(maskConjugateSource.value === "subtracted" ? "subtracted" : "image");
});
maskPreviewSource?.addEventListener("change", () => {
  setMaskPreviewSource(maskPreviewSource.value);
});
maskPreviewColormap?.addEventListener("change", () => {
  setMaskPreviewColormap(maskPreviewColormap.value, { refresh: false });
});
if (maskConjugateClickMode) {
  maskConjugateClickMode.addEventListener("change", () => {
    setConjugateClickMode(maskConjugateClickMode.value);
  });
}
maskConjugatePoint.addEventListener("click", () => setConjugateMode(!maskState.conjugateMode));
maskClearConjugate.addEventListener("click", clearConjugatePoints);
dsplEnabled.addEventListener("change", () => {
  syncDsplControls();
  if (dsplEnabled.checked) {
    autoProjectNameFromCoordinates({ prefix: "DSPL" });
  }
  scheduleProjectSave();
});
sourcePositiveInput?.addEventListener("change", syncSourceNonlinearPriorControls);
sourceLogBrightnessInput?.addEventListener("change", syncSourceNonlinearPriorControls);
startNewModel.addEventListener("click", startNewModelProject);
chooseProjectButton.addEventListener("click", openProjectChooser);
projectChooserClose.addEventListener("click", closeProjectChooser);
projectChooserModal.addEventListener("click", (event) => {
  if (event.target === projectChooserModal) closeProjectChooser();
});
projectChooserViewInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) setProjectChooserView(input.value);
  });
});
projectChooserSearch?.addEventListener("input", () => {
  projectChooserQuery = projectChooserSearch.value;
  renderProjectChooser(projectChooserProjects, { replace: false });
});
projectChooserScore?.addEventListener("change", () => {
  projectChooserMinimumScore = projectChooserScore.value || "all";
  renderProjectChooser(projectChooserProjects, { replace: false });
});
projectChooserSort?.addEventListener("change", () => {
  projectChooserSortMode = ["score", "updated", "name"].includes(projectChooserSort.value)
    ? projectChooserSort.value
    : "score";
  localStorage.setItem("herculens-project-sort", projectChooserSortMode);
  renderProjectChooser(projectChooserProjects, { replace: false });
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!projectChooserModal.hidden) closeProjectChooser();
  if (!settingsModal.hidden) closeSettingsPanel();
});
renameProjectButton.addEventListener("click", renameProject);
projectName.addEventListener("input", (event) => event.stopPropagation());
projectName.addEventListener("change", (event) => event.stopPropagation());
projectName.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    void renameProject();
  }
});
projectSelect.addEventListener("change", loadSelectedProject);
setProjectFolder.addEventListener("click", useImageFolderAsProjectFolder);
folderPath.addEventListener("change", async () => {
  const nextFolder = folderPath.value.trim();
  if (!nextFolder) return;
  projectState.folder = nextFolder;
  projectState.name = canonicalProjectName(projectState);
  projectName.value = projectState.name;
  await loadFolder(nextFolder);
  await saveProjectNow();
});

resetImageView.addEventListener("click", resetImageTransform);

cutoutSize.addEventListener("input", updateCutoutMarker);
cutoutSize.addEventListener("input", renderCutoutPreviewFromCanvas);
bgBoxSize.addEventListener("input", () => {
  updateBgBoxMarker();
  scheduleProjectSave();
});
clearBgBox.addEventListener("click", clearBackgroundBox);

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
    const sourcePoint = sourceDisplayPointFromCanvas(imageView.cutoutCenter);
    const data = await callBackend("/api/image-preprocess/cutout", {
      path: imageView.sourcePath,
      x: Math.round(sourcePoint.x),
      y: Math.round(sourcePoint.y),
      size: Number(cutoutSize.value),
      project_id: projectState.id,
      project_folder: projectState.folder,
    });
    const savedCutoutPath = data.project_data_path || data.fits_path || "";
    appendLog(`${data.message} -> ${savedCutoutPath}`);
    cutoutState.fitsPath = savedCutoutPath;
    imageView.dataCutoutPath = savedCutoutPath;
    cutoutState.previewUrl = data.preview_url || "";
    cutoutState.shape = data.shape || null;
    cutoutState.bounds = data.bounds || null;
    updateLensSigmaMaxAutoHint();
    cutoutPreviewLabel.textContent = PathName(savedCutoutPath);
    cutoutPreviewPanel.hidden = false;
    renderCutoutPreviewFromCanvas();
    await saveCurrentMtfCutoutPreviewNow();
    if (activePageId() === "mask") {
      await loadProjectMask(maskState.type, { silent: true });
    }
    void loadLensScriptCutoutPreview({ silent: true });
    scheduleProjectSave();
  } catch (error) {
    appendLog(`Cutout: ${error.message}`);
  } finally {
    saveCutout.disabled = false;
  }
});

function PathName(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function isProjectImageArtifactPath(path) {
  return /(?:^|[\\/])(Original_file|Data_cutout)\.fits$/i.test(String(path || ""));
}

function projectThumbnailUrl() {
  const folderName = PathName(projectState.folder || projectState.id || "");
  if (!folderName) return "";
  const baseUrl = `/runs/${encodeURIComponent(folderName)}/previews/Data_cutout_preview.png`;
  return projectThumbnailState.version ? `${baseUrl}?t=${encodeURIComponent(projectThumbnailState.version)}` : baseUrl;
}

function markProjectThumbnailDirty() {
  projectThumbnailState.dirty = true;
}

function resetProjectThumbnailCache(src = projectThumbnailUrl()) {
  projectThumbnailState.src = src || "";
  projectThumbnailState.image = null;
  projectThumbnailState.loading = null;
  projectThumbnailState.failed = false;
}

function loadProjectThumbnailImage(src = projectThumbnailUrl()) {
  if (!src) return Promise.reject(new Error("No project thumbnail URL."));
  if (projectThumbnailState.src !== src) {
    resetProjectThumbnailCache(src);
  }
  if (projectThumbnailState.image) return Promise.resolve(projectThumbnailState.image);
  if (projectThumbnailState.failed) return Promise.reject(new Error("Project thumbnail unavailable."));
  if (projectThumbnailState.loading) return projectThumbnailState.loading;
  projectThumbnailState.loading = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      projectThumbnailState.image = img;
      projectThumbnailState.loading = null;
      projectThumbnailState.failed = false;
      resolve(img);
    };
    img.onerror = () => {
      projectThumbnailState.loading = null;
      projectThumbnailState.failed = true;
      reject(new Error("Project thumbnail unavailable."));
    };
    img.src = src;
  });
  return projectThumbnailState.loading;
}

function drawProjectThumbnailRegion(canvas, bounds, targetWidth, targetHeight, onLoaded = null) {
  if (projectThumbnailState.dirty || !canvas || !bounds || !targetWidth || !targetHeight) return false;
  const src = projectThumbnailUrl();
  if (!src) return false;
  const draw = (img) => {
    const x0 = Math.round(Number(bounds.x0) || 0);
    const y0 = Math.round(Number(bounds.y0) || 0);
    const width = Math.round(Number(bounds.width ?? bounds.x1 - bounds.x0) || targetWidth);
    const height = Math.round(Number(bounds.height ?? bounds.y1 - bounds.y0) || targetHeight);
    if (x0 < 0 || y0 < 0 || width <= 0 || height <= 0 || x0 + width > img.naturalWidth || y0 + height > img.naturalHeight) {
      return false;
    }
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.drawImage(img, x0, y0, width, height, 0, 0, targetWidth, targetHeight);
    canvas.classList.add("visible");
    renderCanvasAxes(canvas, currentImagePixelScaleArcsec());
    return true;
  };
  if (projectThumbnailState.src === src && projectThumbnailState.image) {
    return draw(projectThumbnailState.image);
  }
  void loadProjectThumbnailImage(src)
    .then((img) => {
      if (projectThumbnailState.dirty || projectThumbnailUrl() !== src) return;
      if (draw(img) && typeof onLoaded === "function") onLoaded();
    })
    .catch(() => {});
  return false;
}

document.addEventListener("keydown", (event) => {
  const tag = event.target?.tagName?.toLowerCase();
  if (tag === "input" || tag === "textarea" || event.target?.isContentEditable) return;
  if (event.key.toLowerCase() === "g") {
    event.preventDefault();
    fitGaussianAtCutoutCenter();
  }
});

[mtfShadows, mtfMidtones, mtfHighlights].forEach((slider) => {
  slider.addEventListener("input", () => syncSharedMtfFrom("image"));
});

document.querySelector("#reset-mtf").addEventListener("click", resetMtf);

maskOverlayCanvas.addEventListener("pointerdown", (event) => {
  const point = maskPointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  if (maskState.conjugateMode) {
    void addConjugatePointFromMask(point, activeMaskConjugateSource());
    return;
  }
  maskOverlayCanvas.setPointerCapture(event.pointerId);
  const tool = checkedMaskTool();
  if (tool === "brush") {
    maskState.drawing = true;
    maskState.lastPoint = point;
    applyMaskBrush(point);
  } else {
    addMaskPolygonPoint(point);
  }
  renderMaskOverlay();
  if (tool === "line") {
    scheduleMaskAutoSave();
    scheduleProjectSave();
  }
});

maskSubtractionPreview.addEventListener("pointerdown", (event) => {
  if (!maskState.conjugateMode) return;
  event.preventDefault();
  if (maskState.conjugateMeasureSource !== "subtracted") {
    appendLog("Conjugate point: set Measure on to Lens light subtracted before clicking this preview.");
    return;
  }
  const displayedPoint = displayedCanvasPointFromEvent(maskSubtractionPreview, event);
  const point = nativeMaskPointFromCanvasPoint(displayedPoint, maskSubtractionPreview);
  if (!point) return;
  void addConjugatePointFromMask(point, "subtracted");
});

maskOverlayCanvas.addEventListener("pointermove", (event) => {
  if (!maskState.drawing || checkedMaskTool() !== "brush") return;
  const point = maskPointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  drawMaskLine(maskState.lastPoint || point, point);
  maskState.lastPoint = point;
  renderMaskOverlay();
});

maskOverlayCanvas.addEventListener("pointerup", (event) => {
  const wasDrawing = maskState.drawing;
  if (wasDrawing) {
    scheduleMaskAutoSave();
    scheduleProjectSave();
  }
  maskState.drawing = false;
  maskState.lastPoint = null;
  if (maskOverlayCanvas.hasPointerCapture(event.pointerId)) {
    maskOverlayCanvas.releasePointerCapture(event.pointerId);
  }
  if (wasDrawing) {
    renderMaskOverlay();
  }
});

maskOverlayCanvas.addEventListener("pointercancel", () => {
  const wasDrawing = maskState.drawing;
  maskState.drawing = false;
  maskState.lastPoint = null;
  if (wasDrawing) {
    renderMaskOverlay();
  }
});

window.addEventListener("resize", () => {
  fitMaskCanvasToStage();
  fitMaskSubtractionCanvasToStage();
  renderEuclidAxes();
  renderProjectCanvasAxes();
});

imageZoomStage.addEventListener("wheel", (event) => {
  zoomCanvasAtPointer({
    event,
    canvas: imagePreviewCanvas,
    hasImage: Boolean(imageView.original),
    getScale: () => imageView.scale,
    setScale: (scale) => {
      imageView.scale = scale;
    },
    minScale: 0.1,
    maxScale: 20,
    applyTransform: applyImageTransform,
    nudgeView: (dx, dy) => {
      imageView.x += dx;
      imageView.y += dy;
    },
  });
});

imageZoomStage.addEventListener("contextmenu", (event) => {
  if (!imageView.original) return;
  event.preventDefault();
  setBgBoxFromEvent(event);
});

attachPanControl(imagePanX, "x");
attachPanControl(imagePanY, "y");

imageZoomStage.addEventListener("pointerdown", (event) => {
  if (!imageView.original) return;
  if (event.button !== 0) return;
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
  if (!imageView.dragging) return;
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
  scheduleProjectSave();
});

settingsOpen.addEventListener("click", openSettingsPanel);
settingsClose.addEventListener("click", closeSettingsPanel);
settingsModal.addEventListener("click", (event) => {
  if (event.target === settingsModal) closeSettingsPanel();
});
settingsForm.addEventListener("submit", saveSettings);
document.querySelectorAll("[data-settings-check]").forEach((button) => {
  button.addEventListener("click", () => checkRuntimePython(button.dataset.settingsCheck));
});

lensChainSelect?.addEventListener("change", () => {
  lensState.selectedChain = lensChainSelect.value;
  renderSelectedLensModelPreview();
  scheduleProjectSave();
});

lensResultRating?.addEventListener("input", () => {
  lensState.rating = currentLensRating();
  lensResultRating.value = String(lensState.rating);
  scheduleProjectSave();
});

document.querySelector("#check-backend").addEventListener("click", async () => {
  appendLog("Checking backend via same-origin API.");
  try {
    const data = await callBackend("/api/health");
    setBackendState("ready", data.status || "Ready");
    appendLog("Backend check succeeded.");
    scheduleProjectSave();
  } catch (error) {
    setBackendState("error", "Unavailable");
    appendLog(`Backend check failed: ${error.message}`);
  }
});

openFolder.addEventListener("click", async () => {
  try {
    await loadFolder(currentProjectFolderPath());
    appendLog(`Opened folder: ${folderPath.value}`);
    await saveProjectNow();
  } catch (error) {
    appendLog(`Folder browser: ${error.message}`);
  }
});


lensGenerateCode.addEventListener("click", generateLensCode);
lensExtraMassAdd?.addEventListener("click", addLensPlaneMassComponent);
lensExtraMassList?.addEventListener("input", (event) => {
  if (event.target instanceof HTMLInputElement) {
    updateLensPlaneMassComponentFromControl(event.target);
    if (lensPlaneMassState.activePickerId) renderLensScriptCutoutPreview();
  }
});
lensExtraMassList?.addEventListener("change", (event) => {
  if (event.target instanceof HTMLSelectElement) {
    updateLensPlaneMassComponentFromControl(event.target);
  }
});
lensExtraMassList?.addEventListener("click", (event) => {
  const button = event.target.closest?.("[data-mass-action]");
  if (!button) return;
  const component = lensPlaneMassComponentFromElement(button);
  if (!component) return;
  if (button.dataset.massAction === "remove") {
    lensPlaneMassState.components = lensPlaneMassState.components.filter((entry) => entry.id !== component.id);
    if (lensPlaneMassState.activePickerId === component.id) {
      lensPlaneMassState.activePickerId = "";
      renderLensScriptCutoutPreview();
    }
    renderLensPlaneMassComponents();
    markLensScriptStale("Lens-plane mass component removed");
    scheduleProjectSave();
    requestAnimationFrame(syncLensCodePreviewHeight);
  } else if (button.dataset.massAction === "pick") {
    void toggleLensPlaneMassPositionPicker(component.id);
  }
});
lensScriptCutoutOverlay?.addEventListener("click", (event) => {
  if (lensPlaneMassState.activePickerId) void locateLensPlaneMassPosition(event);
});
lensSigmaMaxInput()?.addEventListener("input", updateLensSigmaMaxAutoHint);
lensClearCode.addEventListener("click", clearLensGeneratedCode);
lensEditCode.addEventListener("click", () => setLensScriptEditMode(!lensState.scriptEditMode));
lensStopRun.addEventListener("click", stopLensJob);
lensModelForm?.addEventListener("input", (event) => {
  if (projectState.restoring) return;
  if (event.target === lensLightExternalMode || event.target === lensLightExternalPath) {
    lensState.lensLightPriorAuto = false;
  }
  markLensScriptStale("Lens model settings changed");
});

document.querySelectorAll('input[name="source_pixel_grid_shape"], input[name="source2_pixel_grid_shape"]').forEach((input) => {
  input.addEventListener("input", () => {
    if (input.name === "source2_pixel_grid_shape") {
      lensPixelGridManual.source2 = true;
    } else {
      lensPixelGridManual.source = true;
    }
    scheduleProjectSave();
  });
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", async () => {
    const pageId = activePageId();
    const form = document.querySelector(`[data-form="${pageId}"]`);
    const endpoint = button.dataset.endpoint;
    const action = button.dataset.action;
    const isLensRun = pageId === "lens-model" && (action === "run" || action === "run-gpu" || action === "run-all-gpu");
    const isGpuLensRun = pageId === "lens-model" && (action === "run-gpu" || action === "run-all-gpu");
    const isAllGpuLensRun = pageId === "lens-model" && action === "run-all-gpu";
    const payload = isLensRun ? collectFormPayload(form) : null;

    if (isLensRun && lensState.jobRunning) {
      appendLog(`Lens model: job ${lensState.jobId || ""} is already running.`);
      return;
    }
    if (isLensRun && lensState.scriptStale) {
      const reason = lensState.scriptStaleReason || "Lens model settings changed";
      appendLog(`Lens model: regenerate code before running (${reason}).`);
      setActivity("Regenerate the lens model script before running", false);
      lensGenerateCode.focus();
      return;
    }
    if (isGpuLensRun) {
      const prompt = isAllGpuLensRun
        ? "Upload a slim project to Sciama and request Lens SVI on all known GPU partitions? This tries generic GPU, MIG, A100, and L40 candidates."
        : "Upload a slim project to Sciama and request GPU for Lens SVI? Priority: full A100, then full L40, then A100 fallback partitions.";
      const approved = window.confirm(prompt);
      if (!approved) return;
    }
    if (isLensRun) {
      applyLensScriptEditorIfNeeded();
      if (!lensState.generatedCode.trim()) {
        appendLog("Lens model: click Generate code before running.");
        setActivity("Generate the lens model script before running", false);
        return;
      }
      payload.generated_code = lensState.generatedCode;
      payload.generated_model_config = lensState.modelConfig || {};
      payload.generated_code_edited = Boolean(lensState.scriptEdited);
    }

    appendLog(`${pageCopy[pageId].title}: calling ${endpoint}`);
    if (isLensRun) {
      setLensRunState(true);
      setLensProgress({
        fraction: 0.02,
        message: isGpuLensRun ? "Submitting Sciama GPU lens model request..." : "Submitting lens model request...",
      });
      setActivity(isGpuLensRun ? "Submitting Sciama GPU lens model request" : "Submitting lens model request", true);
    }
    button.disabled = true;

    try {
      const data = await callBackend(endpoint, payload);
      updatePreview(pageId, data);
      appendLog(data.message || `${pageCopy[pageId].title}: request completed.`);
      if (isLensRun) {
        if (data.generated_code) {
          setLensGeneratedCode(data.generated_code, data.model_config || null);
        }
        if (data.job_id) {
          setLensProgress({
            fraction: 0.05,
            message: data.message || "Lens model job started.",
          });
          setActivity(data.message || "Lens model job started", true);
          pollLensJob(data.job_id);
        } else {
          setLensProgress({
            fraction: 1,
            message: data.message || "Lens model endpoint returned without a background job.",
          });
          setActivity(data.message || "Lens model request completed", false);
          setLensRunState(false);
        }
      }
      scheduleProjectSave();
    } catch (error) {
      if (isLensRun) {
        setLensProgress({
          fraction: 1,
          message: `Lens model failed: ${error.message}`,
        });
        setActivity(`Lens model failed: ${error.message}`, false);
        setLensRunState(false);
      }
      appendLog(`${pageCopy[pageId].title}: ${error.message}`);
    } finally {
      if (!isLensRun) {
        button.disabled = false;
      }
    }
  });
});

setPsfMode("input");
setPsfPreviewView("product");
projectChooserSortMode = ["score", "updated", "name"].includes(projectChooserSortMode)
  ? projectChooserSortMode
  : "score";
if (projectChooserSort) projectChooserSort.value = projectChooserSortMode;
if (projectChooserScore) projectChooserScore.value = projectChooserMinimumScore;
projectChooserViewInputs.forEach((input) => {
  input.checked = input.value === projectChooserView;
});
syncDsplControls();
syncSourceNonlinearPriorControls();
renderLensPlaneMassComponents();
setupLensCodeHeightSync();
applyTheme(localStorage.getItem("herculens-gui-theme") || "light");
updateMaskCutoutMtfLabels();
updateMaskSubtractionMtfLabels();
renderSharedMtfPreviews({ save: false });
updateLensSigmaMaxAutoHint();
updateEuclidDisplayModeButton();
refreshCutoutImageSelect();
updateWorkflowStatus();

document.addEventListener("input", scheduleProjectSave);
document.addEventListener("change", scheduleProjectSave);

async function initializeGui() {
  let backendConnected = false;
  try {
    setActivity("Loading project list", true);
    await loadSettings();
    backendConnected = true;
    setBackendState("ready", "Connected");
    await refreshProjectList();
    await loadLatestProject();
    setActivity("Loading data folder", true);
    await loadFolder(folderPath.value.trim());
    setActivity("Ready.", false);
  } catch (error) {
    setBackendState(backendConnected ? "ready" : "error", backendConnected ? "Connected" : "Unavailable");
    setActivity(`Startup failed: ${error.message}`, false);
    appendLog(`Startup: ${error.message}`);
  }
}

initializeGui();
