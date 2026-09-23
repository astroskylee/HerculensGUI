import argparse
import ast
import base64
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import shutil
import sys
import threading
import time
import textwrap
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, unquote, urlparse
from urllib.request import urlopen

from lens_code_generator import generate_lens_hmc_script, generate_lens_script

try:
    import cgi
except ModuleNotFoundError:
    cgi = None


GUI_ROOT = Path(__file__).resolve().parents[1]
SERVER_CWD = Path.cwd().resolve()
WEB_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/mnt/d/lensing").resolve()
OVERLEAF_PAPER_PATH = (
    WORKSPACE_ROOT / "Papers" / "Paper6_EuclidDR1DSPL_overleaf" / "Paper.tex"
)
OVERLEAF_CATALOG_TABLE_LABELS = ("tab:tier1_result", "tab:tier2_ra_dec")
OVERLEAF_CATALOG_MATCH_TOLERANCE_ARCSEC = 2.0
DEFAULT_DATA_DIR = (GUI_ROOT / ".." / "Data" / "Slicelens").resolve()
DEFAULT_FILE_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else SERVER_CWD
RUNS_ROOT = GUI_ROOT / "runs"
PROJECT_FOLDER_DIR = RUNS_ROOT
PROJECT_STATE_PATH = RUNS_ROOT / ".latest_project.json"
RECYCLE_BIN_DIR = Path("/mnt/d/lensing/RecycleBin")
SETTINGS_PATH = GUI_ROOT / "web_gui_settings.local.json"
LEGACY_SETTINGS_PATH = RUNS_ROOT / "web_gui_settings.local.json"
DEFAULT_EUCLID_PSF = WEB_ROOT / "defaults" / "Euclid_default_PSF.fits"
DEFAULT_EUCLID_Y_PSF = WEB_ROOT / "defaults" / "Euclid_default_Y_PSF.fits"
DEFAULT_EUCLID_J_PSF = WEB_ROOT / "defaults" / "Euclid_default_J_PSF.fits"
DEFAULT_EUCLID_H_PSF = WEB_ROOT / "defaults" / "Euclid_default_H_PSF.fits"
EUCLID_CUTOUT_RUNNER = WEB_ROOT / "euclid_cutout_runner.py"
IMAGE_EXTENSIONS = {".fits", ".fit", ".fts", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GUI_IMAGE_PAYLOAD_MAX_PIXELS = 1_000_000
HERCULENS_PYTHON_LEGACY = Path("/home/skylee/anaconda3/envs/herculens/bin/python")
EUCLID_CUTOUT_SCRIPT = Path("/mnt/d/lensing/codex_agent/user_script/download-euclid-data/euclid_cutout_single.py")
EUCLID_RMS_BUNDLE_SCRIPT = Path("/mnt/d/lensing/codex_agent/user_script/download-euclid-data/download_vis_errormap_psf_bundle.py")
EUCLID_PYTHON_LEGACY = Path("/mnt/d/lensing/.venv_euclid/bin/python")
PHOSPHOROS_ROOT = Path(os.getenv("PHOSPHOROS_ROOT", str(Path.home() / "Phosphoros"))).expanduser()
PHOSPHOROS_PYTHON_LEGACY = Path("/home/skylee/anaconda3/envs/phosphoros311m/bin/python")
EUCLID_VIS_DEFAULT_AB_ZP = 24.5
MICROJY_AB_ZP = 23.9
LEGACY_AB_ZP = 22.5
LEGACY_DEFAULT_PIXSCALE = 0.262
DESI_DR1_FP_A = 1.17
DESI_DR1_FP_B_LOGI = -0.803
DESI_DR1_FP_C_KPC_PLANCK18 = -0.094
DESI_R_BAND_SOLAR_AB_MAG = 4.65
DESI_FP_EVOLUTION_Q = 1.1
EUCLID_BAND_LABELS = {
    "VIS": "Euclid VIS",
    "NIR_Y": "Euclid Y",
    "NIR_J": "Euclid J",
    "NIR_H": "Euclid H",
}
EUCLID_BANDS = tuple(EUCLID_BAND_LABELS)
GROUND_BANDS = ("g", "r", "i", "z")
PHOTOZ_FILTERS = {
    "VIS": ("Euclid/VIS.vis", "FLUX_VIS", "FLUXERR_VIS"),
    "NIR_Y": ("Euclid/NISP.Y", "FLUX_Y", "FLUXERR_Y"),
    "NIR_J": ("Euclid/NISP.J", "FLUX_J", "FLUXERR_J"),
    "NIR_H": ("Euclid/NISP.H", "FLUX_H", "FLUXERR_H"),
    "g": ("CTIO/DECam/DECam.g", "FLUX_G", "FLUXERR_G"),
    "r": ("CTIO/DECam/DECam.r", "FLUX_R", "FLUXERR_R"),
    "i": ("CTIO/DECam/DECam.i", "FLUX_I", "FLUXERR_I"),
    "z": ("CTIO/DECam/DECam.z", "FLUX_Z", "FLUXERR_Z"),
}
PHOTOZ_BAND_ORDER = ["g", "r", "i", "z", "VIS", "NIR_Y", "NIR_J", "NIR_H"]
EUCLID_PHOTOZ_BANDS = set(EUCLID_BANDS)
SUPPORTED_PHOTOZ_BANDS = set(PHOTOZ_FILTERS)
LENS_TERMINAL_STATES = {"completed", "failed", "stopped"}
PSF_TERMINAL_STATES = {"completed", "failed", "stopped"}
LENS_LIGHT_TERMINAL_STATES = {"completed", "failed", "stopped"}
LENS_MODEL_RESULT_DIRNAME = "lens_model_result"
LENS_LIGHT_RESULT_DIRNAME = "lens_light_subtraction_result"
LENS_MODEL_SCRIPT_NAME = "run_lens_svi.py"
LENS_HMC_SCRIPT_NAME = "run_lens_hmc.py"
SCIAMA_GPU_RUNNER = WEB_ROOT / "sciama_gpu_runner.py"
SCIAMA_REMOTE_BASE = "/users/tianli/HerculensGUI_svi_tests"
SCIAMA_CONDA_ACTIVATE = "/mnt/lustre2/shared_conda/envs/tianli/herculens_tian/bin/activate"
SCIAMA_GPU_PARTITIONS = ("gpu.q", "sciama5.q", "sciama5-5.q")
SCIAMA_GPU_ALL_PARTITIONS = ",".join(SCIAMA_GPU_PARTITIONS)
SCIAMA_GPU_REQUEST_CANDIDATES = (
    {"partition": "gpu.q", "gres": "gpu:A100:1", "label": "A100 full GPU on gpu.q"},
    {"partition": "gpu.q", "gres": "gpu:L40:1", "label": "L40 full GPU on gpu.q"},
    {"partition": "sciama5.q", "gres": "gpu:A100:1", "label": "A100 full GPU on sciama5.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:A100:1", "label": "A100 full GPU on sciama5-5.q"},
)
SCIAMA_ALL_GPU_REQUEST_CANDIDATES = (
    {"partition": "gpu.q", "gres": "gpu:A100:1", "label": "A100 full GPU on gpu.q"},
    {"partition": "sciama5.q", "gres": "gpu:A100:1", "label": "A100 full GPU on sciama5.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:A100:1", "label": "A100 full GPU on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:L40:1", "label": "L40 full GPU on gpu.q"},
    {"partition": "gpu.q", "gres": "gpu:3g.20gb:2", "label": "A100 MIG 3g.20gb x2 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:3g.20gb:2", "label": "A100 MIG 3g.20gb x2 on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:3g.20gb:1", "label": "A100 MIG 3g.20gb x1 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:3g.20gb:1", "label": "A100 MIG 3g.20gb x1 on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:2g.10gb:4", "label": "A100 MIG 2g.10gb x4 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:2g.10gb:4", "label": "A100 MIG 2g.10gb x4 on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:2g.10gb:2", "label": "A100 MIG 2g.10gb x2 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:2g.10gb:2", "label": "A100 MIG 2g.10gb x2 on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:2g.10gb:1", "label": "A100 MIG 2g.10gb x1 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:2g.10gb:1", "label": "A100 MIG 2g.10gb x1 on sciama5-5.q"},
    {"partition": "gpu.q", "gres": "gpu:1g.5gb:1", "label": "A100 MIG 1g.5gb x1 on gpu.q"},
    {"partition": "sciama5-5.q", "gres": "gpu:1g.5gb:1", "label": "A100 MIG 1g.5gb x1 on sciama5-5.q"},
    {"partition": SCIAMA_GPU_ALL_PARTITIONS, "gres": "gpu:1", "label": "any single GPU on all known GPU partitions"},
)
SCIAMA_SLIM_FILES = (
    LENS_MODEL_SCRIPT_NAME,
    LENS_HMC_SCRIPT_NAME,
    "Tian_infra.py",
    "semilinear_solver.py",
    "theta_e_solver.py",
    "custom_gibbs.py",
    "Data_cutout.fits",
    "RMS_map.fits",
    "PSF_model.fits",
    "mask_1.fits",
    "mask_2.fits",
    "mask_out.fits",
    "lens_light_subtraction_result/kwargs_lens_light.pkl",
    "project.json",
    "project_artifacts.json",
)
ORIGINAL_FILE_NAME = "Original_file.fits"
DATA_CUTOUT_NAME = "Data_cutout.fits"
PROJECT_LIST_CACHE_TTL = 60.0
PROJECT_LIST_CACHE_LOCK = threading.Lock()
PROJECT_LIST_CACHE = {"signature": None, "projects": [], "cached_at": 0.0}
LENS_LIGHT_SCRIPT_NAME = "run_lens_light_subtraction.py"
LENS_LIGHT_RUNNER_NAME = "run_constrained_jax_mge_svi.py"
LENS_LIGHT_SCIAMA_SLIM_FILES = (
    LENS_LIGHT_SCRIPT_NAME,
    LENS_LIGHT_RUNNER_NAME,
    "Tian_infra.py",
    "Data_cutout.fits",
    "RMS_map.fits",
    "PSF_model.fits",
    "mask_1.fits",
    "mask_2.fits",
    "mask_out.fits",
    "project.json",
    "project_artifacts.json",
)


def invalidate_project_list_cache():
    with PROJECT_LIST_CACHE_LOCK:
        PROJECT_LIST_CACHE["signature"] = None
        PROJECT_LIST_CACHE["projects"] = []
        PROJECT_LIST_CACHE["cached_at"] = 0.0


def json_response(handler, payload, status=200):
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler):
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode("utf-8"))


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json_with_retry(path, *, attempts=5, delay=0.05):
    path = Path(path)
    last_error = None
    for _ in range(max(1, attempts)):
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        except Exception as exc:
            last_error = exc
        time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise ValueError(f"JSON file is empty: {path}")


def compact_project_payload_for_save(payload):
    drop = object()

    def compact(value, path=()):
        if path == ("state", "euclid", "image", "data"):
            return drop
        if len(path) >= 5 and path[:3] == ("state", "euclid", "image_options") and path[-1] == "data":
            return drop
        if (
            "pixelated_panels" in path
            or "parametric_panels" in path
            or "semilinear_panels" in path
        ) and path[-1] == "data":
            return drop
        if isinstance(value, str) and len(value) > 2_000_000 and (value.startswith("data:") or (path and path[-1] == "preview_src")):
            return drop
        if isinstance(value, dict):
            compacted = {}
            for key, item in value.items():
                compacted_item = compact(item, (*path, str(key)))
                if compacted_item is not drop:
                    compacted[key] = compacted_item
            return compacted
        if isinstance(value, list):
            return [item for item in (compact(item, path) for item in value) if item is not drop]
        return value

    compacted_payload = compact(payload)
    return compacted_payload if isinstance(compacted_payload, dict) else {}


def default_python_path(env_name, legacy_path=None):
    value = os.getenv(env_name)
    if value:
        return str(Path(value).expanduser())
    if legacy_path and Path(legacy_path).expanduser().exists():
        return str(Path(legacy_path).expanduser())
    return sys.executable


def default_runtime_settings():
    return {
        "herculens_python": default_python_path("HERCULENS_PYTHON", HERCULENS_PYTHON_LEGACY),
        "euclid_python": default_python_path("EUCLID_PYTHON", EUCLID_PYTHON_LEGACY),
        "photoz_python": default_python_path("PHOSPHOROS_PYTHON", PHOSPHOROS_PYTHON_LEGACY),
        "phosphoros_root": str(Path(os.getenv("PHOSPHOROS_ROOT", str(PHOSPHOROS_ROOT))).expanduser()),
    }


def load_runtime_settings():
    settings = default_runtime_settings()
    for settings_path in (SETTINGS_PATH, LEGACY_SETTINGS_PATH):
        if not settings_path.exists():
            continue
        try:
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            for key in settings:
                if str(saved.get(key) or "").strip():
                    settings[key] = str(saved[key]).strip()
            break
        except Exception:
            pass
    return settings


def save_runtime_settings(payload):
    settings = default_runtime_settings()
    for key in settings:
        value = str(payload.get(key) or "").strip()
        if value:
            settings[key] = value
    settings["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(SETTINGS_PATH, settings)
    return settings


def configured_python(key):
    settings = load_runtime_settings()
    return Path(settings[key]).expanduser()


def configured_phosphoros_root():
    return Path(load_runtime_settings()["phosphoros_root"]).expanduser()


def check_python_runtime(payload):
    key = str(payload.get("key") or "herculens_python").strip()
    path_value = str(payload.get("path") or load_runtime_settings().get(key) or sys.executable).strip()
    python_path = Path(path_value).expanduser()
    imports = {
        "herculens_python": "import herculens, jax, numpyro",
        "euclid_python": "import astropy, numpy",
        "photoz_python": "import PhzCLI.Phosphoros",
    }.get(key, "")
    code = (
        "import sys\n"
        "print(sys.executable)\n"
        f"{imports}\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    return {
        "key": key,
        "path": str(python_path),
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def relative_url(path):
    resolved = Path(path).expanduser().resolve()
    try:
        return "/" + resolved.relative_to(GUI_ROOT).as_posix()
    except ValueError:
        pass
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise
    return "/api/file?" + urlencode({"path": str(resolved)})


def relative_url_with_mtime(path):
    url = relative_url(path)
    try:
        version = int(path.stat().st_mtime)
    except Exception:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={version}"


def safe_filename(name):
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)
    return clean or "uploaded_image"


def timestamp_like(value):
    return bool(re.fullmatch(r"\d{8}_\d{6}", str(value or "").strip()))


def auto_project_name_like(value):
    text = str(value or "").strip()
    return bool(
        not text
        or timestamp_like(text)
        or re.fullmatch(r"(?:Euclid_)?RA[^/\\]*_DEC[^/\\]*", text)
        or re.fullmatch(r"DSPL_?RA[^/\\]*DEC[^/\\]*", text, flags=re.IGNORECASE)
    )


def canonical_project_name(project_name, project_id="", project_folder=""):
    name = str(project_name or "").strip()
    project_id = str(project_id or "").strip()
    folder_name = Path(str(project_folder or "")).name if project_folder else ""
    return folder_name or project_id or name


def coordinate_project_token(value, signed=False):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    sign = "m" if signed and numeric < 0 else ("p" if signed else "")
    return f"{sign}{abs(numeric):.6f}".replace(".", "p")


def dspl_coordinate_project_token(value, signed=False):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    sign = "NEG" if signed and numeric < 0 else ""
    text = f"{abs(numeric):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"{sign}{text}"


def payload_is_dspl_project(payload):
    project_name = str(payload.get("project_name") or payload.get("project_id") or "").strip()
    if project_name.upper().startswith("DSPL"):
        return True
    state = payload.get("state") if isinstance(payload, dict) else {}
    lens = state.get("lens") if isinstance(state, dict) else {}
    if isinstance(lens, dict):
        form_payload = lens.get("form_payload")
        model_config = lens.get("model_config")
        if isinstance(form_payload, dict) and bool(form_payload.get("dspl_enabled")):
            return True
        if isinstance(model_config, dict) and bool(model_config.get("dspl_enabled") or model_config.get("dspl", {}).get("enabled")):
            return True
    return False


def contains_euclid_astroquery_path(value):
    if isinstance(value, dict):
        return any(contains_euclid_astroquery_path(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_euclid_astroquery_path(item) for item in value)
    text = str(value or "")
    return "euclid_cutouts" in text and "/legacy/" not in text


def contains_downloaded_cutout_path(value):
    if isinstance(value, dict):
        return any(contains_downloaded_cutout_path(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_downloaded_cutout_path(item) for item in value)
    text = str(value or "")
    return "euclid_cutouts" in text or "/legacy/" in text


def project_name_from_state_coordinates(payload):
    state = payload.get("state") if isinstance(payload, dict) else {}
    euclid = state.get("euclid") if isinstance(state, dict) else {}
    if not isinstance(euclid, dict):
        return ""
    if not contains_downloaded_cutout_path(euclid):
        return ""
    if payload_is_dspl_project(payload):
        ra_token = dspl_coordinate_project_token(euclid.get("ra"))
        dec_token = dspl_coordinate_project_token(euclid.get("dec"), signed=True)
        return f"DSPL_RA{ra_token}DEC{dec_token}" if (ra_token and dec_token) else ""
    ra_token = coordinate_project_token(euclid.get("ra"))
    dec_token = coordinate_project_token(euclid.get("dec"), signed=True)
    if not (ra_token and dec_token):
        return ""
    prefix = "Euclid_" if contains_euclid_astroquery_path(euclid) else ""
    return f"{prefix}RA{ra_token}_DEC{dec_token}"


def project_id_from_name(project_name):
    project_name = str(project_name or "").strip()
    return safe_filename(project_name) if project_name else time.strftime("%Y%m%d_%H%M%S")


def unique_project_id(project_name):
    base = project_id_from_name(project_name)
    if not project_file(base).exists():
        return base
    return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"


def resolve_workspace_path(value, default_base=GUI_ROOT):
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        GUI_ROOT.parents[1] / path,
        default_base / path,
        SERVER_CWD / path,
        GUI_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def existing_project_dir_from_payload(payload):
    raw_project_folder = (payload or {}).get("project_folder")
    if not raw_project_folder:
        raise ValueError("project_folder is required.")
    project_dir = Path(raw_project_folder).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    project_dir = project_dir.resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {project_dir}")
    return project_dir


def output_reference_to_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith(("http://", "https://", "data:", "blob:")):
        return None
    if text.startswith("/"):
        return (GUI_ROOT / text.lstrip("/")).resolve()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (GUI_ROOT / path).resolve()


def is_runs_output_path(path):
    try:
        path.resolve().relative_to(RUNS_ROOT.resolve())
        return True
    except ValueError:
        return False


def collect_output_paths(payload, path_keys):
    paths = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in path_keys:
                    path = output_reference_to_path(item)
                    if path is not None:
                        paths.append(path)
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return paths


def delete_runs_outputs(paths):
    deleted = []
    skipped = []
    unique_paths = sorted({path.resolve() for path in paths}, key=lambda path: len(path.parts), reverse=True)
    for path in unique_paths:
        if not is_runs_output_path(path):
            skipped.append({"path": str(path), "reason": "outside runs directory"})
            continue
        if not path.exists():
            skipped.append({"path": str(path), "reason": "not found"})
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path))
    return deleted, skipped


def clear_psf_outputs(payload):
    path_keys = {"preview_url", "image_url", "figure_url", "npy_path", "fits_path", "npz_path", "job_dir", "status_path", "log_path", "config_path"}
    paths = collect_output_paths(payload, path_keys)
    job_id = str(payload.get("job_id") or "").strip()
    if job_id:
        paths.extend(RUNS_ROOT.glob(f"**/psf_fit_{safe_filename(job_id)}_*"))
    deleted, skipped = delete_runs_outputs(paths)
    return {
        "message": f"PSF output deleted: {len(deleted)} path(s).",
        "deleted": deleted,
        "skipped": skipped,
    }


class UploadedField:
    def __init__(self, value="", filename="", file_obj=None):
        self.value = value
        self.filename = filename
        self.file = file_obj or io.BytesIO()


class MultipartForm(dict):
    pass


def parse_multipart_form_fallback(handler):
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    body = handler.rfile.read(content_length)
    message = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    form = MultipartForm()
    if not message.is_multipart():
        return form
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename() or ""
        payload = part.get_payload(decode=True) or b""
        if filename:
            form[name] = UploadedField(filename=filename, file_obj=io.BytesIO(payload))
        else:
            charset = part.get_content_charset() or "utf-8"
            form[name] = UploadedField(value=payload.decode(charset, errors="replace"))
    return form


def parse_multipart_form(handler):
    if cgi is not None:
        return cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
            },
        )
    return parse_multipart_form_fallback(handler)


def form_text_value(form, key):
    field = form[key] if key in form else None
    if field is None or getattr(field, "filename", ""):
        return ""
    return str(getattr(field, "value", "") or "").strip()


def form_json_value(form, key):
    value = form_text_value(form, key)
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def save_uploaded_field(handler, field_name, *subdir_parts):
    form = parse_multipart_form(handler)
    field = form[field_name] if field_name in form else None
    if field is None or not getattr(field, "filename", ""):
        raise ValueError("No file was uploaded.")

    project_payload = {
        "project_id": form_text_value(form, "project_id"),
        "project_folder": form_text_value(form, "project_folder"),
    }
    for key in ("cutout_bounds", "data_cutout_bounds", "cutout_shape", "data_cutout_shape", "image_shape"):
        value = form_json_value(form, key)
        if value not in (None, ""):
            project_payload[key] = value
    out_dir = project_subdir(project_payload, *subdir_parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(field.filename)}"
    out_path = out_dir / suffix
    with out_path.open("wb") as out:
        shutil.copyfileobj(field.file, out)
    return out_path, project_payload


def save_uploaded_file(handler):
    return save_uploaded_field(handler, "image_file", "data", "uploads")


def save_uploaded_rms(handler):
    return save_uploaded_field(handler, "rms_file", "data", "uploads")


def save_uploaded_psf(handler):
    return save_uploaded_field(handler, "psf_file", "psf", "uploads")


def load_science_image(path):
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits

        try:
            data, header = fits.getdata(path, header=True)
        except Exception:
            with fits.open(path, memmap=False) as hdul:
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    header = getattr(hdu, "header", None)
                    if data is not None:
                        break
                else:
                    raise ValueError("No image HDU found in FITS file.")
        return data, header

    if suffix == ".npy":
        import numpy as np

        return np.load(path), None

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        import matplotlib.image as mpimg

        return mpimg.imread(path), None

    raise ValueError(f"Unsupported backend preview format: {suffix}")


def display_downsample_stride(shape, max_pixels=GUI_IMAGE_PAYLOAD_MAX_PIXELS):
    if not shape or len(shape) < 2:
        return 1
    height, width = int(shape[0]), int(shape[1])
    pixels = max(1, height * width)
    return max(1, int(math.ceil(math.sqrt(pixels / max(1, int(max_pixels))))))


def downsample_display_array(array, max_pixels=GUI_IMAGE_PAYLOAD_MAX_PIXELS):
    import numpy as np

    data = np.asarray(array)
    if data.ndim < 2:
        return data, 1
    stride = display_downsample_stride(data.shape[:2], max_pixels=max_pixels)
    if stride <= 1:
        return data, 1
    if data.ndim == 2:
        return data[::stride, ::stride], stride
    return data[::stride, ::stride, ...], stride


def save_header_metadata(upload_path, header, payload=None):
    if header is None:
        return None
    out_path = project_subdir(payload, "data") / f"{upload_path.stem}_header.json"
    cards = {}
    for key in header.keys():
        value = header.get(key)
        try:
            json.dumps(value)
            cards[key] = value
        except TypeError:
            cards[key] = str(value)
    out_path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    return out_path


def save_preview_png(upload_path, payload=None):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.image as mpimg
    import numpy as np

    raw_data, header = load_science_image(upload_path)
    save_header_metadata(upload_path, header, payload)
    data = np.asarray(raw_data, dtype=float)
    data = np.squeeze(data)
    is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    if data.ndim > 2 and not is_rgb:
        data = data[0]
    if data.ndim != 2 and not is_rgb:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

    source_shape = data.shape[:2]
    display_data = data[..., :3] if is_rgb else data
    display_data, _ = downsample_display_array(display_data)
    finite = np.isfinite(display_data)
    if not np.any(finite):
        raise ValueError("Image contains no finite values.")

    values = display_data[finite]
    vmin, vmax = np.nanpercentile(values, [0.5, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    out_path = project_subdir(payload, "previews") / f"{upload_path.stem}_preview.png"

    scale = max(vmax - vmin, np.finfo(float).eps)
    if is_rgb:
        preview = np.clip((display_data - vmin) / scale, 0.0, 1.0)
    else:
        gray = np.clip((display_data - vmin) / scale, 0.0, 1.0)
        preview = np.repeat(gray[..., None], 3, axis=2)

    # Preserve the previous origin="lower" visual convention while writing a native-pixel PNG.
    preview = np.flipud(preview)
    mpimg.imsave(out_path, preview)
    return out_path, source_shape


def preview_existing_path(path, payload=None):
    preview_path, image_shape = save_preview_png(path, payload)
    return relative_url(preview_path), image_shape


def image_preview_metadata(path):
    try:
        import numpy as np

        raw_data, header = load_science_image(path)
        data = np.asarray(raw_data)
        data = np.squeeze(data)
        if data.ndim > 2 and not (data.ndim == 3 and data.shape[-1] in (3, 4)):
            data = data[0]
        shape = data.shape[:2] if data.ndim >= 2 else None
        return {
            "image_shape": list(shape) if shape else None,
            "pixel_scale_arcsec": pixel_scale_arcsec(header),
        }
    except Exception:
        return {"image_shape": None, "pixel_scale_arcsec": None}


def cutout_bounds_from_display(data_shape, x_display, y_display, side_length):
    import math

    height, width = data_shape[:2]
    x_center = min(max(int(math.floor(float(x_display))), 0), width - 1)
    y_display = min(max(int(math.floor(float(y_display))), 0), height - 1)
    y_center = height - 1 - y_display
    size = max(1, int(round(float(side_length))))
    half = size // 2

    x0 = max(0, x_center - half)
    x1 = min(width, x0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y_center - half)
    y1 = min(height, y0 + size)
    y0 = max(0, y1 - size)
    return x_center, y_center, size, x0, x1, y0, y1


def load_displayable_image(path):
    import numpy as np

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = resolve_workspace_path(path)
    raw_data, header = load_science_image(path)
    data = np.asarray(raw_data)
    data = np.squeeze(data)
    is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    if data.ndim > 2 and not is_rgb:
        data = data[0]
    if data.ndim != 2 and not is_rgb:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")
    return data, header, is_rgb


def scalar_image_from_path(path):
    import numpy as np

    data, header, is_rgb = load_displayable_image(path)
    if is_rgb:
        data = np.asarray(data[..., :3], dtype=float).mean(axis=2)
    else:
        data = np.asarray(data, dtype=float)
    return data, header


def brightest_display_pixel_near(data, x_display, y_display, radius=1):
    import math
    import numpy as np

    height, width = data.shape[:2]
    x_center = min(max(int(math.floor(float(x_display))), 0), width - 1)
    y_display = min(max(int(math.floor(float(y_display))), 0), height - 1)
    y_center = height - 1 - y_display
    x0 = max(0, x_center - int(radius))
    x1 = min(width, x_center + int(radius) + 1)
    y0 = max(0, y_center - int(radius))
    y1 = min(height, y_center + int(radius) + 1)
    patch = np.asarray(data[y0:y1, x0:x1], dtype=float)
    finite = np.isfinite(patch)
    if patch.size == 0 or not np.any(finite):
        raise ValueError("3x3 brightest-pixel search has no finite pixels.")
    peak_index = int(np.nanargmax(np.where(finite, patch, -np.inf)))
    y_peak, x_peak = np.unravel_index(peak_index, patch.shape)
    x_data = int(x0 + x_peak)
    y_data = int(y0 + y_peak)
    return {
        "x_display": float(x_data + 0.5),
        "y_display": float(height - 0.5 - y_data),
        "x_data": float(x_data),
        "y_data": float(y_data),
        "value": float(data[y_data, x_data]),
        "radius": int(radius),
    }


def odd_kernel_size(value):
    size = max(3, int(round(float(value))))
    if size % 2 == 0:
        size += 1
    return size


def save_log_png(data, out_path):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("Image contains no finite values.")

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    positive = np.ma.array(array, mask=(~finite) | (array <= 0))
    if positive.count():
        values = positive.compressed()
        vmin, vmax = np.nanpercentile(values, [0.5, 99.8])
        vmin = max(float(vmin), np.finfo(float).tiny)
        vmax = max(float(vmax), vmin * 1.01)
        ax.imshow(positive, origin="lower", cmap="gray", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        values = array[finite]
        vmin, vmax = np.nanpercentile(values, [0.5, 99.8])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        ax.imshow(array, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def psf_preview_from_path(path, payload=None):
    data, _ = scalar_image_from_path(path)
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_input_{safe_filename(path.stem)}"
    png_path = project_subdir(payload, "psf", "previews") / f"{tag}.png"
    save_log_png(data, png_path)
    return {
        "preview_url": relative_url(png_path),
        "source_path": str(path),
        "shape": list(data.shape),
    }


def project_folder_from_payload(payload):
    payload = payload or {}
    project_folder_value = ""
    if payload.get("project_folder"):
        project_folder_value = str(payload.get("project_folder") or "").strip()
    elif payload.get("project_id"):
        project_folder_value = str(project_folder(payload["project_id"]))
    if not project_folder_value:
        return None
    project_dir = Path(project_folder_value).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    sync_project_runtime_files(project_dir)
    return project_dir


def active_project_folder(payload=None):
    project_dir = project_folder_from_payload(payload or {})
    if project_dir is not None:
        return project_dir
    project = project_payload_or_default()
    return Path(project["project_folder"]).resolve()


def project_subdir(payload, *parts):
    path = active_project_folder(payload)
    for part in parts:
        path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path


def register_project_psf(source_path, payload):
    project_dir = project_folder_from_payload(payload)
    if project_dir is None:
        return None
    return write_standard_psf_artifact(source_path, project_dir / "PSF_model.fits")


def load_default_euclid_psf(payload):
    if not DEFAULT_EUCLID_PSF.exists():
        raise FileNotFoundError(f"Default Euclid PSF not found: {DEFAULT_EUCLID_PSF}")
    preview = psf_preview_from_path(DEFAULT_EUCLID_PSF, payload)
    project_psf_path = register_project_psf(DEFAULT_EUCLID_PSF, payload)
    preview["project_psf_path"] = project_psf_path
    preview["default_psf"] = "Euclid"
    return preview


DEFAULT_EUCLID_NISP_PSFS = {
    "Y": DEFAULT_EUCLID_Y_PSF,
    "J": DEFAULT_EUCLID_J_PSF,
    "H": DEFAULT_EUCLID_H_PSF,
}


def load_default_euclid_nisp_psf(payload, band):
    band = str(band).strip().upper()
    if band not in DEFAULT_EUCLID_NISP_PSFS:
        raise ValueError(f"Unknown Euclid NISP PSF band: {band}")
    psf_path = DEFAULT_EUCLID_NISP_PSFS[band]
    if not psf_path.exists():
        raise FileNotFoundError(f"Default Euclid {band}-band PSF not found: {psf_path}")
    preview = psf_preview_from_path(psf_path, payload)
    project_psf_path = register_project_psf(psf_path, payload)
    preview["project_psf_path"] = project_psf_path
    preview["default_psf"] = f"Euclid {band}"
    return preview


def load_default_euclid_h_psf(payload):
    return load_default_euclid_nisp_psf(payload, "H")


def register_project_rms(source_path, payload):
    project_dir = project_folder_from_payload(payload)
    if project_dir is None:
        return None
    return write_project_rms_artifact(source_path, project_dir / "RMS_map.fits", payload)


def _shape_from_payload(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        height = int(round(float(value[0])))
        width = int(round(float(value[1])))
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _cutout_bounds_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    bounds = payload.get("cutout_bounds") or payload.get("data_cutout_bounds") or payload.get("bounds")
    if not isinstance(bounds, dict):
        return None
    try:
        out = {key: int(round(float(bounds[key]))) for key in ("x0", "x1", "y0", "y1")}
    except Exception:
        return None
    if out["x1"] <= out["x0"] or out["y1"] <= out["y0"]:
        return None
    source_shape = _shape_from_payload(bounds.get("source_shape") or payload.get("image_shape"))
    if source_shape:
        out["source_shape"] = source_shape
    return out


def _target_cutout_shape_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    return _shape_from_payload(
        payload.get("cutout_shape")
        or payload.get("data_cutout_shape")
        or payload.get("data_shape")
    )


def register_rms_request_payload(source_path, payload):
    from astropy.io import fits

    source_path = resolved_existing_or_workspace_path(source_path)
    rms_path = register_project_rms(source_path, payload or {})
    if not rms_path:
        raise ValueError("Project folder is required before registering an RMS file.")
    rms_shape = list(fits.getdata(rms_path).shape[:2])
    return {
        "message": f"Registered RMS map for lens model: {Path(rms_path).name}",
        "source_path": str(source_path),
        "rms_map_path": str(rms_path),
        "rms_shape": rms_shape,
    }


def header_float(header, keys):
    if header is None:
        return None, None
    for key in keys:
        if key in header:
            try:
                return float(header[key]), key
            except Exception:
                pass
    return None, None


def fits_zeropoints(header):
    flux_unit = str(header.get("BUNIT", "") if header is not None else "").strip()
    ab_zp, ab_key = header_float(
        header,
        ("ZPAB", "MAGZERO", "ABMAGZP", "PHOTZP", "MAGZP", "ZEROPOINT", "ZEROPNT", "ZP"),
    )
    vega_zp, vega_key = header_float(header, ("ZPVEGA", "VEGAZP", "MAGZPV", "ZP_VEGA"))

    source = {}
    if ab_zp is not None:
        source["ab"] = ab_key
    elif "ujy" in flux_unit.lower() or "microjy" in flux_unit.lower() or "microjansky" in flux_unit.lower():
        ab_zp = MICROJY_AB_ZP
        source["ab"] = "microJy AB default"
    else:
        ab_zp = EUCLID_VIS_DEFAULT_AB_ZP
        source["ab"] = "Euclid VIS Q1 default"

    if vega_zp is not None:
        source["vega"] = vega_key

    return {
        "zeropoints": {
            "ab": ab_zp,
            "vega": vega_zp,
        },
        "zeropoint_source": source,
        "flux_unit": flux_unit,
    }


def pixel_scale_arcsec(header, default=None):
    if header is None:
        return default
    try:
        if "CD1_1" in header and "CD1_2" in header and "CD2_1" in header and "CD2_2" in header:
            det = float(header["CD1_1"]) * float(header["CD2_2"]) - float(header["CD1_2"]) * float(header["CD2_1"])
            if det != 0:
                return float((abs(det) ** 0.5) * 3600.0)
        if "CDELT1" in header and "CDELT2" in header:
            return float((abs(float(header["CDELT1"]) * float(header["CDELT2"])) ** 0.5) * 3600.0)
        if "PIXSCALE" in header:
            return float(header["PIXSCALE"])
        if "PIXSCAL1" in header and "PIXSCAL2" in header:
            return float((abs(float(header["PIXSCAL1"]) * float(header["PIXSCAL2"])) ** 0.5))
    except Exception:
        return default
    return default


def infer_band_from_path(path):
    name = Path(path).name.upper()
    for band in ("NIR_H", "NIR_J", "NIR_Y", "VIS"):
        if band in name:
            return band
    legacy_name = Path(path).name.lower()
    for band in ("g", "r", "i", "z"):
        if f"_{band}" in legacy_name or legacy_name.endswith(f"{band}.fits"):
            return band
    return None


def image_payload_from_array(array, header, path, photometry, pixel_scale=None, band=None, source=None):
    import numpy as np

    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {array.shape}.")
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("Image has no finite pixels.")
    fill = float(np.nanmedian(array[finite]))
    clean = np.where(finite, array, fill)
    source_shape = [int(clean.shape[0]), int(clean.shape[1])]
    clean, display_stride = downsample_display_array(clean)
    display = np.flipud(clean)
    display_finite = np.isfinite(clean)
    values = clean[display_finite]
    vmin, vmax = np.nanpercentile(values, [0.5, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    resolved_pixel_scale = pixel_scale_arcsec(header, pixel_scale)
    if resolved_pixel_scale is not None:
        resolved_pixel_scale = float(resolved_pixel_scale) * int(display_stride)
    return {
        "path": str(Path(path)),
        "shape": [int(display.shape[0]), int(display.shape[1])],
        "source_shape": source_shape,
        "display_stride": int(display_stride),
        "data": display.tolist(),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "zeropoint": photometry["zeropoints"]["ab"],
        "pixel_scale_arcsec": resolved_pixel_scale,
        "band": band,
        "source": source,
        **photometry,
    }


def fits_payload_for_gui(path):
    data, header = scalar_image_from_path(Path(path))
    photometry = fits_zeropoints(header)
    return image_payload_from_array(data, header, path, photometry, band=infer_band_from_path(path), source="euclid")


def photoz_band_sort_key(band):
    try:
        return (PHOTOZ_BAND_ORDER.index(band), band)
    except ValueError:
        return (999, band)


def median(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if values.size else 0.0


def robust_std(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(values))


def value_to_njy(value, header, photometry):
    import numpy as np

    flux_unit = str(photometry.get("flux_unit") or header.get("BUNIT", "") or "").lower()
    if "nano" in flux_unit and "jy" in flux_unit:
        return float(value)
    if "ujy" in flux_unit or "microjy" in flux_unit or "microjansky" in flux_unit:
        return float(value) * 1000.0
    if "nanomag" in flux_unit:
        return float(value) * 3631.0
    zeropoint = photometry.get("zeropoints", {}).get("ab")
    if zeropoint is None:
        zeropoint = EUCLID_VIS_DEFAULT_AB_ZP
    if not np.isfinite(float(zeropoint)):
        return float(value)
    return float(value) * 3631e9 * 10 ** (-0.4 * float(zeropoint))


def ab_magnitude_from_native_flux(flux_native, photometry):
    import numpy as np

    flux = float(flux_native)
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("Flux must be positive for AB magnitude conversion.")
    zeropoint = float(photometry["zeropoints"]["ab"])
    if not np.isfinite(zeropoint):
        raise ValueError("A finite AB zeropoint is required.")
    return float(zeropoint - 2.5 * np.log10(flux))


def mean_surface_brightness_ab_mag_arcsec2(half_flux_native, r_half_arcsec, photometry):
    import numpy as np

    area_arcsec2 = np.pi * float(r_half_arcsec) ** 2
    if not np.isfinite(area_arcsec2) or area_arcsec2 <= 0:
        raise ValueError("Half-light aperture area must be positive.")
    return ab_magnitude_from_native_flux(float(half_flux_native) / area_arcsec2, photometry)


def log_surface_brightness_lsun_pc2(m_total_ab, r_half_arcsec, redshift, solar_abs_mag, k_correction, evolution_q):
    import numpy as np

    theta_e = float(r_half_arcsec)
    z = float(redshift)
    if not np.isfinite(theta_e) or theta_e <= 0:
        raise ValueError("Half-light radius must be positive.")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Redshift must be positive.")
    return float(
        0.4 * (float(solar_abs_mag) - float(m_total_ab) - float(evolution_q) * z + float(k_correction))
        - np.log10(2.0 * np.pi * theta_e * theta_e)
        + 4.0 * np.log10(1.0 + z)
        + 2.0 * np.log10(64800.0 / np.pi)
    )


def display_pixel_to_sky(path, x_display, y_display):
    from astropy.wcs import WCS

    data, header = scalar_image_from_path(Path(path))
    height, width = data.shape[:2]
    x = float(x_display)
    y = float(y_display)
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    fits_y = (height - 1.0) - y
    wcs = WCS(header).celestial
    coord = wcs.pixel_to_world(x, fits_y)
    return coord, {"x_display": x, "y_display": y, "x_fits": x, "y_fits": fits_y}


def measurement_image_from_path(path, band=None):
    import numpy as np

    data, header = load_science_image(Path(path))
    array = np.asarray(data, dtype=float)
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        band_names = [ch for ch in str(header.get("BANDS", "")).strip().lower() if ch in "griz"]
        band_key = str(band or "").lower()
        if band_key not in band_names:
            raise ValueError(f"Band {band!r} is not present in {Path(path).name}; available={band_names}")
        array = array[band_names.index(band_key)]
    elif array.ndim > 2:
        array = np.squeeze(array)
        if array.ndim > 2:
            raise ValueError(f"Photo-z measurement requires a 2D image plane, got {array.shape}.")
    return np.asarray(array, dtype=float), header


def measure_path_at_sky(path, coord, radius_px, annulus_width_px, band=None):
    import numpy as np
    from astropy.wcs import WCS

    data, header = measurement_image_from_path(Path(path), band=band)
    data = np.asarray(data, dtype=float)
    photometry = fits_zeropoints(header)
    wcs = WCS(header).celestial
    x_pix, y_pix = wcs.world_to_pixel(coord)
    radius = max(0.5, float(radius_px))
    annulus_width = max(0.0, float(annulus_width_px))
    outer = radius + annulus_width
    x0 = max(0, int(np.floor(x_pix - outer)))
    x1 = min(data.shape[1] - 1, int(np.ceil(x_pix + outer)))
    y0 = max(0, int(np.floor(y_pix - outer)))
    y1 = min(data.shape[0] - 1, int(np.ceil(y_pix + outer)))

    aperture_sum = 0.0
    aperture_pixels = 0
    bg_values = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            value = float(data[y, x])
            if not np.isfinite(value):
                continue
            distance = float(np.hypot(x - x_pix, y - y_pix))
            if distance <= radius:
                aperture_sum += value
                aperture_pixels += 1
            elif annulus_width > 0 and distance <= outer:
                bg_values.append(value)

    bg_median = median(bg_values) if annulus_width > 0 else 0.0
    bg_sigma = robust_std(bg_values) if annulus_width > 0 else 0.0
    flux_native = aperture_sum - bg_median * aperture_pixels
    flux_error_native = bg_sigma * float(np.sqrt(max(aperture_pixels, 1)))
    if annulus_width == 0:
        flux_error_native = max(abs(flux_native) * 1e-3, 1e-12)
    flux_njy = value_to_njy(flux_native, header, photometry)
    flux_error_njy = abs(value_to_njy(flux_error_native, header, photometry))
    return {
        "band_key": band,
        "path": str(path),
        "x_pix": float(x_pix),
        "y_pix": float(y_pix),
        "aperture_pixels": int(aperture_pixels),
        "annulus_pixels": int(len(bg_values)),
        "aperture_sum": float(aperture_sum),
        "background_median": float(bg_median),
        "background_sigma": float(bg_sigma),
        "background_subtracted": bool(annulus_width > 0),
        "flux_native": float(flux_native),
        "flux_error_native": float(flux_error_native),
        "flux_native_unit": photometry.get("flux_unit", ""),
        "flux": float(flux_njy),
        "flux_error": float(flux_error_njy),
        "flux_unit": "nJy",
        "snr": None if flux_error_njy <= 0 else float(flux_njy / flux_error_njy),
    }


def phosphoros_env():
    photoz_python = configured_python("photoz_python")
    env = os.environ.copy()
    env["PHOSPHOROS_ROOT"] = str(configured_phosphoros_root())
    env["PATH"] = str(photoz_python.parent) + os.pathsep + env.get("PATH", "")
    return env


def phosphoros_command(action, config_path):
    return [str(configured_python("photoz_python")), "-m", "PhzCLI.Phosphoros", action, f"--config-file={config_path}"]


def write_photoz_configs(measurements, run_key, payload=None):
    from astropy.table import Table

    usable = [
        item
        for item in measurements
        if item.get("band_key") in PHOTOZ_FILTERS
        and item.get("flux") is not None
        and item.get("flux_error") is not None
        and float(item["flux_error"]) > 0
    ]
    usable = unique_photoz_measurements_by_band(usable)
    usable.sort(key=lambda item: photoz_band_sort_key(item["band_key"]))
    if len(usable) < 3:
        raise RuntimeError("Phosphoros needs at least three usable bands.")

    bands = [item["band_key"] for item in usable]
    digest = hashlib.sha1(",".join(bands).encode("utf-8")).hexdigest()[:8]
    clean_run_key = safe_filename(run_key)
    catalog_type = f"HGUI_{digest}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = project_subdir(payload, "photoz", f"{timestamp}_{clean_run_key}")
    run_dir.mkdir(parents=True, exist_ok=True)

    root = PHOSPHOROS_ROOT
    dirs = {
        "catalogs": root / "Catalogs" / catalog_type,
        "intermediate": root / "IntermediateProducts" / catalog_type,
        "model_grids": root / "IntermediateProducts" / catalog_type / "ModelGrids",
        "results": root / "Results" / catalog_type,
        "config": root / "config" / catalog_type,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    catalog_name = f"hgui_click_{timestamp}_{clean_run_key}"
    catalog_path = dirs["catalogs"] / f"{catalog_name}.fits"
    table_row = {"OBJECT_ID": "hgui_click"}
    filter_lines = []
    error_lines = []
    filter_names = []
    for item in usable:
        filter_name, flux_col, err_col = PHOTOZ_FILTERS[item["band_key"]]
        table_row[flux_col] = float(item["flux"])
        table_row[err_col] = float(item["flux_error"])
        filter_names.append(filter_name)
        filter_lines.append(f"{filter_name} {flux_col} {err_col} 3 0 NONE")
        error_lines.append(f"{filter_name} 1 0 0")
    Table(rows=[table_row]).write(catalog_path, overwrite=True)
    (dirs["intermediate"] / "filter_mapping.txt").write_text("\n".join(filter_lines) + "\n", encoding="utf-8")
    (dirs["intermediate"] / "error_adjustment_param.txt").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    model_grid_name = f"Grid_{catalog_type}_MADAU.dat"
    model_grid_path = dirs["model_grids"] / model_grid_name
    cmg_config = dirs["config"] / f"{catalog_type}_ComputeModelGrid.conf"
    cr_config = dirs["config"] / f"{catalog_type}_{catalog_name}_ComputeRedshift.conf"
    sed_names = [
        "COSMOS/Ell1_A_0",
        "COSMOS/Ell2_A_0",
        "COSMOS/Ell3_A_0",
        "COSMOS/Ell4_A_0",
        "COSMOS/Ell5_A_0",
        "COSMOS/Ell6_A_0",
        "COSMOS/Ell7_A_0",
        "COSMOS/S0_A_0",
    ]
    normalization = "Euclid/VIS.vis" if "Euclid/VIS.vis" in filter_names else filter_names[0]
    cmg_lines = [
        f"phosphoros-root={root}",
        f"aux-data-dir={root / 'AuxiliaryData'}",
        f"intermediate-products-dir={root / 'IntermediateProducts'}",
        "thread-no=1",
        f"catalog-type={catalog_type}",
        f"normalization-filter={normalization}",
        *[f"filter-name={name}" for name in filter_names],
        *[f"sed-name={name}" for name in sed_names],
        "normalization-solar-sed=solar_spectrum",
        "igm-absorption-type=MADAU",
        "reddening-curve-name=SB_Calzetti",
        "ebv-value=0.000000",
        "z-range=0.000000 5.000000 0.010000",
        f"output-model-grid={model_grid_name}",
    ]
    cmg_config.write_text("\n".join(cmg_lines) + "\n", encoding="utf-8")

    output_dir = run_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    cr_lines = [
        f"phosphoros-root={root}",
        f"aux-data-dir={root / 'AuxiliaryData'}",
        f"catalogs-dir={root / 'Catalogs'}",
        f"intermediate-products-dir={root / 'IntermediateProducts'}",
        f"results-dir={root / 'Results'}",
        "thread-no=1",
        f"catalog-type={catalog_type}",
        f"normalization-filter={normalization}",
        "normalization-solar-sed=solar_spectrum",
        f"input-catalog-file={catalog_path.name}",
        "source-id-column-name=OBJECT_ID",
        "missing-photometry-flag=-99",
        "enable-upper-limit=YES",
        "upper-limit-use-threshold-flag=-99",
        f"model-grid-file={model_grid_name}",
        f"phz-output-dir={output_dir}",
        "output-catalog-format=FITS",
        "create-output-best-likelihood-model=NO",
        "create-output-best-model=YES",
        "create-output-pdf=Z",
        "input-process-max=1",
    ]
    cr_config.write_text("\n".join(cr_lines) + "\n", encoding="utf-8")
    return {
        "usable_bands": usable,
        "model_grid_path": model_grid_path,
        "cmg_config": cmg_config,
        "cr_config": cr_config,
        "output_dir": output_dir,
        "run_dir": run_dir,
    }


def run_photoz(measurements, run_key="photoz", payload=None):
    from astropy.table import Table
    import numpy as np

    if not (PHOSPHOROS_ROOT / "AuxiliaryData").exists():
        raise RuntimeError(f"Missing Phosphoros AuxiliaryData under {PHOSPHOROS_ROOT}")
    config_payload = write_photoz_configs(measurements, run_key=run_key, payload=payload)
    if (not config_payload["model_grid_path"].exists()) or config_payload["model_grid_path"].stat().st_mtime < config_payload["cmg_config"].stat().st_mtime:
        cmg = subprocess.run(phosphoros_command("CMG", config_payload["cmg_config"]), capture_output=True, text=True, env=phosphoros_env(), check=False)
        if cmg.returncode != 0:
            raise RuntimeError(cmg.stderr.strip() or cmg.stdout.strip() or "Phosphoros CMG failed.")
    cr = subprocess.run(phosphoros_command("CR", config_payload["cr_config"]), capture_output=True, text=True, env=phosphoros_env(), check=False)
    if cr.returncode != 0:
        raise RuntimeError(cr.stderr.strip() or cr.stdout.strip() or "Phosphoros CR failed.")
    candidates = sorted(config_payload["output_dir"].glob("phz_cat*.fits"))
    if not candidates:
        raise RuntimeError(f"No Phosphoros output catalog under {config_payload['output_dir']}")
    table = Table.read(candidates[0])
    row = table[0]
    result = {"output_catalog": str(candidates[0]), "used_bands": [item["band_key"] for item in config_payload["usable_bands"]]}
    for col in row.colnames:
        value = row[col]
        if np.isscalar(value):
            try:
                result[col] = float(value)
            except Exception:
                result[col] = str(value)
    return result


def unique_photoz_measurements_by_band(measurements):
    unique = {}
    for item in measurements:
        band = item.get("band_key")
        if band not in PHOTOZ_FILTERS or band in unique:
            continue
        unique[band] = item
    return list(unique.values())


def run_photoz_modes(measurements, project_payload=None):
    usable = [
        item
        for item in measurements
        if item.get("band_key") in PHOTOZ_FILTERS
        and item.get("flux") is not None
        and item.get("flux_error") is not None
        and float(item.get("flux_error") or 0) > 0
    ]
    usable = unique_photoz_measurements_by_band(usable)
    euclid_measurements = [item for item in usable if item.get("band_key") in EUCLID_PHOTOZ_BANDS]
    all_measurements = list(usable)
    specs = {
        "euclid_only": {
            "label": "Euclid only",
            "measurements": euclid_measurements,
        },
        "euclid_plus_ground": {
            "label": "Euclid + DESI/DECam",
            "measurements": all_measurements,
        },
    }
    results = {}
    for key, spec in specs.items():
        run_measurements = spec["measurements"]
        payload = {
            "label": spec["label"],
            "used_bands": [item.get("band_key") for item in run_measurements],
            "photoz": None,
            "error": None,
        }
        if key == "euclid_plus_ground" and not any(item.get("band_key") not in EUCLID_PHOTOZ_BANDS for item in run_measurements):
            payload["error"] = "No DESI/DECam band is loaded; combined photo-z was not run."
            results[key] = payload
            continue
        if len(run_measurements) < 3:
            payload["error"] = f"Need at least three usable bands; got {len(run_measurements)}."
            results[key] = payload
            continue
        try:
            payload["photoz"] = run_photoz(run_measurements, run_key=key, payload=project_payload)
            payload["used_bands"] = payload["photoz"].get("used_bands", payload["used_bands"])
        except Exception as exc:
            payload["error"] = str(exc)
        results[key] = payload
    return results


def euclid_band_paths_from_job(path):
    path = Path(path)
    if path.parent.name != "fits":
        return {}
    paths = {}
    for band in EUCLID_BANDS:
        matches = sorted(path.parent.glob(f"*_{band}.fits"))
        if matches:
            paths[band] = matches[-1]
    return paths


def add_photoz_image_spec(specs, seen, *, path, band, label=None, source=None):
    path = Path(path).expanduser().resolve()
    if band not in SUPPORTED_PHOTOZ_BANDS or not path.exists():
        return
    key = band
    if key in seen:
        return
    seen.add(key)
    specs.append(
        {
            "path": str(path),
            "band": band,
            "label": label or EUCLID_BAND_LABELS.get(band) or f"DESI {band}",
            "source": source or ("euclid" if band in EUCLID_PHOTOZ_BANDS else "legacy"),
        }
    )


def normalized_photoz_image_specs(raw_images, allowed_bands=None):
    allowed = None if allowed_bands is None else {band for band in allowed_bands if band in SUPPORTED_PHOTOZ_BANDS}
    specs = []
    seen = set()

    for image in raw_images or []:
        band = image.get("band")
        path_text = image.get("path")
        if not path_text or band not in SUPPORTED_PHOTOZ_BANDS:
            continue
        path = Path(path_text).expanduser().resolve()
        if band in EUCLID_PHOTOZ_BANDS:
            for sibling_band, sibling_path in euclid_band_paths_from_job(path).items():
                if allowed is not None and sibling_band not in allowed:
                    continue
                add_photoz_image_spec(
                    specs,
                    seen,
                    path=sibling_path,
                    band=sibling_band,
                    label=EUCLID_BAND_LABELS[sibling_band],
                    source="euclid",
                )
        if allowed is not None and band not in allowed:
            continue
        add_photoz_image_spec(
            specs,
            seen,
            path=path,
            band=band,
            label=image.get("label"),
            source=image.get("source"),
        )
    return specs


def measure_photoz_payload(payload):
    reference_path = Path(payload.get("reference_path") or "").expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference FITS does not exist: {reference_path}")
    coord, click = display_pixel_to_sky(reference_path, payload.get("x"), payload.get("y"))
    radius_px = float(payload["aperture_radius"])
    annulus_width_px = float(payload["annulus_width"])
    allowed_bands = payload.get("photoz_bands") if "photoz_bands" in payload else None

    measurements = []
    for image in normalized_photoz_image_specs(payload.get("images") or [], allowed_bands=allowed_bands):
        path = Path(image.get("path") or "").expanduser().resolve()
        band = image.get("band")
        try:
            item = measure_path_at_sky(path, coord, radius_px, annulus_width_px, band=band)
            item["band_key"] = band
            item["label"] = image["label"]
            measurements.append(item)
        except Exception as exc:
            measurements.append({"path": str(path), "band_key": band, "label": image["label"], "error": str(exc)})

    photoz_results = run_photoz_modes(measurements, project_payload=payload)
    primary = photoz_results.get("euclid_plus_ground", {})
    if primary.get("photoz") is None:
        primary = photoz_results.get("euclid_only", {})
    photoz = primary.get("photoz")
    photoz_error = primary.get("error")

    return {
        "message": "Photo-z measurement completed." if photoz else "Photometry completed; photo-z did not run.",
        "skycoord_deg": {"ra": float(coord.ra.deg), "dec": float(coord.dec.deg)},
        "click": click,
        "photometry": sorted(measurements, key=lambda item: photoz_band_sort_key(item.get("band_key"))),
        "photoz_results": photoz_results,
        "photoz": photoz,
        "photoz_error": photoz_error,
    }


def annulus_values(data, x_pix, y_pix, r_inner, r_outer):
    import numpy as np

    x0 = max(0, int(np.floor(x_pix - r_outer)))
    x1 = min(data.shape[1] - 1, int(np.ceil(x_pix + r_outer)))
    y0 = max(0, int(np.floor(y_pix - r_outer)))
    y1 = min(data.shape[0] - 1, int(np.ceil(y_pix + r_outer)))
    values = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            value = float(data[y, x])
            if not np.isfinite(value):
                continue
            radius = float(np.hypot(x - x_pix, y - y_pix))
            if r_inner < radius <= r_outer:
                values.append(value)
    return values


def half_light_radius_pixels(data, x_pix, y_pix, max_radius_px, background=0.0):
    import numpy as np

    max_radius = float(max_radius_px)
    x0 = max(0, int(np.floor(x_pix - max_radius)))
    x1 = min(data.shape[1] - 1, int(np.ceil(x_pix + max_radius)))
    y0 = max(0, int(np.floor(y_pix - max_radius)))
    y1 = min(data.shape[0] - 1, int(np.ceil(y_pix + max_radius)))
    samples = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            value = float(data[y, x])
            if not np.isfinite(value):
                continue
            radius = float(np.hypot(x - x_pix, y - y_pix))
            if radius <= max_radius:
                samples.append((radius, max(value - background, 0.0)))
    if not samples:
        raise ValueError("No finite VIS pixels inside the requested FP aperture.")
    samples.sort(key=lambda item: item[0])
    radii = np.asarray([item[0] for item in samples], dtype=float)
    fluxes = np.asarray([item[1] for item in samples], dtype=float)
    cumulative = np.cumsum(fluxes)
    total_flux = float(cumulative[-1])
    if not np.isfinite(total_flux) or total_flux <= 0:
        raise ValueError("VIS curve-of-growth has non-positive total flux.")
    half_flux = 0.5 * total_flux
    index = int(np.searchsorted(cumulative, half_flux, side="left"))
    if index <= 0:
        r_half = float(radii[0])
    else:
        f0 = cumulative[index - 1]
        f1 = cumulative[index]
        r0 = radii[index - 1]
        r1 = radii[index]
        weight = 0.0 if f1 <= f0 else float((half_flux - f0) / (f1 - f0))
        r_half = float(r0 + weight * (r1 - r0))
    return r_half, total_flux, half_flux, int(fluxes.size)


def fundamental_plane_payload(payload):
    from astropy.cosmology import Planck18
    from astropy.wcs import WCS
    import numpy as np

    reference_path = Path(payload["reference_path"]).expanduser().resolve()
    vis_path = Path(payload["vis_path"]).expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference FITS does not exist: {reference_path}")
    if not vis_path.exists():
        raise FileNotFoundError(f"VIS FITS does not exist: {vis_path}")

    coord, click = display_pixel_to_sky(reference_path, payload["x"], payload["y"])
    data, header = measurement_image_from_path(vis_path, band="VIS")
    wcs = WCS(header).celestial
    x_pix, y_pix = wcs.world_to_pixel(coord)
    pixel_scale = pixel_scale_arcsec(header)
    if pixel_scale is None or not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("VIS FITS header does not contain a usable pixel scale.")

    max_radius_px = float(payload["max_radius_px"])
    annulus_width_px = float(payload["annulus_width_px"])
    background = 0.0
    background_subtracted = annulus_width_px > 0
    if background_subtracted:
        background = median(annulus_values(data, x_pix, y_pix, max_radius_px, max_radius_px + annulus_width_px))

    r_half_px, total_flux, half_flux, n_pixels = half_light_radius_pixels(
        data,
        x_pix,
        y_pix,
        max_radius_px,
        background=background,
    )
    r_half_arcsec = r_half_px * pixel_scale
    z = float(payload["redshift"])
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Redshift must be positive.")
    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(z).value / 60.0)
    r_half_kpc = r_half_arcsec * kpc_per_arcsec
    if r_half_kpc <= 0:
        raise ValueError("Half-light radius is not positive.")

    photometry = fits_zeropoints(header)
    m_total_ab = ab_magnitude_from_native_flux(total_flux, photometry)
    mu_e = mean_surface_brightness_ab_mag_arcsec2(half_flux, r_half_arcsec, photometry)
    solar_abs_mag = float(payload.get("solar_abs_mag", DESI_R_BAND_SOLAR_AB_MAG))
    k_correction = float(payload.get("k_correction", 0.0))
    evolution_q = float(payload.get("evolution_q", DESI_FP_EVOLUTION_Q))
    log_i_e = log_surface_brightness_lsun_pc2(
        m_total_ab,
        r_half_arcsec,
        z,
        solar_abs_mag,
        k_correction,
        evolution_q,
    )

    a = float(payload.get("fp_a", DESI_DR1_FP_A))
    b = float(payload.get("fp_b", DESI_DR1_FP_B_LOGI))
    c = float(payload.get("fp_c", DESI_DR1_FP_C_KPC_PLANCK18))
    if a == 0:
        raise ValueError("FP coefficient a cannot be zero.")
    log_sigma = (np.log10(r_half_kpc) - b * log_i_e - c) / a
    sigma = 10 ** log_sigma

    return {
        "message": "Fundamental-plane sigma calculated.",
        "cosmology": "Planck18",
        "fp_form": "log10(Re/kpc) = a log10(sigma/km/s) + b log10(Ie/Lsun/pc^2) + c",
        "skycoord_deg": {"ra": float(coord.ra.deg), "dec": float(coord.dec.deg)},
        "click": click,
        "vis_path": str(vis_path),
        "vis_pixel": {"x": float(x_pix), "y": float(y_pix)},
        "redshift": z,
        "pixel_scale_arcsec": float(pixel_scale),
        "kpc_per_arcsec": kpc_per_arcsec,
        "r_half_px": float(r_half_px),
        "r_half_arcsec": float(r_half_arcsec),
        "r_half_kpc": float(r_half_kpc),
        "m_total_ab": float(m_total_ab),
        "mu_e_mag_arcsec2": float(mu_e),
        "log10_Ie_Lsun_pc2": float(log_i_e),
        "total_flux_native": float(total_flux),
        "half_flux_native": float(half_flux),
        "flux_native_unit": photometry.get("flux_unit", ""),
        "zeropoint_ab": float(photometry["zeropoints"]["ab"]),
        "zeropoint_source": photometry.get("zeropoint_source", {}),
        "solar_abs_mag": float(solar_abs_mag),
        "k_correction": float(k_correction),
        "evolution_q": float(evolution_q),
        "background": float(background),
        "background_subtracted": bool(background_subtracted),
        "n_pixels": n_pixels,
        "fp": {"a": a, "b": b, "c": c},
        "log10_sigma_km_s": float(log_sigma),
        "sigma_km_s": float(sigma),
    }


def download_legacy_cutout(payload):
    import numpy as np
    from astropy.io import fits

    ra = float(payload.get("ra"))
    dec = float(payload.get("dec"))
    radius_arcsec = float(payload.get("cutout_size") or payload.get("radius_arcsec") or 10.0)
    bands = "".join(ch for ch in str(payload.get("bands") or "griz").lower() if ch in "griz")
    if not bands:
        bands = "griz"
    pixscale = float(payload.get("pixscale") or LEGACY_DEFAULT_PIXSCALE)
    size_pixels = max(8, int(round((2.0 * radius_arcsec) / pixscale)))
    if size_pixels % 2 == 0:
        size_pixels += 1

    params = urlencode(
        {
            "ra": f"{ra:.8f}",
            "dec": f"{dec:.8f}",
            "layer": str(payload.get("layer") or "ls-dr10"),
            "pixscale": f"{pixscale:.4f}",
            "size": str(size_pixels),
            "bands": bands,
        }
    )
    url = f"https://www.legacysurvey.org/viewer/cutout.fits?{params}"
    tag = safe_filename(f"{time.strftime('%Y%m%d_%H%M%S')}_legacy_ra{ra:.6f}_dec{dec:.6f}_r{radius_arcsec:g}_{bands}.fits")
    out_path = project_subdir(payload, "legacy") / tag
    with urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())

    with fits.open(out_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    if data.ndim == 2:
        band_names = [bands[:1] or "image"]
        planes = [data]
    elif data.ndim == 3:
        band_header = str(header.get("BANDS", "")).strip().lower()
        band_names = [ch for ch in (band_header or bands) if ch in "griz"]
        if len(band_names) != data.shape[0]:
            band_names = list(bands[: data.shape[0]])
        planes = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError(f"Unexpected Legacy Survey FITS shape: {data.shape}.")

    photometry = {
        "zeropoints": {"ab": LEGACY_AB_ZP, "vega": None},
        "zeropoint_source": {"ab": "Legacy Survey nanomaggies"},
        "flux_unit": "nanomaggies",
    }
    images = {}
    for band, plane in zip(band_names, planes):
        images[band] = image_payload_from_array(
            plane,
            header,
            out_path,
            photometry,
            pixel_scale=pixscale,
            band=band,
            source="legacy",
        )

    active_band = bands[0] if bands[0] in images else next(iter(images))
    return {
        "message": f"Loaded DESI Legacy Survey cutout: {Path(out_path).name}",
        "fits_path": str(out_path),
        "url": url,
        "bands": list(images.keys()),
        "active_band": active_band,
        "images": images,
        "pixel_scale_arcsec": pixscale,
    }


def start_euclid_cutout_job(payload):
    if not EUCLID_CUTOUT_SCRIPT.exists():
        raise FileNotFoundError(f"Euclid helper not found: {EUCLID_CUTOUT_SCRIPT}")
    if not EUCLID_RMS_BUNDLE_SCRIPT.exists():
        raise FileNotFoundError(f"Euclid VIS RMS helper not found: {EUCLID_RMS_BUNDLE_SCRIPT}")
    if not EUCLID_CUTOUT_RUNNER.exists():
        raise FileNotFoundError(f"Euclid GUI runner not found: {EUCLID_CUTOUT_RUNNER}")
    ra = float(payload.get("ra"))
    dec = float(payload.get("dec"))
    radius_arcsec = float(payload.get("cutout_size") or payload.get("radius_arcsec") or 10.0)
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")

    job_id = time.strftime("%Y%m%d_%H%M%S")
    active_project = normalize_project_payload(
        {
            "project_id": payload.get("project_id") or job_id,
            "project_name": payload.get("project_name") or payload.get("project_id") or job_id,
            "project_folder": payload.get("project_folder"),
        },
        create_folder=True,
    )
    project_output_dir = Path(active_project["project_folder"]).expanduser()
    project_output_dir.mkdir(parents=True, exist_ok=True)
    output_root = project_output_dir / "euclid_cutouts" / f"euclid_cutout_{job_id}"
    job_dir = output_root
    job_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"
    status_path = job_dir / "status.json"
    credentials_path = None
    if username and password:
        credentials_path = job_dir / "credentials.txt"
        credentials_path.write_text(f"{username}\n{password}\n", encoding="utf-8")
        try:
            credentials_path.chmod(0o600)
        except Exception:
            pass

    config = {
        "job_id": job_id,
        "ra": ra,
        "dec": dec,
        "radius_arcsec": radius_arcsec,
        "output_root": str(output_root),
        "rms_bundle_dir": str(output_root / "vis_rms_bundle"),
        "project_folder": str(project_output_dir),
        "log_path": str(log_path),
    }
    atomic_write_json(job_dir / "config.json", config)
    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Euclid cutout job queued.",
            "job_dir": str(job_dir),
            "output_root": str(output_root),
            "project_folder": str(project_output_dir),
            "log_path": str(log_path),
            "progress": {"fraction": 0.0, "message": "Euclid cutout job queued."},
        },
    )
    python_exe = configured_python("euclid_python")
    command = [
        str(python_exe),
        str(EUCLID_CUTOUT_RUNNER),
        "--single-script",
        str(EUCLID_CUTOUT_SCRIPT),
        "--bundle-script",
        str(EUCLID_RMS_BUNDLE_SCRIPT),
        "--target-ra",
        str(ra),
        "--target-dec",
        str(dec),
        "--env",
        "IDR",
        "--radius-arcsec",
        str(radius_arcsec),
        "--output-root",
        str(output_root),
    ]
    if credentials_path is not None:
        command.extend(["--credentials-file", str(credentials_path)])
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd="/mnt/d/lensing",
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    return {
        "job_id": job_id,
        "state": "queued",
        "message": f"Started Euclid cutout job {job_id}.",
        "job_dir": str(job_dir),
        "output_root": str(output_root),
        "project_folder": str(project_output_dir),
        "log_path": str(log_path),
        "progress": {"fraction": 0.05, "message": "Euclid cutout job started."},
    }


def non_empty_files(paths):
    """Drop zero-byte files.

    The Euclid downloader leaves 0-byte placeholders for tiles that returned no
    data, and opening one raises "Empty or corrupt FITS file", which previously
    took the whole job-status endpoint down with a 404.
    """
    kept = []
    for path in paths:
        try:
            if path.stat().st_size > 0:
                kept.append(path)
        except OSError:
            continue
    return kept


def read_euclid_job_status(job_id, project_folder=None):
    safe_job_id = safe_filename(job_id)
    job_dir_name = f"euclid_cutout_{safe_job_id}"
    matches = []
    if project_folder:
        try:
            project_dir = resolve_project_folder_reference(project_folder)
            if project_dir is not None:
                candidate = project_dir / "euclid_cutouts" / job_dir_name / "status.json"
                if candidate.exists():
                    matches.append(candidate)
        except Exception:
            pass
    matches.extend(sorted(RUNS_ROOT.glob(f"**/{job_dir_name}/status.json")))
    if not matches:
        raise FileNotFoundError(f"Unknown Euclid cutout job: {job_id}")
    status_path = matches[-1]
    status = read_json_with_retry(status_path)
    job_dir = status_path.parent
    config_path = job_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    output_root = Path(status.get("output_root") or config.get("output_root") or job_dir / "cutout")
    project_folder_value = status.get("project_folder") or config.get("project_folder")
    log_path = Path(status.get("log_path") or job_dir / "run.log")
    fits_paths = sorted(non_empty_files((output_root / "fits").glob("*.fits")))
    jpeg_paths = sorted((output_root / "jpeg").glob("*.jpg"))
    rms_bundle_dir = Path(config.get("rms_bundle_dir") or output_root / "vis_rms_bundle")
    rms_bundle_paths = sorted(rms_bundle_dir.glob("*_VIS_IMAGE_ERRORMAP_PSF.fits"))
    rms_bundle_path = rms_bundle_paths[-1] if rms_bundle_paths else None
    project_rms_path = None
    if rms_bundle_path is not None:
        project_rms_path = register_project_rms(
            rms_bundle_path,
            {"project_id": config.get("project_id"), "project_folder": project_folder_value},
        )
    pid_path = job_dir / "pid.txt"
    running = False
    if pid_path.exists():
        try:
            running = Path(f"/proc/{pid_path.read_text().strip()}").exists()
        except Exception:
            running = False
    found_bands = {band for band in (infer_band_from_path(path) for path in fits_paths) if band in EUCLID_PHOTOZ_BANDS}
    expected_bands = set(EUCLID_BAND_LABELS)
    all_expected_fits = expected_bands.issubset(found_bands)

    def attach_downloaded_cutout_artifacts(message=None):
        status["fits_paths"] = [str(path) for path in fits_paths]
        status["jpeg_paths"] = [str(path) for path in jpeg_paths]
        images = {}
        for fits_path in fits_paths:
            band = infer_band_from_path(fits_path)
            if band not in EUCLID_PHOTOZ_BANDS:
                continue
            image = fits_payload_for_gui(fits_path)
            image["band"] = band
            image["label"] = EUCLID_BAND_LABELS[band]
            images[band] = image
        preferred = next((path for path in fits_paths if path.name.endswith("_VIS.fits")), fits_paths[0])
        artifacts = {}
        if project_folder_value:
            try:
                artifacts = register_project_original_file(
                    preferred,
                    {"project_folder": project_folder_value},
                    make_default_cutout=True,
                )
            except Exception:
                artifacts = {}
        status["images"] = images
        status["active_band"] = "VIS" if "VIS" in images else next(iter(images), "")
        status["image"] = fits_payload_for_gui(preferred)
        status["original_path"] = artifacts.get("original_path", "")
        status["data_cutout_path"] = artifacts.get("data_cutout_path", "")
        status["data_cutout_preview_url"] = artifacts.get("data_cutout_preview_url", "")
        status["data_cutout_bounds"] = artifacts.get("data_cutout_bounds")
        status["data_cutout_shape"] = artifacts.get("data_cutout_shape")
        status["psf_source_path"] = artifacts.get("original_path", str(preferred))
        if message:
            status["message"] = message

    if project_rms_path:
        status["rms_bundle_path"] = str(rms_bundle_path)
        status["rms_map_path"] = project_rms_path

    if fits_paths and all_expected_fits and running and not project_rms_path:
        status["state"] = "running"
        status["message"] = "Euclid cutout downloaded; downloading VIS RMS map."
        status["progress"] = {"fraction": 0.90, "message": "Downloading VIS RMS map."}
    elif fits_paths and (all_expected_fits or not running):
        text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        if not running and not project_rms_path:
            if "Traceback (most recent call last)" in text or "Error" in text or "Exception" in text:
                last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "Euclid VIS RMS download failed.")
                status["state"] = "completed"
                warning = f"Euclid cutout completed, but VIS RMS map failed: {last}"
                attach_downloaded_cutout_artifacts(warning)
                status["warning"] = last
                status["progress"] = {"fraction": 1.0, "message": warning}
                return status
            status["state"] = "completed"
            status["message"] = "Euclid cutout completed but VIS RMS map is missing."
            attach_downloaded_cutout_artifacts(status["message"])
            status["progress"] = {"fraction": 1.0, "message": status["message"]}
            return status
        status["state"] = "completed"
        if all_expected_fits:
            status["message"] = "Euclid cutout and VIS RMS map completed."
        else:
            missing = ", ".join(sorted(expected_bands - found_bands))
            status["message"] = f"Euclid cutout completed with missing bands: {missing}."
        status["progress"] = {"fraction": 1.0, "message": status["message"]}
        attach_downloaded_cutout_artifacts()
    elif fits_paths and running:
        status["state"] = "running"
        status["message"] = "Euclid cutout is downloading all bands."
        fraction = 0.20 + 0.70 * min(len(found_bands), len(expected_bands)) / max(len(expected_bands), 1)
        status["progress"] = {
            "fraction": fraction,
            "message": f"Downloaded {len(found_bands)}/{len(expected_bands)} Euclid FITS bands.",
        }
    elif log_path.exists() and not running:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if "Traceback (most recent call last)" in text or "Error" in text or "Exception" in text:
            last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "Euclid cutout failed.")
            status["state"] = "failed"
            status["message"] = last
            status["progress"] = {"fraction": 1.0, "message": last}
        else:
            status["state"] = "running"
            status["message"] = "Euclid cutout is running."
            status["progress"] = {"fraction": 0.45, "message": "Euclid cutout is running."}
    else:
        status["state"] = "running"
        status["message"] = "Euclid cutout is running."
        status["progress"] = {"fraction": 0.25, "message": "Euclid cutout is running."}
    return status


def fixed_cutout(data, x_center, y_center, size):
    import numpy as np

    size = odd_kernel_size(size)
    half = size // 2
    x_center = int(round(float(x_center)))
    y_center = int(round(float(y_center)))
    height, width = data.shape[:2]
    x0, x1 = x_center - half, x_center + half + 1
    y0, y1 = y_center - half, y_center + half + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    cutout = np.asarray(data[y0:y1, x0:x1], dtype=float)
    if cutout.shape != (size, size):
        return None
    return cutout, {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def normalize_psf_kernel(data):
    import numpy as np

    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("PSF kernel contains no finite values.")

    values = array[finite]
    fill = float(np.nanmedian(values))
    clean = np.where(finite, array, fill)
    if clean.ndim != 2:
        raise ValueError(f"Expected a 2D PSF kernel, got shape {clean.shape}.")

    border = np.concatenate([clean[0, :], clean[-1, :], clean[:, 0], clean[:, -1]])
    background = float(np.nanmedian(border[np.isfinite(border)])) if np.any(np.isfinite(border)) else fill
    kernel = clean - background
    kernel[~np.isfinite(kernel)] = 0.0
    kernel[kernel < 0] = 0.0
    total = float(np.sum(kernel))
    if total <= 0:
        kernel = clean - float(np.nanmin(clean))
        kernel[~np.isfinite(kernel)] = 0.0
        total = float(np.sum(kernel))
    if total <= 0:
        raise ValueError("PSF kernel cannot be normalized.")
    return kernel / total


def save_fitted_psf(kernel, tag, payload=None):
    import numpy as np
    from astropy.io import fits

    safe_tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(tag)}"
    psf_dir = project_subdir(payload, "psf")
    npy_path = psf_dir / f"{safe_tag}.npy"
    fits_path = psf_dir / f"{safe_tag}.fits"
    png_path = psf_dir / f"{safe_tag}.png"
    np.save(npy_path, kernel)
    fits.writeto(fits_path, kernel, overwrite=False)
    save_log_png(kernel, png_path)
    return {
        "preview_url": relative_url(png_path),
        "npy_path": str(npy_path),
        "fits_path": str(fits_path),
        "shape": list(kernel.shape),
    }


def find_psf_stars(path, kernel_size, threshold_sigma=5.0, max_stars=120, fwhm=1.6, payload=None):
    import numpy as np

    data, _ = scalar_image_from_path(path)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Science image contains no finite values.")

    values = data[finite]
    fill = float(np.nanmedian(values))
    work = np.where(finite, data, fill)
    try:
        from astropy.stats import sigma_clipped_stats

        mean, median, std = sigma_clipped_stats(work, sigma=3.0, maxiters=5)
        background = float(median)
        noise = float(std)
    except Exception:
        background = fill
        noise = float(np.nanstd(values))
    if not np.isfinite(noise) or noise <= 0:
        noise = max(float(np.nanstd(values)), np.finfo(float).eps)

    size = odd_kernel_size(kernel_size)
    threshold = background + float(threshold_sigma) * noise
    records = []

    def add_record(x, y, flux, peak):
        if not np.isfinite(x) or not np.isfinite(y):
            return
        for existing in records:
            if (existing["x"] - x) ** 2 + (existing["y"] - y) ** 2 <= 4.0:
                if flux > existing.get("flux", -np.inf):
                    existing.update({"x": float(x), "y": float(y), "flux": float(flux), "peak": float(peak)})
                return
        records.append({"x": float(x), "y": float(y), "flux": float(flux), "peak": float(peak)})

    try:
        from photutils.detection import DAOStarFinder

        finder_fwhm = max(0.5, float(fwhm))
        finder = DAOStarFinder(fwhm=finder_fwhm, threshold=float(threshold - background))
        table = finder(work - background)
        if table is not None:
            for row in table:
                flux = float(row["flux"]) if "flux" in table.colnames else float(row["peak"])
                peak = float(row["peak"]) if "peak" in table.colnames else flux
                add_record(float(row["xcentroid"]), float(row["ycentroid"]), flux, peak)
    except Exception:
        pass

    from scipy.ndimage import maximum_filter

    filter_size = max(3, min(size // 2, 15))
    maxima = (work == maximum_filter(work, size=filter_size)) & (work > threshold)
    coords = np.argwhere(maxima)
    if coords.size:
        peak_values = work[maxima]
        order = np.argsort(peak_values)[::-1]
        for idx in order:
            y, x = coords[idx]
            peak = float(work[y, x])
            add_record(float(x), float(y), peak, peak)

    records.sort(key=lambda item: item.get("flux", item.get("peak", 0.0)), reverse=True)
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(path.stem)}_k{size}"
    candidates_dir = project_subdir(payload, "psf", "candidates", tag)
    stars = []
    for record in records:
        extracted = fixed_cutout(work, record["x"], record["y"], size)
        if extracted is None:
            continue
        cutout, bounds = extracted
        star_id = len(stars) + 1
        png_path = candidates_dir / f"star_{star_id:03d}.png"
        save_log_png(cutout, png_path)
        stars.append(
            {
                "id": star_id,
                "x": float(record["x"]),
                "y": float(record["y"]),
                "flux": float(record.get("flux", record.get("peak", 0.0))),
                "peak": float(record.get("peak", record.get("flux", 0.0))),
                "bounds": bounds,
                "preview_url": relative_url(png_path),
            }
        )
        if len(stars) >= max_stars:
            break

    return {
        "source_path": str(path),
        "kernel_size": size,
        "fwhm": float(fwhm),
        "threshold_sigma": float(threshold_sigma),
        "stars": stars,
    }


def parse_selected_ids(value):
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").replace(" ", ",").split(",")
    ids = []
    seen = set()
    for item in raw:
        if item == "":
            continue
        try:
            star_id = int(item)
        except (TypeError, ValueError):
            continue
        if star_id in seen:
            continue
        seen.add(star_id)
        ids.append(star_id)
    return ids


def fit_psf_from_input(path, payload=None):
    data, _ = scalar_image_from_path(path)
    kernel = normalize_psf_kernel(data)
    result = save_fitted_psf(kernel, f"input_psf_{path.stem}", payload)
    result["source_path"] = str(path)
    project_psf_path = register_project_psf(result["fits_path"], payload or {})
    if project_psf_path:
        result["project_psf_path"] = project_psf_path
    return result


def fit_psf_from_selected_stars(path, kernel_size, selected_ids, stars, payload=None):
    import numpy as np

    data, _ = scalar_image_from_path(path)
    finite = np.isfinite(data)
    fill = float(np.nanmedian(data[finite])) if np.any(finite) else 0.0
    work = np.where(finite, data, fill)
    size = odd_kernel_size(kernel_size)
    star_map = {int(star["id"]): star for star in stars}
    kernels = []
    used_ids = []
    for star_id in selected_ids:
        star = star_map.get(int(star_id))
        if not star:
            continue
        extracted = fixed_cutout(work, star["x"], star["y"], size)
        if extracted is None:
            continue
        cutout, _ = extracted
        kernels.append(normalize_psf_kernel(cutout))
        used_ids.append(int(star_id))

    if not kernels:
        raise ValueError("No selected star IDs could be converted into PSF cutouts.")

    kernel = normalize_psf_kernel(np.mean(kernels, axis=0))
    result = save_fitted_psf(kernel, f"auto_psf_{path.stem}_n{len(used_ids)}", payload)
    result["source_path"] = str(path)
    result["selected_ids"] = used_ids
    return result


def start_psf_fit_job(payload):
    selected_ids = parse_selected_ids(payload.get("selected_ids"))
    if not selected_ids:
        raise ValueError("Select at least one star ID before running PSF fit.")

    source_path = Path(payload.get("image_path") or payload.get("path")).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Science image not found: {source_path}")

    job_id = time.strftime("%Y%m%d_%H%M%S")
    job_dir = project_subdir(payload, "psf", "jobs") / f"psf_fit_{job_id}_{safe_filename(source_path.stem)}"
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.json"
    log_path = job_dir / "run.log"
    status_path = job_dir / "status.json"
    ss_factor = int(payload.get("ss_factor", 3))
    if ss_factor not in (1, 3):
        ss_factor = 3

    config = {
        "job_id": job_id,
        "image_path": str(source_path),
        "kernel_size": int(odd_kernel_size(payload.get("kernel_size", 31))),
        "fwhm": float(payload.get("fwhm", 1.6)),
        "selected_ids": selected_ids,
        "stars": payload.get("stars", []),
        "base_psf_path": payload.get("base_psf_path") or payload.get("psf_path") or "",
        "bg_box": payload.get("bg_box"),
        "svi_steps": int(payload.get("svi_steps", 2000)),
        "num_chains": 1,
        "ss_factor": ss_factor,
        "analytic_psf_profile": payload.get("analytic_psf_profile", "moffat"),
        "psf_model_centered": bool(payload.get("psf_model_centered", True)),
        "background_subtracted": False,
        "psf_two_stage": True,
        "psf_init_strategy": "first_selected",
        "psf_stage1_zero_moments": False,
        "psf_stage1_recenter_weighted": bool(payload.get("psf_stage1_recenter_weighted", True)),
        "psf_position_reference": "absolute",
        "psf_correction_mode": payload.get("psf_correction_mode", "additive"),
        "psf_corr_model": payload.get("psf_corr_model", "pixel"),
        "psf_corr_sigma_low": float(payload.get("psf_corr_sigma_low", 1e-9)),
        "psf_corr_sigma_high": float(payload.get("psf_corr_sigma_high", 1e-4)),
        "psf_corr_zero_moments": bool(payload.get("psf_corr_zero_moments", True)),
        "psf_corr_zero_moments_mode": "delta",
        "psf_fit_background": bool(payload.get("psf_fit_background", True)),
        "psf_solve_flux": bool(payload.get("psf_solve_flux", True)),
        "psf_mult_log_clip": float(payload.get("psf_mult_log_clip", 12.0)),
        "psf_positive_transform": payload.get("psf_positive_transform", "softplus"),
        "output_dir": str(job_dir),
    }
    atomic_write_json(config_path, config)
    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "PSF fit job queued.",
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
        },
    )

    script_path = WEB_ROOT / "psf_svi_runner.py"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [
                str(configured_python("herculens_python")),
                str(script_path),
                "--config",
                str(config_path),
            ],
            cwd=str(GUI_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["pid"] = process.pid
    atomic_write_json(status_path, status)

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "status_path": str(status_path),
        "log_path": str(log_path),
        "config_path": str(config_path),
        "pid": process.pid,
        "state": "queued",
    }


def pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def psf_status_paths():
    paths = list(RUNS_ROOT.glob("**/psf_fit_*/status.json"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def status_path_for_psf_job(job_id):
    matches = sorted(RUNS_ROOT.glob(f"**/psf_fit_{safe_filename(job_id)}_*/status.json"))
    if not matches:
        raise FileNotFoundError(f"Unknown PSF job: {job_id}")
    return matches[-1]


def psf_job_pid(job_dir):
    pid_path = job_dir / "pid.txt"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def find_active_psf_job():
    for status_path in psf_status_paths():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("state") in PSF_TERMINAL_STATES:
            continue
        pid = psf_job_pid(status_path.parent)
        if pid is not None and pid_is_running(pid):
            return status_path
    return None


def stop_psf_fit_job(payload):
    job_id = payload.get("job_id")
    if job_id:
        status_path = status_path_for_psf_job(job_id)
    else:
        status_path = find_active_psf_job()
        if status_path is None:
            raise FileNotFoundError("No active PSF fit job to stop.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    job_dir = status_path.parent
    pid = psf_job_pid(job_dir)
    killed = False
    if pid is not None and pid_is_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            os.kill(pid, signal.SIGTERM)
        killed = True

    status.update(
        {
            "job_id": status.get("job_id") or job_dir.name.split("_", 2)[-1],
            "state": "stopped",
            "message": "PSF fit job stopped.",
            "job_dir": str(job_dir),
            "log_path": status.get("log_path", str(job_dir / "run.log")),
            "progress": {**(status.get("progress") or {}), "stopped": True},
            "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stopped_pid": pid,
        }
    )
    atomic_write_json(status_path, status)
    return {
        **status,
        "killed": killed,
        "message": "PSF fit job stopped." if killed else "PSF fit job was not running.",
    }


def lens_status_paths():
    paths = list(GUI_ROOT.glob("runs/**/lens_svi_*/status.json"))
    paths.extend(GUI_ROOT.glob(f"runs/*/{LENS_MODEL_RESULT_DIRNAME}/status.json"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def status_path_for_lens_job(job_id):
    legacy = sorted(GUI_ROOT.glob(f"runs/**/lens_svi_{job_id}/status.json"))
    if legacy:
        return legacy[-1]
    for status_path in lens_status_paths():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(status.get("job_id") or "") == str(job_id):
            return status_path
        config_path = status_path.parent / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            if str(config.get("job_id") or "") == str(job_id):
                return status_path
    raise FileNotFoundError(f"Unknown Lens job: {job_id}")


def lens_job_pid(job_dir):
    pid_path = job_dir / "pid.txt"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def find_active_lens_job():
    for status_path in lens_status_paths():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("state") in LENS_TERMINAL_STATES:
            continue
        job_dir = status_path.parent
        pid = lens_job_pid(job_dir)
        if pid is None or not pid_is_running(pid):
            continue
        log_path = Path(status.get("log_path") or job_dir / "run.log")
        return {
            "job_id": status.get("job_id", job_dir.name.removeprefix("lens_svi_")),
            "state": "running",
            "job_dir": str(job_dir),
            "config_path": status.get("config_path", str(job_dir / "config.json")),
            "log_path": str(log_path),
            "script_path": status.get("script_path", str(job_dir / LENS_MODEL_SCRIPT_NAME)),
            "progress": parse_lens_progress_from_log(log_path),
        }
    return None


def stop_lens_svi_job(payload):
    job_id = payload.get("job_id")
    if job_id:
        status_path = status_path_for_lens_job(job_id)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        job_dir = status_path.parent
    else:
        active_job = find_active_lens_job()
        if not active_job:
            raise FileNotFoundError("No active Lens SVI job to stop.")
        status_path = Path(active_job["job_dir"]) / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        job_dir = status_path.parent
        job_id = active_job["job_id"]

    pid = lens_job_pid(job_dir)
    killed = False
    if pid is not None and pid_is_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            os.kill(pid, signal.SIGTERM)
        killed = True

    log_path = Path(status.get("log_path") or job_dir / "run.log")
    progress = parse_lens_progress_from_log(log_path)
    progress["message"] = "Lens job stopped."
    progress["stopped"] = True

    status.update(
        {
            "job_id": job_id,
            "state": "stopped",
            "message": "Lens job stopped.",
            "job_dir": str(job_dir),
            "log_path": str(log_path),
            "progress": progress,
            "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stopped_pid": pid,
        }
    )
    atomic_write_json(status_path, status)
    return {
        **status,
        "killed": killed,
        "message": "Lens job stopped." if killed else "Lens job was not running.",
    }


def project_output_base(payload, subdir):
    explicit_output = str(payload.get("output_dir") or "").strip()
    if explicit_output:
        return resolve_workspace_path(explicit_output)
    base = active_project_folder(payload)
    output_base = base / subdir
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def start_lens_svi_job(payload):
    active_job = find_active_lens_job()
    if active_job:
        return {
            **active_job,
            "message": f"Lens job {active_job['job_id']} is already running; not starting a second job.",
            "existing_job": True,
        }

    payload, artifact_manifest = standardize_project_artifacts(payload)
    project_dir = Path(payload["project_folder"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    validate_source_arc_mask(project_dir)
    job_id = time.strftime("%Y%m%d_%H%M%S")
    job_dir = project_dir / LENS_MODEL_RESULT_DIRNAME
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.json"
    log_path = job_dir / "run.log"
    status_path = job_dir / "status.json"
    provided_code = str(payload.get("generated_code") or "")
    provided_config = payload.get("generated_model_config") or payload.get("model_config") or {}
    if provided_code.strip() and bool(payload.get("generated_code_edited")):
        generated_code = provided_code
        model_config = provided_config if isinstance(provided_config, dict) else {}
    else:
        generated_code, model_config = generate_lens_script(
            payload | {"output_dir": str(job_dir)},
            job_dir=job_dir,
            default_data_dir=DEFAULT_DATA_DIR,
        )
    script_path = project_dir / LENS_MODEL_SCRIPT_NAME
    script_path.write_text(generated_code, encoding="utf-8")
    config = {
        "job_id": job_id,
        "runner": str(script_path),
        "job_dir": str(job_dir),
        "log_path": str(log_path),
        "status_path": str(status_path),
        "script_path": str(script_path),
        "model_config": model_config,
        "max_iter_parametric": int(payload.get("max_iter_parametric", 10000)),
        "max_iter_pixelated": int(payload.get("max_iter_pixelated", 10000)),
        "num_chains": int(payload.get("num_chains", 1)),
        "source_grid_scale": float(payload.get("source_grid_scale", 1.0)),
        "seed": int(payload.get("seed", 100)),
        "project_artifacts": artifact_manifest,
    }
    atomic_write_json(config_path, config)
    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Lens SVI job queued.",
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "progress": {"fraction": 0.0, "message": "Lens SVI job queued."},
        },
    )
    python_exe = configured_python("herculens_python")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(python_exe), "-u", str(script_path)],
            cwd=str(project_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    return {
        "job_id": job_id,
        "state": "queued",
        "message": f"Started Lens SVI job {job_id}.",
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "script_path": str(script_path),
        "generated_code": generated_code,
        "model_config": model_config,
        "project_artifacts": artifact_manifest,
        "progress": {
            "fraction": 0.01,
            "message": "Lens SVI job started.",
            "stages": {
                "parametric": {"fraction": 0.0, "message": "Parametric SVI pending.", "state": "pending"},
                "pixelated": {"fraction": 0.0, "message": "Pixelated SVI pending.", "state": "pending"},
                        "solver": {"fraction": 0.0, "message": "Exact-solver SVI pending.", "state": "pending"},
                "semilinear": {"fraction": 0.0, "message": "Semilinear source inversion pending.", "state": "pending"},
            },
        },
    }


def safe_sciama_name(value):
    text = str(value or "project").strip() or "project"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]


def validate_lens_job_payload(payload, model_config):
    is_dspl = bool(payload.get("dspl_enabled"))
    if isinstance(model_config, dict):
        is_dspl = is_dspl or bool(model_config.get("dspl_enabled") or model_config.get("dspl", {}).get("enabled"))
    if not is_dspl:
        return
    if payload.get("conjugate_points_source1") or (isinstance(model_config, dict) and model_config.get("conjugate_points_source1")):
        return
    raise ValueError(
        "DSPL lens model requires Source 1 conjugate points. "
        "Please mark Source 1 conjugate points in the Mask tab before running Lens SVI."
    )


def validate_source_arc_mask(project_dir):
    """Reject an empty or incompatible Source 1 arc mask before model execution."""
    import numpy as np
    from astropy.io import fits

    project_dir = Path(project_dir).expanduser()
    data_path = project_dir / "Data_cutout.fits"
    mask_path = project_dir / "mask_1.fits"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot start Lens SVI; missing project image: {data_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Cannot start Lens SVI; missing Source 1 arc mask: {mask_path}")

    data_shape = tuple(np.asarray(fits.getdata(data_path)).shape[:2])
    mask = np.asarray(fits.getdata(mask_path))
    if tuple(mask.shape[:2]) != data_shape:
        raise ValueError(
            f"Cannot start Lens SVI; mask_1 shape {tuple(mask.shape[:2])} does not match "
            f"Data_cutout shape {data_shape}. Reload the image in Mask Preview and save mask_1 again."
        )
    mask_pixels = int(np.count_nonzero(np.isfinite(mask) & (mask > 0)))
    if mask_pixels == 0:
        raise ValueError(
            "Cannot start Lens SVI; mask_1.fits contains 0 selected pixels. "
            "In Mask Preview select mask_1 and Add, paint with Brush or define a polygon with "
            "at least three Line clicks, then save the mask."
        )
    return mask_pixels


def model_config_from_lens_script(script_text):
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return {}
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if any(isinstance(target, ast.Name) and target.id == "MODEL_CONFIG" for target in targets):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError, SyntaxError):
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def model_config_value(model_config, key, default=None):
    current = model_config if isinstance(model_config, dict) else {}
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def lens_sciama_slim_files(project_dir, model_config):
    """Return the standard slim bundle plus configured project-local model inputs."""
    project_dir = Path(project_dir).expanduser().resolve()
    slim_files = list(SCIAMA_SLIM_FILES)
    lens_light_path = model_config_value(
        model_config,
        "light.lens.external_kwargs_path",
        model_config.get("lens_light_external_path", "") if isinstance(model_config, dict) else "",
    )
    if lens_light_path:
        path = Path(str(lens_light_path)).expanduser()
        relative_path = path.resolve().relative_to(project_dir) if path.is_absolute() else path
        relative_text = relative_path.as_posix()
        if relative_text not in slim_files:
            slim_files.append(relative_text)
    components = model_config.get("lens_plane_mass_components", []) if isinstance(model_config, dict) else []
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            continue
        if str(component.get("profile") or "").upper() != "FIXED_MASS_MAP":
            continue
        path_text = str(component.get("path") or "").strip()
        if not path_text:
            raise ValueError(f"Fixed mass map component {index} has no FITS path.")
        path = Path(path_text).expanduser()
        if path.is_absolute():
            try:
                relative_path = path.resolve().relative_to(project_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Fixed mass map must be copied into the project folder before Sciama upload: {path}"
                ) from exc
        else:
            relative_path = path
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe project-relative fixed mass map path: {relative_path}")
        source = project_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Project-local fixed mass map does not exist: {source}")
        relative_text = relative_path.as_posix()
        if relative_text not in slim_files:
            slim_files.append(relative_text)
    return slim_files


def prepare_lens_job_inputs(payload, *, prefer_existing_script=False):
    payload, artifact_manifest = standardize_project_artifacts(payload)
    project_dir = Path(payload["project_folder"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    validate_source_arc_mask(project_dir)
    job_id = time.strftime("%Y%m%d_%H%M%S")
    job_dir = project_dir / LENS_MODEL_RESULT_DIRNAME
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.json"
    log_path = job_dir / "run.log"
    status_path = job_dir / "status.json"
    script_path = project_dir / LENS_MODEL_SCRIPT_NAME
    provided_code = str(payload.get("generated_code") or "")
    provided_config = payload.get("generated_model_config") or payload.get("model_config") or {}
    if prefer_existing_script and script_path.exists():
        generated_code = script_path.read_text(encoding="utf-8")
        model_config = model_config_from_lens_script(generated_code)
    elif provided_code.strip() and bool(payload.get("generated_code_edited")):
        generated_code = provided_code
        model_config = provided_config if isinstance(provided_config, dict) else {}
    else:
        generated_code, model_config = generate_lens_script(
            payload | {"output_dir": str(job_dir)},
            job_dir=job_dir,
            default_data_dir=DEFAULT_DATA_DIR,
        )
    validate_lens_job_payload(payload, model_config)
    if not (prefer_existing_script and script_path.exists()):
        script_path.write_text(generated_code, encoding="utf-8")
    return {
        "payload": payload,
        "artifact_manifest": artifact_manifest,
        "project_dir": project_dir,
        "job_id": job_id,
        "job_dir": job_dir,
        "config_path": config_path,
        "log_path": log_path,
        "status_path": status_path,
        "script_path": script_path,
        "generated_code": generated_code,
        "model_config": model_config,
    }


def start_sciama_lens_svi_job(payload, *, use_all_gpu_candidates=False):
    active_job = find_active_lens_job()
    if active_job:
        return {
            **active_job,
            "message": f"Lens job {active_job['job_id']} is already running; not starting a second job.",
            "existing_job": True,
        }

    prepared = prepare_lens_job_inputs(payload, prefer_existing_script=True)
    payload = prepared["payload"]
    model_config = prepared["model_config"]
    svi_config = model_config.get("svi", {}) if isinstance(model_config, dict) else {}
    project_dir = prepared["project_dir"]
    job_id = prepared["job_id"]
    job_dir = prepared["job_dir"]
    config_path = prepared["config_path"]
    log_path = prepared["log_path"]
    status_path = prepared["status_path"]
    script_path = prepared["script_path"]
    project_name = payload.get("project_name") or payload.get("project_id") or project_dir.name
    safe_project = safe_sciama_name(project_name)
    remote_base = str(payload.get("sciama_remote_base") or SCIAMA_REMOTE_BASE).rstrip("/")
    remote_dir = f"{remote_base}/{safe_project}_slim_{job_id}"
    stage_dir = RUNS_ROOT / "_sciama_upload" / f"{safe_project}_slim_{job_id}"
    slurm_job_name = f"HGUI_{safe_project[:12]}_{job_id[-6:]}"
    requested_gres = str(payload.get("sciama_gres") or "").strip()
    requested_partition = str(payload.get("sciama_partition") or "gpu.q").strip() or "gpu.q"
    candidate_mode = "all_gpu" if use_all_gpu_candidates else "default"
    if requested_gres:
        gpu_candidates = [
            {
                "partition": requested_partition,
                "gres": requested_gres,
                "label": f"{requested_gres} on {requested_partition}",
            }
        ]
        candidate_mode = "custom"
    else:
        source_candidates = SCIAMA_ALL_GPU_REQUEST_CANDIDATES if use_all_gpu_candidates else SCIAMA_GPU_REQUEST_CANDIDATES
        gpu_candidates = [dict(candidate) for candidate in source_candidates]
    gres_candidates = [str(candidate["gres"]) for candidate in gpu_candidates]
    first_gpu_candidate = gpu_candidates[0]
    gpu_candidate_summary = ",".join(f"{candidate['partition']}:{candidate['gres']}" for candidate in gpu_candidates)
    config = {
        "job_id": job_id,
        "runner": str(SCIAMA_GPU_RUNNER),
        "runner_type": "sciama_gpu",
        "project_id": payload.get("project_id"),
        "project_name": payload.get("project_name"),
        "project_dir": str(project_dir),
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "status_path": str(status_path),
        "script_path": str(script_path),
        "stage_dir": str(stage_dir),
        "remote_base": remote_base,
        "remote_dir": remote_dir,
        "slim_files": lens_sciama_slim_files(project_dir, model_config),
        "conda_activate": str(payload.get("sciama_conda_activate") or SCIAMA_CONDA_ACTIVATE),
        "partition": str(first_gpu_candidate["partition"]),
        "gres": str(first_gpu_candidate["gres"]),
        "gres_candidates": gres_candidates,
        "gpu_candidates": gpu_candidates,
        "gpu_candidate_mode": candidate_mode,
        "slurm_job_name": slurm_job_name,
        "model_config": model_config,
        "project_artifacts": prepared["artifact_manifest"],
        "max_iter_parametric": int(svi_config.get("max_iter_parametric", payload.get("max_iter_parametric", 10000))),
        "max_iter_pixelated": int(svi_config.get("max_iter_pixelated", payload.get("max_iter_pixelated", 10000))),
        "num_chains": int(svi_config.get("num_chains", payload.get("num_chains", 1))),
        "source_grid_scale": float(model_config_value(model_config, "source_grid.scale1", payload.get("source_grid_scale", 1.0))),
        "seed": int(svi_config.get("seed", payload.get("seed", 100))),
    }
    atomic_write_json(config_path, config)
    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Sciama GPU Lens SVI job queued.",
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "runner": "sciama_gpu",
            "remote_dir": remote_dir,
            "partition": str(first_gpu_candidate["partition"]),
            "gres": str(first_gpu_candidate["gres"]),
            "gres_candidates": gres_candidates,
            "gpu_candidates": gpu_candidates,
            "gpu_candidate_mode": candidate_mode,
            "progress": {"fraction": 0.0, "message": "Sciama GPU Lens SVI job queued."},
        },
    )
    log_path.write_text(
        f"SCIAMA_GPU_JOB job_id={job_id} mode={candidate_mode} remote_dir={remote_dir} gpu_candidates={gpu_candidate_summary}\n",
        encoding="utf-8",
    )
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(sys.executable), "-u", str(SCIAMA_GPU_RUNNER), str(config_path)],
            cwd=str(GUI_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    return {
        "job_id": job_id,
        "state": "queued",
        "message": f"Started Sciama GPU Lens SVI job {job_id}.",
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "script_path": str(script_path),
        "generated_code": prepared["generated_code"],
        "model_config": prepared["model_config"],
        "project_artifacts": prepared["artifact_manifest"],
        "remote_dir": remote_dir,
        "runner": "sciama_gpu",
        "partition": str(first_gpu_candidate["partition"]),
        "gres": str(first_gpu_candidate["gres"]),
        "gres_candidates": gres_candidates,
        "gpu_candidates": gpu_candidates,
        "gpu_candidate_mode": candidate_mode,
        "progress": {
            "fraction": 0.01,
            "message": f"Sciama GPU Lens SVI job started; trying {first_gpu_candidate['label']} first.",
            "stages": {
                "parametric": {"fraction": 0.0, "message": "Parametric SVI pending.", "state": "pending"},
                "pixelated": {"fraction": 0.0, "message": "Pixelated SVI pending.", "state": "pending"},
                        "solver": {"fraction": 0.0, "message": "Exact-solver SVI pending.", "state": "pending"},
                "semilinear": {"fraction": 0.0, "message": "Semilinear source inversion pending.", "state": "pending"},
            },
        },
    }


def parse_lens_progress_from_log(log_path):
    def stage_progress(text, stage_name, next_marker=None, done=False):
        marker = f"LENS_STAGE {stage_name}"
        title = "Parametric" if stage_name == "parametric" else "Pixelated"
        if marker not in text:
            return {"fraction": 0.0, "message": f"{title} SVI pending.", "state": "pending"}

        stage_text = text.split(marker, 1)[1]
        if next_marker and next_marker in stage_text:
            stage_text = stage_text.split(next_marker, 1)[0]

        chain_matches = re.findall(rf"LENS_CHAIN\s+{stage_name}\s+(\d+)\s+(\d+)", stage_text)
        current_chain = None
        total_chains = None
        if chain_matches:
            current_chain, total_chains = (int(chain_matches[-1][0]), int(chain_matches[-1][1]))
            stage_text = stage_text.rsplit(f"LENS_CHAIN {stage_name} {current_chain} {total_chains}", 1)[-1]

        matches = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", stage_text)
        if not matches:
            if current_chain is not None and total_chains:
                local = 1.0 if done else max(0.0, (current_chain - 1) / max(total_chains, 1))
                return {
                    "fraction": local,
                    "message": f"{title} SVI chain {current_chain}/{total_chains}",
                    "state": "completed" if done else "running",
                }
            return {"fraction": 1.0 if done else 0.0, "message": f"{title} SVI {'completed' if done else 'running'}.", "state": "completed" if done else "running"}

        step, total = (int(matches[-1][0]), int(matches[-1][1]))
        if done:
            step = total
        local_single = min(1.0, max(0.0, step / max(total, 1)))
        if current_chain is not None and total_chains:
            local = min(1.0, max(0.0, ((current_chain - 1) + local_single) / max(total_chains, 1)))
            message = f"{title} SVI chain {current_chain}/{total_chains} {step}/{total}"
        else:
            local = local_single
            message = f"{title} SVI {step}/{total}"
        state = "completed" if local >= 1.0 else "running"
        return {
            "fraction": local,
            "message": message,
            "state": state,
            "step": step,
            "total": total,
        }

    def semilinear_progress(text, done=False):
        if "LENS_STAGE outputs (semilinear disabled)" in text:
            return {
                "fraction": 1.0,
                "message": "Semilinear source inversion disabled.",
                "state": "completed",
            }
        marker = "LENS_STAGE semilinear"
        if marker not in text:
            return {
                "fraction": 0.0,
                "message": "Semilinear source inversion pending.",
                "state": "pending",
            }
        stage_text = text.split(marker, 1)[1]
        matches = re.findall(
            r"LENS_SEMILINEAR\s+([0-9]+(?:\.[0-9]+)?)%\s+([^\r\n]+)",
            stage_text,
        )
        if not matches:
            return {
                "fraction": 1.0 if done else 0.0,
                "message": (
                    "Semilinear source inversion completed."
                    if done
                    else "Semilinear source inversion running."
                ),
                "state": "completed" if done else "running",
            }
        percent, message = matches[-1]
        fraction = 1.0 if done else min(1.0, max(0.0, float(percent) / 100.0))
        return {
            "fraction": fraction,
            "message": message,
            "state": "completed" if fraction >= 1.0 else "running",
        }

    if not log_path.exists():
        return {
            "fraction": 0.0,
            "message": "Waiting for lens SVI log.",
            "stages": {
                "parametric": {"fraction": 0.0, "message": "Parametric SVI pending.", "state": "pending"},
                "pixelated": {"fraction": 0.0, "message": "Pixelated SVI pending.", "state": "pending"},
                        "solver": {"fraction": 0.0, "message": "Exact-solver SVI pending.", "state": "pending"},
                "semilinear": {"fraction": 0.0, "message": "Semilinear source inversion pending.", "state": "pending"},
            },
        }
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "LENS_STAGE parametric" not in text and "LENS_STAGE pixelated" not in text:
        sciama_stages = [
            ("SCIAMA_STAGE download", 0.95, "Downloading Sciama lens model results."),
            ("SCIAMA_STAGE run", 0.12, "Running Lens SVI on Sciama GPU."),
            ("SCIAMA_STAGE upload", 0.08, "Uploading slim project to Sciama."),
            ("SCIAMA_GPU_JOB", 0.02, "Preparing Sciama GPU Lens SVI job."),
        ]
        for marker, fraction, message in sciama_stages:
            if marker in text:
                return {
                    "fraction": fraction,
                    "message": message,
                    "stage": "sciama",
                    "stages": {
                        "parametric": {"fraction": 0.0, "message": "Parametric SVI pending.", "state": "pending"},
                        "pixelated": {"fraction": 0.0, "message": "Pixelated SVI pending.", "state": "pending"},
                        "solver": {"fraction": 0.0, "message": "Exact-solver SVI pending.", "state": "pending"},
                        "semilinear": {"fraction": 0.0, "message": "Semilinear source inversion pending.", "state": "pending"},
                    },
                }
    completed = "LONG_SVI_RUN_DONE" in text
    pixelated_started = "LENS_STAGE pixelated" in text
    # The exact-solver stage is optional: it only runs when the two-image solver is on.
    solver_started = "LENS_STAGE solver" in text
    semilinear_started = "LENS_STAGE semilinear" in text
    stages = {
        "parametric": stage_progress(text, "parametric", next_marker="LENS_STAGE pixelated", done=pixelated_started or completed),
        "pixelated": stage_progress(
            text,
            "pixelated",
            next_marker="LENS_STAGE solver" if solver_started else "LENS_STAGE semilinear",
            done=solver_started or semilinear_started or completed,
        ),
        "semilinear": semilinear_progress(text, done=completed),
    }
    if solver_started:
        stages["solver"] = stage_progress(
            text,
            "solver",
            next_marker="LENS_STAGE semilinear",
            done=semilinear_started or completed,
        )
    if "Traceback (most recent call last)" in text or "IndentationError:" in text or "SyntaxError:" in text:
        last_error = next(
            (line.strip() for line in reversed(text.splitlines()) if line.strip()),
            "Lens SVI failed.",
        )
        return {"fraction": 1.0, "message": f"Lens SVI failed: {last_error}", "failed": True, "stages": stages}

    if completed:
        return {"fraction": 1.0, "message": "Lens model completed.", "stages": stages}

    stage = (
        "semilinear"
        if semilinear_started
        else (
            "solver"
            if solver_started
            else ("pixelated" if pixelated_started else "parametric")
        )
    )
    stage_state = stages[stage]
    local = stage_state["fraction"]
    if stage == "parametric":
        fraction = 0.02 + 0.48 * local
    elif stage == "pixelated":
        fraction = 0.52 + 0.42 * local
    else:
        fraction = 0.94 + 0.05 * local
    return {
        "fraction": fraction,
        "message": stage_state["message"],
        "stage": stage,
        "step": stage_state.get("step"),
        "total": stage_state.get("total"),
        "stages": stages,
    }


LENS_MODEL_PREVIEW_GLOBS = (
    "latest_four_panel_comparison*.png",
    "chain_*_lens_model_comparison.png",
    "pixelated_model_comparison.png",
    "parametric_model_comparison.png",
    "figure_*.png",
)


def latest_lens_preview_in_job(job_dir):
    for pattern in LENS_MODEL_PREVIEW_GLOBS:
        previews = sorted(job_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if previews:
            return previews[0]
    return None


def lens_result_url_for_job(value, job_dir):
    text = str(value or "").strip()
    if not text:
        return text
    resolved = lens_result_path_for_job(text, job_dir)
    if resolved is not None:
        try:
            return relative_url(resolved)
        except ValueError:
            return text
    return text


def lens_result_path_for_job(value, job_dir):
    text = str(value or "").strip()
    if not text:
        return None
    candidates = []
    direct_path = Path(text).expanduser()
    if direct_path.exists():
        candidates.append(direct_path)
    if text.startswith("/"):
        relative_text = text.lstrip("/")
        candidates.extend(
            [
                GUI_ROOT / relative_text,
                job_dir.parent / relative_text,
                job_dir / Path(relative_text).name,
            ]
        )
    else:
        candidates.extend([job_dir / text, job_dir.parent / text, GUI_ROOT / text])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def lens_panel_payloads_for_job(panel_fits, job_dir):
    if not isinstance(panel_fits, dict):
        return {}
    labels = {
        "data": "Data",
        "model": "LensModel",
        "data_minus_model_over_rms": "Residual",
        "source": "Source",
        "data_minus_lens_light": "Lens Light subtracted",
        "lensed_source_without_lens_light": "Lensed source without lens light",
        "lensed_arc1": "Lensed arc 1",
        "lensed_arc2": "Lensed arc 2",
        "source1": "Source 1",
        "source2": "Source 2",
    }
    payloads = {}
    for key, value in panel_fits.items():
        path = lens_result_path_for_job(value, job_dir)
        if path is None:
            continue
        try:
            image = fits_payload_for_gui(path)
        except Exception:
            continue
        image["label"] = labels.get(key, key.replace("_", " "))
        image["panel_key"] = key
        image["fits_url"] = relative_url(path)
        payloads[key] = image
    return payloads


def normalize_lens_chain_previews(chain_previews, job_dir, include_panels=False):
    normalized = []
    for item in chain_previews or []:
        if not isinstance(item, dict):
            continue
        fixed = dict(item)
        for key in (
            "preview_url",
            "parametric_url",
            "pixelated_url",
            "semilinear_url",
        ):
            if fixed.get(key):
                fixed[key] = lens_result_url_for_job(fixed[key], job_dir)
        try:
            chain_number = int(fixed.get("chain"))
        except (TypeError, ValueError):
            chain_number = None
        if chain_number is not None:
            display_candidates = list(
                job_dir.glob(
                    f"chain_{chain_number:02d}_lens_model_comparison_six_panel*.png"
                )
            )
            current_preview = lens_result_path_for_job(fixed.get("preview_url"), job_dir)
            if current_preview is not None:
                display_candidates.append(current_preview)
            if display_candidates:
                latest_display = max(
                    display_candidates,
                    key=lambda candidate: candidate.stat().st_mtime,
                )
                fixed["preview_url"] = relative_url(latest_display)
        if include_panels and isinstance(fixed.get("pixelated_panel_fits"), dict):
            fixed["pixelated_panels"] = lens_panel_payloads_for_job(fixed["pixelated_panel_fits"], job_dir)
        if include_panels and isinstance(fixed.get("parametric_panel_fits"), dict):
            fixed["parametric_panels"] = lens_panel_payloads_for_job(fixed["parametric_panel_fits"], job_dir)
        if include_panels and isinstance(fixed.get("semilinear_panel_fits"), dict):
            fixed["semilinear_panels"] = lens_panel_payloads_for_job(fixed["semilinear_panel_fits"], job_dir)
        normalized.append(fixed)
    attach_lens_chain_mass_parameters(normalized, job_dir)
    return normalized


def lens_state_json_value(value):
    import numpy as np

    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    if array.size == 1:
        return float(array[0])
    return [float(item) for item in array]


def lens_state_scalar(params, key):
    if not isinstance(params, dict) or key not in params:
        return None
    value = lens_state_json_value(params[key])
    if isinstance(value, list):
        return value[0] if value else None
    return value


def lens_state_vector(params, key):
    value = lens_state_json_value(params.get(key) if isinstance(params, dict) else None)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def mass_parameters_from_lens_state_median(params):
    if not isinstance(params, dict):
        return {}
    e_values = lens_state_vector(params, "e_1")
    center_values = lens_state_vector(params, "center_1")
    shear_values = lens_state_vector(params, "gamma_sheer_1")
    summary = {}
    eta = lens_state_scalar(params, "eta")
    if eta is not None:
        summary["eta"] = eta
    key_map = (
        ("theta_E_1", "theta_E"),
        ("gamma_1", "gamma"),
    )
    for state_key, output_key in key_map:
        value = lens_state_scalar(params, state_key)
        if value is not None:
            summary[output_key] = value
    if len(e_values) >= 2:
        summary["e1"] = e_values[0]
        summary["e2"] = e_values[1]
    if len(center_values) >= 2:
        summary["center_x"] = center_values[0]
        summary["center_y"] = center_values[1]
    if len(shear_values) >= 2:
        summary["gamma1_ext"] = shear_values[0]
        summary["gamma2_ext"] = shear_values[1]
    sis_theta = lens_state_scalar(params, "theta_E_SIS_s1")
    if sis_theta is not None:
        summary["theta_E_SIS_s1"] = sis_theta
    sis_x = lens_state_scalar(params, "center_1_SIS_s1")
    sis_y = lens_state_scalar(params, "center_2_SIS_s1")
    if sis_x is not None:
        summary["SIS_s1_center_x"] = sis_x
    if sis_y is not None:
        summary["SIS_s1_center_y"] = sis_y
    return {key: value for key, value in summary.items() if value is not None}


def load_lens_chain_mass_parameters(job_dir):
    import pickle
    import __main__

    def model_lens(*_args, **_kwargs):
        return None

    old_model_lens = getattr(__main__, "model_lens", None)
    had_model_lens = hasattr(__main__, "model_lens")
    setattr(__main__, "model_lens", model_lens)
    inserted_paths = []
    for path in (str(job_dir.parent), str(job_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted_paths.append(path)
    try:
        for name in ("pixelated_states.pkl", "parametric_states.pkl"):
            path = job_dir / name
            if not path.exists():
                continue
            try:
                states = pickle.loads(path.read_bytes())
            except Exception:
                continue
            if not isinstance(states, list):
                continue
            summaries = []
            for state in states:
                median = state.get("median") if isinstance(state, dict) else None
                summaries.append(mass_parameters_from_lens_state_median(median))
            if any(summaries):
                return summaries
    finally:
        for path in inserted_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        if had_model_lens:
            setattr(__main__, "model_lens", old_model_lens)
        else:
            try:
                delattr(__main__, "model_lens")
            except AttributeError:
                pass
    return []


def attach_lens_chain_mass_parameters(chain_previews, job_dir):
    if not chain_previews or all(isinstance(item, dict) and item.get("mass_parameters") for item in chain_previews):
        return chain_previews
    try:
        summaries = load_lens_chain_mass_parameters(job_dir)
    except Exception:
        summaries = []
    if not summaries:
        return chain_previews
    for index, item in enumerate(chain_previews):
        if not isinstance(item, dict) or item.get("mass_parameters"):
            continue
        try:
            chain_index = int(item.get("chain", index + 1)) - 1
        except Exception:
            chain_index = index
        if 0 <= chain_index < len(summaries) and summaries[chain_index]:
            item["mass_parameters"] = summaries[chain_index]
    return chain_previews


def discover_lens_chain_previews(job_dir, summary=None):
    summary = summary if isinstance(summary, dict) else {}
    selected_index = summary.get("selected_chain_index", 0)
    try:
        selected_index = int(selected_index)
    except Exception:
        selected_index = 0
    try:
        num_chains = int(summary.get("num_chains") or 0)
    except Exception:
        num_chains = 0

    discovered_indices = set()
    for path in job_dir.glob("chain_*_lens_model_comparison.png"):
        parts = path.stem.split("_")
        if len(parts) >= 2:
            try:
                discovered_indices.add(int(parts[1]))
            except Exception:
                pass
    if num_chains > 0:
        indices = list(range(1, num_chains + 1))
    else:
        indices = sorted(discovered_indices)

    previews = []
    for index in indices:
        chain_tag = f"chain_{index:02d}"
        preview_path = job_dir / f"{chain_tag}_lens_model_comparison.png"
        if not preview_path.exists():
            figure_fallback = job_dir / f"figure_{index:02d}.png"
            preview_path = figure_fallback if figure_fallback.exists() else None
        if not preview_path:
            continue
        item = {
            "chain": index,
            "label": f"Chain {index}",
            "selected": bool(index - 1 == selected_index),
            "preview_url": relative_url(preview_path),
        }
        parametric_png = job_dir / f"{chain_tag}_parametric_model_comparison.png"
        pixelated_png = job_dir / f"{chain_tag}_pixelated_model_comparison.png"
        semilinear_png = job_dir / f"{chain_tag}_semilinear_model_comparison.png"
        if parametric_png.exists():
            item["parametric_url"] = relative_url(parametric_png)
        if pixelated_png.exists():
            item["pixelated_url"] = relative_url(pixelated_png)
        if semilinear_png.exists():
            item["semilinear_url"] = relative_url(semilinear_png)
        previews.append(item)
    attach_lens_chain_mass_parameters(previews, job_dir)
    return previews


def read_lens_job_status(job_id, include_panels=False):
    status_path = status_path_for_lens_job(job_id)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    job_dir = status_path.parent
    config_path = job_dir / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    status.setdefault("job_id", config.get("job_id", job_id))
    status.setdefault("config_path", str(config_path))
    status.setdefault("script_path", config.get("script_path", str(job_dir.parent / LENS_MODEL_SCRIPT_NAME)))
    status.setdefault("log_path", config.get("log_path", str(job_dir / "run.log")))
    if config.get("model_config") and not status.get("model_config"):
        status["model_config"] = config["model_config"]
    log_path = Path(status.get("log_path") or job_dir / "run.log")
    parsed_progress = parse_lens_progress_from_log(log_path)
    if status.get("state") not in LENS_TERMINAL_STATES:
        progress = parsed_progress
        status["state"] = "failed" if progress.get("failed") else "running"
        status["progress"] = progress
        status["message"] = status["progress"].get("message", "Lens SVI running.")
    elif "stages" in parsed_progress:
        progress = dict(status.get("progress") or {})
        progress["stages"] = parsed_progress["stages"]
        status["progress"] = progress
    if status.get("state") not in LENS_TERMINAL_STATES:
        return status
    latest_preview = latest_lens_preview_in_job(job_dir)
    if latest_preview:
        status["preview_url"] = relative_url(latest_preview)
    summary_path = job_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status["run_summary"] = summary
            if isinstance(summary.get("chain_previews"), list):
                status["chain_previews"] = normalize_lens_chain_previews(
                    summary["chain_previews"],
                    job_dir,
                    include_panels=include_panels,
                )
            if isinstance(summary.get("mass_parameters"), dict):
                status["mass_parameters"] = summary["mass_parameters"]
        except Exception:
            pass
    elif isinstance(status.get("chain_previews"), list):
        status["chain_previews"] = normalize_lens_chain_previews(
            status["chain_previews"],
            job_dir,
            include_panels=include_panels,
        )
    if not status.get("chain_previews"):
        status["chain_previews"] = discover_lens_chain_previews(job_dir, summary)
    return status


def latest_lens_model_payload(project_id=None, include_panels=False):
    if project_id:
        try:
            search_root = Path(normalize_project_payload(read_project_payload(project_id)).get("project_folder")).expanduser()
        except Exception:
            search_root = project_folder(project_id)
        preview_globs = [
            f"{LENS_MODEL_RESULT_DIRNAME}/latest_four_panel_comparison*.png",
            f"{LENS_MODEL_RESULT_DIRNAME}/chain_*_lens_model_comparison.png",
            f"{LENS_MODEL_RESULT_DIRNAME}/pixelated_model_comparison.png",
            f"{LENS_MODEL_RESULT_DIRNAME}/parametric_model_comparison.png",
            f"{LENS_MODEL_RESULT_DIRNAME}/figure_*.png",
            "lens_model_runs/lens_svi_*/latest_four_panel_comparison*.png",
        ]
    else:
        search_root = GUI_ROOT / "runs"
        preview_globs = [
            f"*/{LENS_MODEL_RESULT_DIRNAME}/latest_four_panel_comparison*.png",
            f"*/{LENS_MODEL_RESULT_DIRNAME}/chain_*_lens_model_comparison.png",
            f"*/{LENS_MODEL_RESULT_DIRNAME}/pixelated_model_comparison.png",
            f"*/{LENS_MODEL_RESULT_DIRNAME}/parametric_model_comparison.png",
            f"*/{LENS_MODEL_RESULT_DIRNAME}/figure_*.png",
            "**/lens_svi_*/latest_four_panel_comparison*.png",
        ]
    previews = []
    for pattern in preview_globs:
        previews = sorted(search_root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if previews:
            break
    if not previews:
        raise FileNotFoundError("No lens model figure found.")

    preview_path = previews[0]
    job_dir = preview_path.parent
    payload = {
        "message": "Loaded latest GUI lens model product.",
        "preview_url": relative_url(preview_path),
        "job_dir": str(job_dir),
    }

    status_path = job_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            for key in ("job_id", "state", "progress", "model_config"):
                if key in status:
                    payload[key] = status[key]
        except Exception:
            pass

    summary_path = job_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["run_summary"] = summary
            if isinstance(summary.get("chain_previews"), list):
                payload["chain_previews"] = normalize_lens_chain_previews(
                    summary["chain_previews"],
                    job_dir,
                    include_panels=include_panels,
                )
            if isinstance(summary.get("mass_parameters"), dict):
                payload["mass_parameters"] = summary["mass_parameters"]
        except Exception:
            pass
    if not payload.get("chain_previews"):
        payload["chain_previews"] = discover_lens_chain_previews(job_dir, summary)

    config_path = job_dir / "config.json"
    model_config = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_config = config.get("model_config", payload.get("model_config"))
            payload["model_config"] = model_config
            payload["script_path"] = config.get("script_path")
        except Exception:
            pass

    project_script = job_dir.parent / LENS_MODEL_SCRIPT_NAME
    script_path = project_script if project_script.exists() else job_dir / LENS_MODEL_SCRIPT_NAME
    if script_path.exists():
        payload["script_path"] = str(script_path)

    if model_config:
        code_payload = dict(model_config)
        mass_prior = dict(code_payload.pop("mass_prior", {}) or {})
        project_dir = job_dir.parent.resolve()
        generated_code, normalized_config = generate_lens_script(
            {
                **code_payload,
                **mass_prior,
                "output_dir": str(job_dir),
                "project_folder": str(project_dir),
            },
            job_dir=job_dir,
            default_data_dir=project_dir,
        )
        payload["generated_code"] = generated_code
        payload["model_config"] = normalized_config

    return payload


def project_state_has_content(state):
    if not isinstance(state, dict):
        return False
    euclid = state.get("euclid") if isinstance(state.get("euclid"), dict) else {}
    image = state.get("image") if isinstance(state.get("image"), dict) else {}
    mask = state.get("mask") if isinstance(state.get("mask"), dict) else {}
    psf = state.get("psf") if isinstance(state.get("psf"), dict) else {}
    lens = state.get("lens") if isinstance(state.get("lens"), dict) else {}
    cutout_artifact = image.get("cutout_artifact") if isinstance(image.get("cutout_artifact"), dict) else {}
    return any(
        str(value or "").strip()
        for value in (
            euclid.get("image_path"),
            euclid.get("rms_map_path"),
            euclid.get("rms_bundle_path"),
            image.get("source_path"),
            image.get("input_path"),
            cutout_artifact.get("fits_path"),
            cutout_artifact.get("preview_url"),
            mask.get("saved_path"),
            mask.get("lens_light_preview_url"),
            psf.get("input_path"),
            psf.get("job_id"),
            lens.get("job_id"),
        )
    ) or bool(
        euclid.get("image")
        or euclid.get("image_options")
        or mask.get("data")
        or mask.get("conjugate_points")
        or mask.get("conjugate_points_source1")
        or mask.get("conjugate_points_source2")
        or lens.get("preview_data")
    )


def project_folder_has_products(project_dir):
    project_dir = Path(project_dir).expanduser()
    if not project_dir.exists():
        return False
    product_paths = (
        project_dir / "Data_cutout.fits",
        project_dir / "RMS_map.fits",
        project_dir / "PSF_model.fits",
        project_dir / LENS_MODEL_RESULT_DIRNAME / "status.json",
        project_dir / LENS_MODEL_RESULT_DIRNAME / "summary.json",
        project_dir / LENS_MODEL_RESULT_DIRNAME / "latest_four_panel_comparison.png",
        project_dir / LENS_LIGHT_RESULT_DIRNAME / "status.json",
        project_dir / LENS_LIGHT_RESULT_DIRNAME / "lens_light_subtracted.fits",
    )
    if any(path.exists() for path in product_paths):
        return True
    manifest_path = project_dir / "project_artifacts.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files") if isinstance(manifest, dict) else {}
            return any(str(path or "").strip() for path in (files or {}).values())
        except Exception:
            return True
    return False


def project_payload_has_content(payload):
    payload = dict(payload or {})
    if project_state_has_content(payload.get("state")):
        return True
    folder_value = str(payload.get("project_folder") or "").strip()
    if not folder_value:
        return False
    folder = Path(folder_value).expanduser()
    if not folder.is_absolute():
        folder = resolve_workspace_path(folder)
    return project_folder_has_products(folder)


def project_payload_is_restore_candidate(payload):
    payload = dict(payload or {})
    if not payload.get("project_id"):
        return False
    folder_value = str(payload.get("project_folder") or "").strip()
    if folder_value:
        folder = Path(folder_value).expanduser()
        if not folder.is_absolute():
            folder = resolve_workspace_path(folder)
        if folder.resolve() == DEFAULT_DATA_DIR.resolve() and not project_state_has_content(payload.get("state")):
            return False
    return project_payload_has_content(payload)


def synthetic_project_payload():
    payload = normalize_project_payload(
        {
            "project_id": time.strftime("%Y%m%d_%H%M%S"),
            "project_name": "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": {},
        },
        create_folder=False,
    )
    payload["_project_exists"] = False
    return payload


def normalized_existing_project_payload(payload):
    payload = normalize_project_payload(payload, create_folder=False)
    payload["_project_exists"] = True
    return payload


def latest_restorable_project_payload():
    paths = sorted(PROJECT_FOLDER_DIR.glob("*/project.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            payload = normalize_project_payload(json.loads(path.read_text(encoding="utf-8")), create_folder=False)
        except Exception:
            continue
        if project_payload_is_restore_candidate(payload):
            return normalized_existing_project_payload(payload)
    return None


def project_payload_or_default():
    if PROJECT_STATE_PATH.exists():
        try:
            latest_payload = json.loads(PROJECT_STATE_PATH.read_text(encoding="utf-8"))
            latest_project_path = project_payload_path(latest_payload)
            if latest_project_path.exists():
                payload = normalize_project_payload(json.loads(latest_project_path.read_text(encoding="utf-8")), create_folder=False)
                return normalized_existing_project_payload(payload)
            latest_id = str(latest_payload.get("project_id") or "").strip()
            if latest_id and project_file(latest_id).exists():
                payload = normalize_project_payload(json.loads(project_file(latest_id).read_text(encoding="utf-8")), create_folder=False)
                return normalized_existing_project_payload(payload)
            latest_payload = normalize_project_payload(latest_payload, create_folder=False)
            if project_payload_is_restore_candidate(latest_payload):
                return normalized_existing_project_payload(latest_payload)
        except Exception:
            pass
    fallback = latest_restorable_project_payload()
    if fallback is not None:
        return fallback
    return synthetic_project_payload()


def project_file(project_id):
    return project_folder(project_id) / "project.json"


def project_folder(project_id):
    return PROJECT_FOLDER_DIR / safe_filename(str(project_id))


def resolve_project_folder_reference(value):
    text = str(value or "").strip()
    if not text:
        return None
    folder = Path(text).expanduser()
    if folder.is_absolute():
        return folder.resolve()
    if len(folder.parts) == 1:
        return (PROJECT_FOLDER_DIR / safe_filename(text)).resolve()
    return resolve_workspace_path(folder)


def project_payload_path(payload):
    folder_value = str((payload or {}).get("project_folder") or "").strip()
    if folder_value:
        folder = resolve_project_folder_reference(folder_value)
        return folder / "project.json"
    return project_file((payload or {}).get("project_id", time.strftime("%Y%m%d_%H%M%S")))


def read_project_payload(project_id):
    path = project_file(project_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if PROJECT_STATE_PATH.exists():
        latest_payload = project_listing_metadata(PROJECT_STATE_PATH)
        if str(latest_payload.get("project_id") or "") == str(project_id):
            latest_path = project_payload_path(latest_payload)
            if latest_path.exists():
                return json.loads(latest_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Project not found: {project_id}")


def normalize_project_payload(payload, create_folder=False):
    payload = dict(payload or {})
    payload.setdefault("project_id", time.strftime("%Y%m%d_%H%M%S"))
    payload.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    payload.setdefault("state", {})
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    lens_state = state.get("lens") if isinstance(state.get("lens"), dict) else {}
    try:
        payload["rating"] = max(0, min(5, int(round(float(payload.get("rating", lens_state.get("rating", 0)) or 0)))))
    except Exception:
        payload["rating"] = 0
    comment = str(payload.get("comment") or "").strip()
    payload["comment"] = comment[:5000]
    project_name = str(payload.get("project_name") or "").strip()
    if not project_name or timestamp_like(project_name):
        project_name = project_name_from_state_coordinates(payload) or project_name or str(payload["project_id"])
    payload["project_name"] = project_name
    folder_value = str(payload.get("project_folder") or "").strip()
    if folder_value:
        folder = Path(folder_value).expanduser()
        if not folder.is_absolute():
            folder = resolve_workspace_path(folder)
    else:
        folder = project_folder(payload["project_id"])
    payload["project_folder"] = str(folder.resolve())
    payload["project_name"] = canonical_project_name(payload.get("project_name"), payload.get("project_id"), payload.get("project_folder"))
    if isinstance(payload.get("state"), dict):
        payload["state"]["data_folder"] = payload["project_folder"]
    if create_folder:
        project_dir = Path(payload["project_folder"])
        project_dir.mkdir(parents=True, exist_ok=True)
        sync_project_runtime_files(project_dir)
    return payload


def save_project_preview_data_url(payload, data_url_key, filename, message):
    data_url = str(payload.get(data_url_key) or "").strip()
    if not data_url:
        raise ValueError(f"{data_url_key} is required")
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:image/png;base64"):
        raise ValueError(f"{data_url_key} must be a PNG data URL")
    image_bytes = base64.b64decode(encoded, validate=True)
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Preview PNG is too large")
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Preview payload is not a PNG")

    folder_value = str(payload.get("project_folder") or "").strip()
    project_id = str(payload.get("project_id") or "").strip()
    if folder_value:
        project_dir = resolve_project_folder_reference(folder_value)
    elif project_id:
        project_dir = project_folder(project_id)
    else:
        raise ValueError("project_id or project_folder is required")

    project_dir = project_dir.resolve()
    try:
        project_dir.relative_to(PROJECT_FOLDER_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Project preview can only be saved inside the GUI runs folder") from exc

    if filename == "Data_cutout_preview.png" and not project_data_cutout_path(project_dir).exists():
        raise FileNotFoundError(f"Data_cutout.fits not found in {project_dir}")

    preview_dir = project_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    png_path = preview_dir / filename
    tmp_path = png_path.with_suffix(".png.tmp")
    tmp_path.write_bytes(image_bytes)
    tmp_path.replace(png_path)
    invalidate_project_list_cache()
    return {
        "message": message,
        "preview_url": relative_url_with_mtime(png_path),
        "preview_path": str(png_path),
    }


def save_project_thumbnail(payload):
    return save_project_preview_data_url(
        payload,
        "thumbnail_data_url",
        "Project_thumbnail.png",
        "Project thumbnail saved.",
    )


def save_project_cutout_preview(payload):
    return save_project_preview_data_url(
        payload,
        "preview_data_url",
        "Data_cutout_preview.png",
        "Data cutout preview saved.",
    )


def apply_replacements(text, replacements):
    """Apply every replacement in a single left-to-right pass.

    Chained str.replace() calls re-scan text an earlier rule already rewrote, so a
    rename whose new id contains the old id (OLD -> OLD_VIS) matches a second time
    and yields OLD_VIS_VIS. Matching longest-first in one pass leaves already
    substituted regions alone.
    """
    seen = {}
    for old, new in replacements:
        if old and old not in seen:
            seen[old] = new
    if not seen:
        return text
    # Text that already holds a replacement result must survive untouched: when the
    # new id contains the old one, matching the longer result first keeps a second
    # rewrite from appending the suffix again. Skip values that are themselves keys
    # so a genuine A->B, B->C chain is not short-circuited.
    for result in list(seen.values()):
        if result and result not in seen:
            seen[result] = result
    ordered = sorted(seen, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(old) for old in ordered))
    return pattern.sub(lambda match: seen[match.group(0)], text)


def rewrite_text_fields(value, replacements):
    if isinstance(value, dict):
        return {key: rewrite_text_fields(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_text_fields(item, replacements) for item in value]
    if isinstance(value, str):
        return apply_replacements(value, replacements)
    return value


def rewrite_project_text_files(project_dir, replacements):
    text_suffixes = {".json", ".py", ".txt", ".md", ".csv"}
    for path in Path(project_dir).rglob("*"):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text = apply_replacements(text, replacements)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def rename_project_payload(payload):
    payload = dict(payload or {})
    old_id = str(payload.get("project_id") or "").strip()
    new_name = str(payload.get("project_name") or "").strip()
    if not old_id:
        raise ValueError("project_id is required.")
    if not new_name:
        raise ValueError("project_name is required.")

    old_folder_value = str(payload.get("project_folder") or "").strip()
    if old_folder_value:
        old_folder = Path(old_folder_value).expanduser()
        if not old_folder.is_absolute():
            old_folder = resolve_workspace_path(old_folder)
    else:
        old_folder = project_folder(old_id)
    new_id = project_id_from_name(new_name)
    new_folder = old_folder.parent / new_id

    if new_folder.resolve() != old_folder.resolve():
        if new_folder.exists():
            raise ValueError(f"Project folder already exists: {new_folder.name}")
        if old_folder.exists():
            old_folder.rename(new_folder)
        else:
            new_folder.mkdir(parents=True, exist_ok=True)
    else:
        new_folder.mkdir(parents=True, exist_ok=True)

    replacements = [
        (str(old_folder), str(new_folder)),
        (f"/runs/{old_id}/", f"/runs/{new_id}/"),
        (f"runs/{old_id}/", f"runs/{new_id}/"),
        (old_id, new_id),
    ]

    payload["project_id"] = new_id
    payload["project_name"] = new_name
    payload["project_folder"] = str(new_folder)
    payload = rewrite_text_fields(payload, replacements)
    payload["project_id"] = new_id
    payload["project_name"] = new_name
    payload["project_folder"] = str(new_folder)
    if isinstance(payload.get("state"), dict):
        payload["state"]["data_folder"] = str(new_folder)

    rewrite_project_text_files(new_folder, replacements)
    payload = normalize_project_payload(payload, create_folder=True)
    atomic_write_json(project_payload_path(payload), payload)
    atomic_write_json(PROJECT_STATE_PATH, payload)
    return payload


def copy_project_payload(payload):
    source_id = str(payload.get("project_id") or "").strip()
    if not source_id:
        raise ValueError("project_id is required.")
    source_project = normalize_project_payload(read_project_payload(source_id), create_folder=False)
    source_dir = Path(source_project["project_folder"]).expanduser().resolve()
    try:
        source_dir.relative_to(PROJECT_FOLDER_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to copy project outside runs/: {source_dir}") from exc
    if not (source_dir / "project.json").exists():
        raise FileNotFoundError(f"Project folder does not contain project.json: {source_dir}")

    base_name = str(payload.get("project_name") or source_project.get("project_name") or source_id).strip()
    if not base_name:
        base_name = source_id
    copy_name_base = f"{base_name}_copy"
    copy_name = copy_name_base
    copy_id = project_id_from_name(copy_name)
    copy_dir = source_dir.parent / copy_id
    index = 2
    while copy_dir.exists() or project_file(copy_id).exists():
        copy_name = f"{copy_name_base}{index}"
        copy_id = project_id_from_name(copy_name)
        copy_dir = source_dir.parent / copy_id
        index += 1

    shutil.copytree(source_dir, copy_dir)
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    replacements = [
        (str(source_dir), str(copy_dir)),
        (f"/runs/{source_id}/", f"/runs/{copy_id}/"),
        (f"runs/{source_id}/", f"runs/{copy_id}/"),
        (source_id, copy_id),
    ]
    copied_project = rewrite_text_fields(dict(source_project), replacements)
    copied_project["project_id"] = copy_id
    copied_project["project_name"] = copy_name
    copied_project["project_folder"] = str(copy_dir)
    copied_project["saved_at"] = saved_at
    copied_project["created_at"] = saved_at
    copied_project["copied_from_project_id"] = source_id
    copied_project["copied_from_project_name"] = source_project.get("project_name", source_id)
    if isinstance(copied_project.get("state"), dict):
        copied_project["state"]["data_folder"] = str(copy_dir)

    rewrite_project_text_files(copy_dir, replacements)
    copied_project = normalize_project_payload(copied_project, create_folder=True)
    atomic_write_json(project_payload_path(copied_project), copied_project)
    invalidate_project_list_cache()
    return {
        "message": f"Copied project {source_project.get('project_name', source_id)} to {copy_name}.",
        **copied_project,
    }


def save_project_comment(payload):
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id is required.")
    project = normalize_project_payload(read_project_payload(project_id), create_folder=False)
    comment = str(payload.get("comment") or "").strip()[:5000]
    project["comment"] = comment
    project["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    project = normalize_project_payload(project, create_folder=True)
    front_keys = ["project_id", "project_name", "project_folder", "comment", "rating", "created_at", "saved_at"]
    ordered_project = {key: project[key] for key in front_keys if key in project}
    ordered_project.update({key: value for key, value in project.items() if key not in ordered_project})
    project = ordered_project
    atomic_write_json(project_payload_path(project), project)
    if str(project.get("project_id") or "") == str(project_id):
        try:
            latest = normalize_project_payload(json.loads(PROJECT_STATE_PATH.read_text(encoding="utf-8")), create_folder=False)
            if str(latest.get("project_id") or "") == str(project_id):
                atomic_write_json(PROJECT_STATE_PATH, project)
        except Exception:
            pass
    invalidate_project_list_cache()
    return {
        "message": "Project comment saved.",
        "project_id": project.get("project_id", project_id),
        "project_name": project.get("project_name", project_id),
        "project_folder": project.get("project_folder", ""),
        "comment": comment,
        "saved_at": project.get("saved_at", ""),
    }


def save_project_rating(payload):
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id is required.")
    project = normalize_project_payload(read_project_payload(project_id), create_folder=False)
    try:
        rating = max(0, min(5, int(round(float(payload.get("rating", 0) or 0)))))
    except (TypeError, ValueError):
        rating = 0
    project["rating"] = rating
    state = project.get("state")
    if isinstance(state, dict):
        lens_state = state.setdefault("lens", {})
        if isinstance(lens_state, dict):
            lens_state["rating"] = rating
    project["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    project = normalize_project_payload(project, create_folder=True)
    front_keys = ["project_id", "project_name", "project_folder", "comment", "rating", "created_at", "saved_at"]
    ordered_project = {key: project[key] for key in front_keys if key in project}
    ordered_project.update({key: value for key, value in project.items() if key not in ordered_project})
    project = ordered_project
    atomic_write_json(project_payload_path(project), project)
    try:
        latest = normalize_project_payload(json.loads(PROJECT_STATE_PATH.read_text(encoding="utf-8")), create_folder=False)
        if str(latest.get("project_id") or "") == str(project_id):
            atomic_write_json(PROJECT_STATE_PATH, project)
    except Exception:
        pass
    invalidate_project_list_cache()
    return {
        "message": "Project rating saved.",
        "project_id": project.get("project_id", project_id),
        "project_name": project.get("project_name", project_id),
        "project_folder": project.get("project_folder", ""),
        "rating": rating,
        "saved_at": project.get("saved_at", ""),
    }


def existing_path(value):
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = resolve_workspace_path(path)
    return path.resolve() if path.exists() else None


def first_existing_path(*values):
    for value in values:
        path = existing_path(value)
        if path is not None:
            return path
    return None


def copy_project_artifact(source_path, destination_path):
    if source_path is None:
        return None
    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path == destination_path:
        return str(destination_path)
    shutil.copy2(source_path, destination_path)
    return str(destination_path)


def project_original_path(project_dir):
    return Path(project_dir) / ORIGINAL_FILE_NAME


def project_data_cutout_path(project_dir):
    return Path(project_dir) / DATA_CUTOUT_NAME


def resolved_existing_or_workspace_path(value):
    path = existing_path(value)
    if path is not None:
        return path
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = resolve_workspace_path(path)
    return path.resolve()


def image_shape_pixels(path):
    try:
        shape = image_preview_metadata(path).get("image_shape")
        if not shape or len(shape) < 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return None
        return height * width
    except Exception:
        return None


def project_original_source_candidates(project_dir, payload=None):
    project_dir = Path(project_dir)
    payloads = []
    if isinstance(payload, dict):
        payloads.append(payload)
    for candidate in (project_dir / "project.json", PROJECT_STATE_PATH):
        try:
            if candidate.exists():
                payloads.append(json.loads(candidate.read_text(encoding="utf-8")))
        except Exception:
            pass

    seen = set()

    def add_candidate(value):
        path = existing_path(value)
        if path is None:
            return None
        if path.name in {ORIGINAL_FILE_NAME, DATA_CUTOUT_NAME}:
            return None
        key = str(path)
        if key in seen:
            return None
        seen.add(key)
        return path

    for project_payload in payloads:
        state = project_payload.get("state", {}) if isinstance(project_payload.get("state"), dict) else {}
        image_state = state.get("image", {}) if isinstance(state.get("image"), dict) else {}
        euclid_state = state.get("euclid", {}) if isinstance(state.get("euclid"), dict) else {}
        for value in (
            image_state.get("original_source_path"),
            euclid_state.get("image_path"),
            (euclid_state.get("image") or {}).get("path") if isinstance(euclid_state.get("image"), dict) else None,
            image_state.get("source_path"),
            image_state.get("input_path"),
        ):
            candidate = add_candidate(value)
            if candidate is not None:
                yield candidate

        image_options = euclid_state.get("image_options", {})
        if isinstance(image_options, dict):
            for option in image_options.values():
                image = option.get("image") if isinstance(option, dict) else None
                if isinstance(image, dict):
                    candidate = add_candidate(image.get("path"))
                    if candidate is not None:
                        yield candidate


def larger_recorded_original_source(project_dir, reference_path=None, payload=None):
    reference_pixels = image_shape_pixels(reference_path) if reference_path else None
    for candidate in project_original_source_candidates(project_dir, payload):
        candidate_pixels = image_shape_pixels(candidate)
        if candidate_pixels is None:
            continue
        if reference_pixels is None or candidate_pixels > reference_pixels:
            return candidate
    return None


def project_preprocess_original_source(source_path, project_dir, payload=None, prefer_project_original=True):
    source_path = resolved_existing_or_workspace_path(source_path)
    if not prefer_project_original or project_dir is None:
        return source_path, {}
    project_dir = Path(project_dir)
    original_path = existing_path(project_original_path(project_dir))
    data_cutout_path = existing_path(project_data_cutout_path(project_dir))
    source_is_data_cutout = (
        source_path.name == DATA_CUTOUT_NAME
        or (data_cutout_path is not None and source_path == data_cutout_path)
    )
    source_is_original = (
        source_path.name == ORIGINAL_FILE_NAME
        or (original_path is not None and source_path == original_path)
    )
    if not source_is_data_cutout and not source_is_original:
        return source_path, {}

    reference_path = original_path or data_cutout_path or source_path
    recovered_source = larger_recorded_original_source(project_dir, reference_path, payload)
    if recovered_source is not None:
        return recovered_source, {
            "requested_source_path": str(source_path),
            "resolved_project_original_source": str(recovered_source),
        }
    if original_path is not None:
        return original_path, {"requested_source_path": str(source_path)}
    return source_path, {}


def register_project_original_file(source_path, payload=None, make_default_cutout=True):
    import numpy as np
    from astropy.io import fits

    project_dir = project_folder_from_payload(payload or {})
    if project_dir is not None:
        source_path, source_resolution = project_preprocess_original_source(
            source_path,
            project_dir,
            payload,
            prefer_project_original=True,
        )
    else:
        source_path = resolved_existing_or_workspace_path(source_path)
        source_resolution = {}
    source_path = existing_path(source_path)
    if source_path is None:
        raise FileNotFoundError("Original image file not found.")
    if project_dir is None:
        return {}
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    data, header, _ = load_displayable_image(source_path)
    data = np.asarray(data)
    header = header.copy() if header is not None else fits.Header()
    header["ORIGSRC"] = (str(source_path)[:68], "Source copied by HerculensGUI")
    original_path = project_original_path(project_dir)
    fits.writeto(original_path, data, header=header, overwrite=True, output_verify="silentfix")

    result = {
        "original_path": str(original_path),
        "original_source_path": str(source_path),
        "shape": list(data.shape),
        **source_resolution,
    }

    if make_default_cutout:
        data_path = project_data_cutout_path(project_dir)
        cutout_header = header.copy()
        height, width = data.shape[:2]
        cutout_header["X0"] = 0
        cutout_header["Y0"] = 0
        cutout_header["WIDTH"] = int(width)
        cutout_header["HEIGHT"] = int(height)
        cutout_header["DEFAULT"] = (1, "Default cutout copied from Original_file")
        fits.writeto(data_path, data, header=cutout_header, overwrite=True, output_verify="silentfix")
        bounds = {"x0": 0, "x1": int(width), "y0": 0, "y1": int(height), "width": int(width), "height": int(height)}
        write_zero_project_masks(project_dir, data.shape[:2], source_path=data_path, bounds=bounds)
        try:
            preview_path, _ = save_preview_png(data_path, {"project_folder": str(project_dir)})
            result["data_cutout_preview_url"] = relative_url(preview_path)
        except Exception:
            pass
        result.update(
            {
                "data_cutout_path": str(data_path),
                "data_cutout_bounds": bounds,
                "data_cutout_shape": list(data.shape),
            }
        )
    return result


def write_zero_project_masks(project_dir, data_shape, source_path=None, bounds=None, overwrite=False):
    import numpy as np
    from astropy.io import fits

    project_dir = Path(project_dir)
    height, width = int(data_shape[0]), int(data_shape[1])
    for mask_type in ("mask_1", "mask_2", "mask_out"):
        mask_path = project_dir / f"{mask_type}.fits"
        if mask_path.exists() and not overwrite:
            try:
                existing = np.asarray(fits.getdata(mask_path))
                if tuple(existing.shape[:2]) == (height, width):
                    continue
            except Exception:
                pass
        fits.writeto(mask_path, np.zeros((height, width), dtype=np.uint8), overwrite=True)


def write_project_cutout_artifacts(project_dir, source_path, cutout, header, bounds):
    import numpy as np
    from astropy.io import fits

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    data_path = project_data_cutout_path(project_dir)
    fits.writeto(data_path, cutout, header=header, overwrite=True)

    rms_path = project_dir / "RMS_map.fits"
    if rms_path.exists():
        try:
            rms_data, rms_header = fits.getdata(rms_path, header=True)
            rms_data = np.asarray(rms_data)
            if rms_data.shape[:2] == tuple(bounds["source_shape"]):
                rms_cutout = rms_data[bounds["y0"] : bounds["y1"], bounds["x0"] : bounds["x1"]]
                fits.writeto(rms_path, np.asarray(rms_cutout, dtype=np.float32), header=rms_header, overwrite=True)
            elif rms_data.shape[:2] != np.asarray(cutout).shape[:2]:
                corner = np.asarray(cutout, dtype=float)[: min(10, cutout.shape[0]), : min(10, cutout.shape[1])]
                rms = max(float(np.nanstd(corner)), 1e-12)
                hdr = fits.Header()
                hdr["EXTNAME"] = "RMS"
                hdr["RMSMODE"] = ("CONST", "Fallback RMS after project cutout resize")
                fits.writeto(rms_path, np.full(np.asarray(cutout).shape[:2], rms, dtype=np.float32), header=hdr, overwrite=True)
        except Exception:
            corner = np.asarray(cutout, dtype=float)[: min(10, cutout.shape[0]), : min(10, cutout.shape[1])]
            rms = max(float(np.nanstd(corner)), 1e-12)
            hdr = fits.Header()
            hdr["EXTNAME"] = "RMS"
            hdr["RMSMODE"] = ("CONST", "Fallback RMS after project cutout resize")
            fits.writeto(rms_path, np.full(np.asarray(cutout).shape[:2], rms, dtype=np.float32), header=hdr, overwrite=True)

    write_zero_project_masks(project_dir, np.asarray(cutout).shape[:2], source_path=source_path, bounds=bounds)
    return data_path, rms_path if rms_path.exists() else None


def sync_project_runtime_files(project_dir):
    project_dir = Path(project_dir)
    source = GUI_ROOT / "Tian_infra.py"
    if source.exists():
        shutil.copy2(source, project_dir / "Tian_infra.py")
    semilinear_source = GUI_ROOT / "semilinear_solver.py"
    if semilinear_source.exists():
        shutil.copy2(
            semilinear_source,
            project_dir / "semilinear_solver.py",
        )
    solver_source = GUI_ROOT / "theta_e_solver.py"
    if solver_source.exists():
        shutil.copy2(solver_source, project_dir / "theta_e_solver.py")
    lens_light_runner_source = WEB_ROOT / LENS_LIGHT_RUNNER_NAME
    if lens_light_runner_source.exists():
        shutil.copy2(lens_light_runner_source, project_dir / LENS_LIGHT_RUNNER_NAME)
    custom_gibbs_source = GUI_ROOT.parent / "Herculens-Extensions" / "custom_gibbs.py"
    if custom_gibbs_source.exists():
        shutil.copy2(custom_gibbs_source, project_dir / "custom_gibbs.py")


def project_data_artifact(project_dir, payload=None):
    payload = payload or {}
    manifest_image = None
    manifest_path = project_dir / "project_artifacts.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_image = manifest.get("files", {}).get("image")
        except Exception:
            manifest_image = None
    return first_existing_path(
        project_data_cutout_path(project_dir),
        manifest_image,
        payload.get("source_path"),
    )


def write_standard_psf_artifact(source_path, destination_path):
    if source_path is None:
        return None
    import numpy as np
    from astropy.io import fits

    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".npy":
        kernel = np.load(source_path)
    else:
        with fits.open(source_path) as hdul:
            if "DET_PSF_MODEL" in hdul:
                kernel = hdul["DET_PSF_MODEL"].data
            elif hdul[0].data is not None:
                kernel = hdul[0].data
            else:
                kernel = hdul[1].data
    fits.writeto(destination_path, np.asarray(kernel, dtype=np.float32), overwrite=True)
    return str(destination_path)


def write_standard_rms_artifact(source_path, destination_path):
    if source_path is None:
        return None
    import numpy as np
    from astropy.io import fits

    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path == destination_path:
        return str(destination_path)
    with fits.open(source_path, memmap=False) as hdul:
        if "ERRORMAP" in hdul:
            rms = hdul["ERRORMAP"].data
            header = hdul["ERRORMAP"].header
        elif hdul[0].data is not None:
            rms = hdul[0].data
            header = hdul[0].header
        else:
            rms = hdul[1].data
            header = hdul[1].header
        out_header = header.copy()
        out_header["EXTNAME"] = "RMS"
    fits.writeto(destination_path, np.asarray(rms, dtype=np.float32), header=out_header, overwrite=True)
    return str(destination_path)


def write_project_rms_artifact(source_path, destination_path, payload=None):
    if source_path is None:
        return None
    import numpy as np
    from astropy.io import fits

    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = _cutout_bounds_from_payload(payload or {})
    target_shape = _target_cutout_shape_from_payload(payload or {})

    with fits.open(source_path, memmap=False) as hdul:
        if "ERRORMAP" in hdul:
            rms = hdul["ERRORMAP"].data
            header = hdul["ERRORMAP"].header
        elif hdul[0].data is not None:
            rms = hdul[0].data
            header = hdul[0].header
        else:
            rms = hdul[1].data
            header = hdul[1].header
        rms = np.asarray(rms, dtype=np.float32)
        out_header = header.copy()

    if bounds:
        source_shape = bounds.get("source_shape")
        crop_shape = (bounds["y1"] - bounds["y0"], bounds["x1"] - bounds["x0"])
        can_crop = (
            0 <= bounds["y0"] < bounds["y1"] <= rms.shape[0]
            and 0 <= bounds["x0"] < bounds["x1"] <= rms.shape[1]
            and (source_shape is None or tuple(rms.shape[:2]) == tuple(source_shape))
            and (target_shape is None or tuple(crop_shape) == tuple(target_shape))
        )
        if can_crop and (target_shape is None or tuple(rms.shape[:2]) != tuple(target_shape)):
            rms = rms[bounds["y0"] : bounds["y1"], bounds["x0"] : bounds["x1"]]
            if "CRPIX1" in out_header:
                out_header["CRPIX1"] = float(out_header["CRPIX1"]) - bounds["x0"]
            if "CRPIX2" in out_header:
                out_header["CRPIX2"] = float(out_header["CRPIX2"]) - bounds["y0"]
            out_header["RMSCROP"] = (True, "Cropped by HerculensGUI image preprocess")

    if target_shape and tuple(rms.shape[:2]) != tuple(target_shape):
        raise ValueError(
            f"RMS shape {tuple(rms.shape[:2])} does not match current cutout {tuple(target_shape)}. "
            "Choose a cutout-sized RMS map, or choose the full image RMS before saving the cutout."
        )

    out_header["EXTNAME"] = "RMS"
    fits.writeto(destination_path, rms, header=out_header, overwrite=True)
    return str(destination_path)


def project_payload_for_request(payload):
    project_folder_value = str(payload.get("project_folder") or "").strip()
    if project_folder_value:
        candidate = project_payload_path(payload)
        if candidate.exists():
            return normalize_project_payload(json.loads(candidate.read_text(encoding="utf-8")), create_folder=True)
        return normalize_project_payload(payload, create_folder=True)
    project_id = str(payload.get("project_id") or "").strip()
    if project_id:
        try:
            return normalize_project_payload(read_project_payload(project_id), create_folder=True)
        except Exception:
            pass
    return project_payload_or_default()


def standardize_project_artifacts(payload):
    payload = dict(payload or {})
    project = project_payload_for_request(payload)
    state = project.get("state", {}) if isinstance(project.get("state"), dict) else {}
    mask_state = state.get("mask", {}) if isinstance(state.get("mask"), dict) else {}

    def saved_conjugate_points(key, legacy_key=None):
        points = []
        raw_points = mask_state.get(key)
        if raw_points is None and legacy_key:
            raw_points = mask_state.get(legacy_key)
        for point in raw_points or []:
            arcsec = point.get("arcsec", {}) if isinstance(point, dict) else {}
            if "x" in arcsec and "y" in arcsec:
                points.append([arcsec["x"], arcsec["y"]])
        return points

    if not payload.get("conjugate_points_source1"):
        conjugate_points = saved_conjugate_points("conjugate_points_source1", legacy_key="conjugate_points")
        if conjugate_points:
            payload["conjugate_points_source1"] = conjugate_points
    if not payload.get("conjugate_points_source2"):
        conjugate_points = saved_conjugate_points("conjugate_points_source2")
        if conjugate_points:
            payload["conjugate_points_source2"] = conjugate_points

    project_dir = Path(project["project_folder"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    sync_project_runtime_files(project_dir)

    image_state = state.get("image", {}) if isinstance(state.get("image"), dict) else {}
    cutout_artifact = image_state.get("cutout_artifact", {}) if isinstance(image_state.get("cutout_artifact"), dict) else {}
    euclid_state = state.get("euclid", {}) if isinstance(state.get("euclid"), dict) else {}
    data_source = first_existing_path(
        cutout_artifact.get("fits_path"),
        project_data_cutout_path(project_dir),
        image_state.get("source_path"),
        image_state.get("input_path"),
        euclid_state.get("image_path"),
        project_original_path(project_dir),
    )
    rms_source = first_existing_path(
        payload.get("rms_map_path"),
        euclid_state.get("rms_map_path"),
        euclid_state.get("rms_bundle_path"),
        project_dir / "RMS_map.fits",
    )

    psf_state = state.get("psf", {}) if isinstance(state.get("psf"), dict) else {}
    psf_status = psf_state.get("job_status", {}) if isinstance(psf_state.get("job_status"), dict) else {}
    psf_source = first_existing_path(
        psf_status.get("fits_path"),
        psf_state.get("input_path"),
        payload.get("psf_path"),
    )

    manifest = {
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name"),
        "project_folder": str(project_dir),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {},
        "sources": {},
    }

    data_dest = project_data_cutout_path(project_dir)
    copied = copy_project_artifact(data_source, data_dest)
    if copied:
        manifest["files"]["image"] = copied
        manifest["sources"]["image"] = str(data_source)
    data_shape = None
    if data_dest.exists():
        try:
            from astropy.io import fits

            data_shape = fits.getdata(data_dest).shape
        except Exception:
            data_shape = None

    rms_dest = project_dir / "RMS_map.fits"
    copied = write_standard_rms_artifact(rms_source, rms_dest)
    if copied:
        manifest["files"]["rms"] = copied
        manifest["sources"]["rms"] = str(rms_source)
    if data_shape is not None and rms_dest.exists():
        try:
            import numpy as np
            from astropy.io import fits

            rms_shape = fits.getdata(rms_dest).shape
            if tuple(rms_shape[:2]) != tuple(data_shape[:2]):
                data_for_rms = np.asarray(fits.getdata(data_dest), dtype=float)
                corner = data_for_rms[: min(10, data_for_rms.shape[0]), : min(10, data_for_rms.shape[1])]
                rms = max(float(np.nanstd(corner)), 1e-12)
                hdr = fits.Header()
                hdr["EXTNAME"] = "RMS"
                hdr["RMSMODE"] = ("CONST", "Fallback RMS after project cutout resize")
                fits.writeto(rms_dest, np.full(data_shape[:2], rms, dtype=np.float32), header=hdr, overwrite=True)
                manifest["files"]["rms"] = str(rms_dest)
                manifest["sources"]["rms"] = "generated_constant_for_current_cutout"
        except Exception:
            pass

    mask_state = state.get("mask", {}) if isinstance(state.get("mask"), dict) else {}
    mask_entries = []
    if mask_state.get("saved_path"):
        mask_entries.append((str(mask_state.get("type") or "mask_1"), mask_state.get("saved_path")))
    masks_state = state.get("masks", {}) if isinstance(state.get("masks"), dict) else {}
    for mask_type in ("mask_1", "mask_2", "mask_out"):
        mask_info = masks_state.get(mask_type, {}) if isinstance(masks_state.get(mask_type), dict) else {}
        if mask_info.get("saved_path"):
            mask_entries.append((mask_type, mask_info.get("saved_path")))

    for mask_type, source in mask_entries:
        if mask_type not in {"mask_1", "mask_2", "mask_out"}:
            continue
        mask_source = existing_path(source)
        copied = copy_project_artifact(mask_source, project_dir / f"{mask_type}.fits")
        if copied:
            manifest["files"][mask_type] = copied
            manifest["sources"][mask_type] = str(mask_source)

    if data_shape is not None:
        import numpy as np
        from astropy.io import fits

        for mask_type, fill_value in (("mask_1", 0), ("mask_2", 0), ("mask_out", 0)):
            mask_path = project_dir / f"{mask_type}.fits"
            reset_mask = not mask_path.exists()
            if mask_path.exists():
                try:
                    reset_mask = tuple(fits.getdata(mask_path).shape[:2]) != tuple(data_shape[:2])
                except Exception:
                    reset_mask = True
            if reset_mask:
                fits.writeto(mask_path, np.full(data_shape[:2], fill_value, dtype=np.uint8), overwrite=True)
            manifest["files"].setdefault(mask_type, str(mask_path))
            manifest["sources"].setdefault(mask_type, "generated_default")

    psf_dest = project_dir / "PSF_model.fits"
    copied = write_standard_psf_artifact(psf_source, psf_dest)
    if copied:
        manifest["files"]["psf"] = copied
        manifest["sources"]["psf"] = str(psf_source)

    lens_light_dir = project_dir / LENS_LIGHT_RESULT_DIRNAME
    for artifact_key, filename in (
        ("lens_light_kwargs", "kwargs_lens_light.pkl"),
        ("lens_light_kwargs_constrained", "kwargs_lens_light_constrained.pkl"),
        ("lens_light_subtracted", "lens_light_subtracted.fits"),
        ("lens_light_subtracted_constrained", "lens_light_subtracted_constrained.fits"),
    ):
        artifact_path = lens_light_dir / filename
        if artifact_path.exists():
            manifest["files"][artifact_key] = str(artifact_path)
            manifest["sources"][artifact_key] = "step_4_lens_light_subtraction"

    atomic_write_json(project_dir / "project_artifacts.json", manifest)
    payload["project_id"] = project.get("project_id")
    payload["project_name"] = project.get("project_name")
    payload["project_folder"] = str(project_dir)
    payload["data_folder"] = str(project_dir)
    payload["project_artifacts"] = manifest
    return payload, manifest


def project_preview_for_payload(payload):
    project = normalize_project_payload(payload)
    project_dir = Path(project["project_folder"]).expanduser()
    state = project.get("state", {}) if isinstance(project.get("state"), dict) else {}
    image_state = state.get("image", {}) if isinstance(state.get("image"), dict) else {}
    euclid_state = state.get("euclid", {}) if isinstance(state.get("euclid"), dict) else {}
    cutout_artifact = image_state.get("cutout_artifact", {}) if isinstance(image_state.get("cutout_artifact"), dict) else {}
    image_payload = euclid_state.get("image", {}) if isinstance(euclid_state.get("image"), dict) else {}
    preview_candidates = [
        project_dir / "previews" / "Data_cutout_preview.png",
        cutout_artifact.get("preview_url"),
        image_state.get("preview_src"),
    ]
    preview_candidates.extend(sorted(project_dir.glob("previews/*_preview.png"), key=lambda path: path.stat().st_mtime, reverse=True))
    preview_candidates.extend(sorted(project_dir.glob("cutouts/*.png"), key=lambda path: path.stat().st_mtime, reverse=True))
    preview_path = first_existing_path(*preview_candidates)
    if preview_path is not None:
        try:
            return relative_url(preview_path), preview_path.name
        except ValueError:
            pass

    preview_source = first_existing_path(
        project_dir / "Data_cutout.fits",
        cutout_artifact.get("fits_path"),
        image_payload.get("path"),
        euclid_state.get("image_path"),
        image_state.get("source_path"),
    )
    if preview_source is None:
        return "", ""
    try:
        preview_url, _ = preview_existing_path(preview_source, {"project_folder": str(project_dir)})
        return preview_url, Path(preview_source).name
    except Exception:
        return "", Path(preview_source).name


def project_preview_for_listing(project_dir):
    project_dir = Path(project_dir).expanduser()
    preview_candidates = [
        project_dir / "previews" / "Data_cutout_preview.png",
        project_dir / "previews" / f"{project_dir.name}_preview.png",
    ]
    preview_path = first_existing_path(*preview_candidates)
    if preview_path is not None:
        try:
            return relative_url_with_mtime(preview_path), preview_path.name
        except ValueError:
            return "", preview_path.name

    preview_candidates = []
    preview_candidates.extend(sorted(project_dir.glob("previews/*_preview.png"), key=lambda path: path.stat().st_mtime, reverse=True))
    preview_candidates.extend(sorted(project_dir.glob("cutouts/*.png"), key=lambda path: path.stat().st_mtime, reverse=True))
    preview_path = first_existing_path(*preview_candidates)
    if preview_path is not None:
        try:
            return relative_url_with_mtime(preview_path), preview_path.name
        except ValueError:
            return "", preview_path.name

    data_path = project_dir / "Data_cutout.fits"
    if not data_path.exists():
        return "", ""
    try:
        preview_url, _ = preview_existing_path(data_path, {"project_folder": str(project_dir)})
        generated_preview = first_existing_path(project_dir / "previews" / "Data_cutout_preview.png")
        if generated_preview is not None:
            return relative_url_with_mtime(generated_preview), data_path.name
        return preview_url, data_path.name
    except Exception:
        return "", data_path.name


def project_svi_listing_status(project_dir):
    project_dir = Path(project_dir).expanduser()
    result_dir = project_dir / LENS_MODEL_RESULT_DIRNAME
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            chain_previews = summary.get("chain_previews")
            if isinstance(chain_previews, list) and chain_previews:
                return {
                    "has_svi_result": True,
                    "svi_badge": "SVI",
                    "svi_label": f"{len(chain_previews)} SVI chain(s)",
                }
        except Exception:
            pass
    comparison_paths = list(result_dir.glob("chain_*_lens_model_comparison.png"))
    comparison_paths.extend(result_dir.glob("*four_panel_comparison.png"))
    if comparison_paths:
        return {
            "has_svi_result": True,
            "svi_badge": "SVI",
            "svi_label": "SVI result available",
        }
    return {"has_svi_result": False, "svi_badge": "", "svi_label": ""}


def project_json_has_generated_code(project_json_path):
    try:
        text = Path(project_json_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    match = re.search(r'"generated_code"\s*:\s*"', text)
    if not match:
        return False
    value_start = match.end()
    return value_start < len(text) and text[value_start] != '"'


def project_generated_script_status(project_dir, project_json_path=None):
    project_dir = Path(project_dir).expanduser()
    script_path = project_dir / LENS_MODEL_SCRIPT_NAME
    has_saved_generated_code = project_json_has_generated_code(project_json_path or (project_dir / "project.json"))
    if script_path.exists() and has_saved_generated_code:
        return {
            "has_generated_script": True,
            "script_badge": "PY",
            "script_label": f"Generated lens model script: {script_path.name}",
            "script_path": str(script_path),
        }
    return {"has_generated_script": False, "script_badge": "", "script_label": "", "script_path": ""}


def project_listing_metadata(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(32768)

    def text_field(key):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*("(?:\\.|[^"\\])*")', text)
        if not match:
            return ""
        try:
            return json.loads(match.group(1))
        except Exception:
            return match.group(1).strip('"')

    def number_field(key, default=0.0):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if not match:
            return default
        try:
            return float(match.group(1))
        except Exception:
            return default

    project_id = text_field("project_id") or path.parent.name
    project_folder = text_field("project_folder") or str(path.parent.resolve())
    project_name = canonical_project_name(text_field("project_name"), project_id, project_folder)
    rating = max(0.0, min(5.0, number_field("rating", 0.0)))
    return {
        "project_id": project_id,
        "project_name": project_name or project_id,
        "project_folder": project_folder,
        "comment": text_field("comment"),
        "rating": rating,
        "saved_at": text_field("saved_at") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "created_at": text_field("created_at"),
        "path": str(path),
    }


def project_list_signature(paths):
    signature = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(path), None, None))
    return tuple(signature)


def overleaf_catalog_coordinates():
    try:
        text = OVERLEAF_PAPER_PATH.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ()

    coordinates = []
    seen = set()
    for label in OVERLEAF_CATALOG_TABLE_LABELS:
        label_marker = rf"\label{{{label}}}"
        label_index = text.find(label_marker)
        if label_index < 0:
            continue
        table_end_candidates = [
            index
            for marker in (r"\end{table}", r"\end{table*}")
            if (index := text.find(marker, label_index)) >= 0
        ]
        table_end = min(table_end_candidates) if table_end_candidates else len(text)
        table_text = text[label_index:table_end].replace("$", "")
        for line in table_text.splitlines():
            match = re.match(
                r"^\s*\d+\s*&\s*EUCL-DR1.*?&\s*"
                r"(-?\d+(?:\.\d+)?)\s*&\s*(-?\d+(?:\.\d+)?)\s*(?:&|\\\\)",
                line,
            )
            if not match:
                continue
            coordinate = (float(match.group(1)), float(match.group(2)))
            rounded_coordinate = (round(coordinate[0], 7), round(coordinate[1], 7))
            if rounded_coordinate in seen:
                continue
            seen.add(rounded_coordinate)
            coordinates.append(coordinate)
    return tuple(coordinates)


def project_id_coordinates(project_id):
    project_id = str(project_id or "").strip()
    canonical_match = re.search(
        r"DSPL_RA(\d+(?:_\d+)?)DEC(NEG)?(\d+(?:_\d+)?)",
        project_id,
        flags=re.IGNORECASE,
    )
    if canonical_match:
        ra = float(canonical_match.group(1).replace("_", "."))
        dec = float(canonical_match.group(3).replace("_", "."))
        if canonical_match.group(2):
            dec = -dec
        return ra, dec

    if re.search(r"_(?:VIS|Y|J|H)$", project_id, flags=re.IGNORECASE):
        return None
    alias_match = re.search(
        r"(?:^|_)RA(\d+)p(\d+)_DEC([mp])(\d+)p(\d+)(?:$|_)",
        project_id,
        flags=re.IGNORECASE,
    )
    if not alias_match:
        return None
    ra = float(f"{alias_match.group(1)}.{alias_match.group(2)}")
    dec = float(f"{alias_match.group(4)}.{alias_match.group(5)}")
    if alias_match.group(3).lower() == "m":
        dec = -dec
    return ra, dec


def overleaf_catalog_match(project_id, catalog_coordinates):
    project_coordinates = project_id_coordinates(project_id)
    if project_coordinates is None:
        return None
    project_ra, project_dec = project_coordinates
    best_match = None
    for catalog_ra, catalog_dec in catalog_coordinates:
        mean_dec = math.radians((project_dec + catalog_dec) / 2.0)
        delta_ra = (project_ra - catalog_ra) * math.cos(mean_dec)
        delta_dec = project_dec - catalog_dec
        distance_arcsec = math.hypot(delta_ra, delta_dec) * 3600.0
        if best_match is None or distance_arcsec < best_match["distance_arcsec"]:
            best_match = {
                "ra": catalog_ra,
                "dec": catalog_dec,
                "distance_arcsec": distance_arcsec,
            }
    if (
        best_match is None
        or best_match["distance_arcsec"] > OVERLEAF_CATALOG_MATCH_TOLERANCE_ARCSEC
    ):
        return None
    return best_match


def recycle_destination(path):
    RECYCLE_BIN_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    base = RECYCLE_BIN_DIR / f"HerculensGUI_{path.name}"
    if not base.exists():
        return base
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = RECYCLE_BIN_DIR / f"HerculensGUI_{path.name}_{timestamp}"
    index = 1
    while candidate.exists():
        candidate = RECYCLE_BIN_DIR / f"HerculensGUI_{path.name}_{timestamp}_{index}"
        index += 1
    return candidate


def delete_project_payload(payload):
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id is required.")
    project = normalize_project_payload(read_project_payload(project_id), create_folder=False)
    project_dir = Path(project["project_folder"]).expanduser().resolve()
    try:
        project_dir.relative_to(PROJECT_FOLDER_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to delete project outside runs/: {project_dir}") from exc
    if project_dir == PROJECT_FOLDER_DIR.resolve():
        raise ValueError("Refusing to delete the runs directory.")
    if not (project_dir / "project.json").exists():
        raise FileNotFoundError(f"Project folder does not contain project.json: {project_dir}")

    destination = recycle_destination(project_dir)
    shutil.move(str(project_dir), str(destination))

    next_project = latest_restorable_project_payload()
    if next_project is not None:
        clean_next = dict(next_project)
        clean_next.pop("_project_exists", None)
        atomic_write_json(PROJECT_STATE_PATH, clean_next)

    return {
        "message": f"Moved project to RecycleBin: {destination.name}",
        "project_id": project_id,
        "project_name": project.get("project_name", project_id),
        "recycle_path": str(destination),
        "next_project": next_project,
    }


def project_cutout_overlay_payload(payload):
    import numpy as np
    from astropy.io import fits

    project = project_payload_for_request(payload)
    project_dir = Path(project["project_folder"]).expanduser()
    source_path = project_data_artifact(project_dir, payload)
    if source_path is None:
        raise FileNotFoundError("Project has no Data_cutout.fits or image source.")

    image = fits_payload_for_gui(source_path)
    height, width = image["shape"]
    preview_path = project_dir / "previews" / "Data_cutout_preview.png"
    preview_url = relative_url_with_mtime(preview_path) if preview_path.exists() else ""
    masks = {}
    for mask_type in ("mask_1", "mask_2", "mask_out"):
        mask_path = project_dir / f"{mask_type}.fits"
        if mask_path.exists():
            data = np.asarray(fits.getdata(mask_path))
            mask = np.flipud(np.where(data > 0, 1, 0).astype(np.uint8))
            if mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        masks[mask_type] = mask.tolist()

    state = project.get("state", {}) if isinstance(project.get("state"), dict) else {}
    mask_state = state.get("mask", {}) if isinstance(state.get("mask"), dict) else {}
    display_mtf = project_display_mtf(project, payload)
    subtracted_display_mtf = project_lens_light_display_mtf(project, payload)
    display_range = project_display_range(project, payload) or finite_image_display_range(source_path)
    active_mask_type = str(payload.get("mask_type") or mask_state.get("type") or "mask_1")
    if active_mask_type not in {"mask_1", "mask_2", "mask_out"}:
        active_mask_type = "mask_1"
    active_mask_rows = payload.get("mask")
    if active_mask_rows is None:
        active_mask_rows = masks.get(active_mask_type)
    if "conjugate_points_source1" in payload:
        points1 = payload.get("conjugate_points_source1") or []
    else:
        points1 = mask_state.get("conjugate_points_source1") or mask_state.get("conjugate_points") or []
    if "conjugate_points_source2" in payload:
        points2 = payload.get("conjugate_points_source2") or []
    else:
        points2 = mask_state.get("conjugate_points_source2") or []
    matplotlib_preview_url = ""
    subtracted_matplotlib_preview_url = ""
    try:
        matplotlib_preview_path = save_mask_matplotlib_preview(
            project_dir,
            source_path,
            active_mask_rows,
            active_mask_type,
            points1,
            points2,
            display_mtf,
            "image",
            display_range,
        )
        matplotlib_preview_url = relative_url_with_mtime(matplotlib_preview_path)
    except Exception as exc:
        print(f"Lens model cutout matplotlib preview failed: {exc}", file=sys.stderr, flush=True)
    subtracted_path = project_dir / LENS_LIGHT_RESULT_DIRNAME / "lens_light_subtracted.fits"
    subtracted_image = None
    subtracted_stale_message = ""
    if subtracted_path.exists():
        try:
            subtracted_image = fits_payload_for_gui(subtracted_path)
            if list(subtracted_image.get("shape") or []) != list(image.get("shape") or []):
                subtracted_stale_message = (
                    f"Lens-light subtraction is {subtracted_image.get('shape')}; "
                    f"current cutout is {image.get('shape')}. Run lens-light subtraction again."
                )
                subtracted_image = None
            else:
                try:
                    subtracted_preview_path = save_mask_matplotlib_preview(
                        project_dir,
                        subtracted_path,
                        active_mask_rows,
                        active_mask_type,
                        points1,
                        points2,
                        subtracted_display_mtf,
                        "subtracted",
                        None,
                    )
                    subtracted_matplotlib_preview_url = relative_url_with_mtime(subtracted_preview_path)
                except Exception as exc:
                    print(f"Lens model subtracted matplotlib preview failed: {exc}", file=sys.stderr, flush=True)
        except Exception:
            subtracted_image = None
    return {
        "message": f"Loaded cutout overlay: {Path(source_path).name}",
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "project_folder": str(project_dir),
        "source_path": str(source_path),
        "preview_url": preview_url,
        "matplotlib_preview_url": matplotlib_preview_url,
        "subtracted_matplotlib_preview_url": subtracted_matplotlib_preview_url,
        "image": image,
        "subtracted_image": subtracted_image,
        "subtracted_stale": bool(subtracted_stale_message),
        "subtracted_stale_message": subtracted_stale_message,
        "masks": masks,
        "conjugate_points_source1": points1,
        "conjugate_points_source2": points2,
    }


def conjugate_points_arcsec_array(points):
    values = []
    for point in points or []:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            values.append([float(point[0]), float(point[1])])
            continue
        if isinstance(point, dict):
            arcsec = point.get("arcsec") or point.get("center_arcsec") or {}
            if isinstance(arcsec, dict) and "x" in arcsec and "y" in arcsec:
                values.append([float(arcsec["x"]), float(arcsec["y"])])
    return values


def conjugate_points_extent_array(points, width, height, pix_scale):
    values = []
    for point in points or []:
        if isinstance(point, dict):
            data = point.get("data") or point.get("center_data") or {}
            if isinstance(data, dict) and "x" in data and "y" in data:
                try:
                    x_data = float(data["x"])
                    y_data = float(data["y"])
                    values.append(
                        [
                            (x_data - (width - 1) / 2.0) * pix_scale,
                            (y_data - (height - 1) / 2.0) * pix_scale,
                        ]
                    )
                    continue
                except (TypeError, ValueError):
                    pass
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            values.append([float(point[0]), float(point[1])])
            continue
        if isinstance(point, dict):
            arcsec = point.get("arcsec") or point.get("center_arcsec") or {}
            if isinstance(arcsec, dict) and "x" in arcsec and "y" in arcsec:
                values.append([float(arcsec["x"]), float(arcsec["y"])])
    return values


def normalize_display_mtf(value):
    value = value if isinstance(value, dict) else {}
    return {
        "shadows": float(value.get("shadows", 0.0) or 0.0),
        "midtones": float(value.get("midtones", 0.5) or 0.5),
        "highlights": float(value.get("highlights", 1.0) or 1.0),
    }


def subtracted_display_mtf(value):
    return normalize_display_mtf(value)


def normalize_display_cmap(value):
    value = str(value or "twilight").strip().lower()
    return "gray" if value in {"gray", "grey", "grayscale", "greyscale"} else "twilight"


def normalize_display_range(value):
    import math

    value = value if isinstance(value, dict) else {}
    try:
        vmin = float(value.get("min"))
        vmax = float(value.get("max"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return None
    return {"min": vmin, "max": vmax}


def finite_image_display_range(path):
    import numpy as np

    try:
        data, _ = scalar_image_from_path(path)
    except Exception:
        return None
    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    if not np.any(finite):
        return None
    vmin = float(np.nanmin(data[finite]))
    vmax = float(np.nanmax(data[finite]))
    return normalize_display_range({"min": vmin, "max": vmax})


def mtf_curve_value(midtones, x):
    import numpy as np

    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    result = np.where(x >= 1, 1.0, result)
    active = (x > 0) & (x < 1)
    if abs(float(midtones) - 0.5) < 1e-12:
        result = np.where(active, x, result)
    elif float(midtones) <= 0:
        result = np.where(active, 1.0, result)
    elif float(midtones) >= 1:
        result = np.where(active, 0.0, result)
    else:
        m = float(midtones)
        transformed = ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)
        result = np.where(active, transformed, result)
    return np.clip(result, 0.0, 1.0)


def mtf_stretch_image(data, mtf_values, display_range=None):
    import numpy as np

    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=float)
    display_range = normalize_display_range(display_range)
    if display_range is not None:
        vmin = display_range["min"]
        vmax = display_range["max"]
    else:
        values = data[finite]
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(data, dtype=float)
    mtf = normalize_display_mtf(mtf_values)
    shadows = mtf["shadows"]
    highlights = mtf["highlights"]
    if highlights <= shadows + 0.005:
        highlights = min(1.0, shadows + 0.005)
    width = max(highlights - shadows, 1e-6)
    normalized = (data - vmin) / max(vmax - vmin, 1e-12)
    clipped = np.clip((normalized - shadows) / width, 0.0, 1.0)
    stretched = mtf_curve_value(mtf["midtones"], clipped)
    return np.where(finite, stretched, 0.0)


def project_display_mtf(project, payload=None):
    payload = payload if isinstance(payload, dict) else {}
    if isinstance(payload.get("display_mtf"), dict):
        return payload["display_mtf"]
    state = project.get("state", {}) if isinstance(project.get("state"), dict) else {}
    image_state = state.get("image", {}) if isinstance(state.get("image"), dict) else {}
    if isinstance(image_state.get("mtf"), dict):
        return image_state["mtf"]
    return {"shadows": 0.0, "midtones": 0.125, "highlights": 1.0}


def project_lens_light_display_mtf(project, payload=None):
    payload = payload if isinstance(payload, dict) else {}
    if isinstance(payload.get("subtracted_display_mtf"), dict):
        return payload["subtracted_display_mtf"]
    if isinstance(payload.get("display_mtf"), dict):
        return payload["display_mtf"]
    state = project.get("state", {}) if isinstance(project.get("state"), dict) else {}
    mask_state = state.get("mask", {}) if isinstance(state.get("mask"), dict) else {}
    if isinstance(mask_state.get("lens_light_mtf"), dict):
        return mask_state["lens_light_mtf"]
    if isinstance(mask_state.get("mask_mtf"), dict):
        return mask_state["mask_mtf"]
    return {"shadows": 0.0, "midtones": 0.125, "highlights": 1.0}


def project_display_range(project, payload=None):
    payload = payload if isinstance(payload, dict) else {}
    display_range = normalize_display_range(payload.get("display_range"))
    if display_range is not None:
        return display_range
    return None


def save_lens_script_cutout_overlay_preview(project_dir, source_path, mask_state, display_mtf=None, display_range=None):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

    data, header = scalar_image_from_path(source_path)
    data = np.asarray(data, dtype=float)
    height, width = data.shape
    pix_scale = pixel_scale_arcsec(header, 1.0) or 1.0
    extent = [
        -0.5 * width * pix_scale,
        0.5 * width * pix_scale,
        -0.5 * height * pix_scale,
        0.5 * height * pix_scale,
    ]

    def read_mask(name):
        path = Path(project_dir) / f"{name}.fits"
        if not path.exists():
            return None
        mask = np.asarray(fits.getdata(path), dtype=float)
        return mask if mask.shape == data.shape else None

    def plot_mask_contour(ax, mask, color):
        if mask is None or not np.any(mask > 0):
            return
        ax.contour(
            mask,
            levels=[0.5],
            colors=color,
            alpha=0.95,
            linestyles="dashed",
            origin="lower",
            extent=extent,
        )

    points1 = np.asarray(
        conjugate_points_arcsec_array(mask_state.get("conjugate_points_source1") or mask_state.get("conjugate_points") or []),
        dtype=float,
    ).reshape(-1, 2)
    points2 = np.asarray(
        conjugate_points_arcsec_array(mask_state.get("conjugate_points_source2") or []),
        dtype=float,
    ).reshape(-1, 2)

    preview_dir = Path(project_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output = preview_dir / "lens_script_cutout_overlay.png"

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(
        mtf_stretch_image(data, display_mtf, display_range),
        extent=extent,
        cmap="twilight",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plot_mask_contour(ax, read_mask("mask_1"), "#9933ff")
    plot_mask_contour(ax, read_mask("mask_2"), "#ff3300")
    if points1.size:
        ax.plot(points1[:, 0], points1[:, 1], "o", color="black", markersize=1)
    if points2.size:
        ax.plot(points2[:, 0], points2[:, 1], "o", color="C1", markersize=1)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_lens_script_subtracted_preview(project_dir, source_path, display_mtf=None, display_range=None):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    data, header = scalar_image_from_path(source_path)
    data = np.asarray(data, dtype=float)
    height, width = data.shape
    pix_scale = pixel_scale_arcsec(header, 1.0) or 1.0
    extent = [
        -0.5 * width * pix_scale,
        0.5 * width * pix_scale,
        -0.5 * height * pix_scale,
        0.5 * height * pix_scale,
    ]
    preview_dir = Path(project_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output = preview_dir / "lens_script_subtracted_preview.png"

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(
        mtf_stretch_image(data, subtracted_display_mtf(display_mtf), None),
        extent=extent,
        cmap="twilight",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_mask_matplotlib_preview(
    project_dir,
    source_path,
    mask_rows,
    mask_type,
    points1=None,
    points2=None,
    display_mtf=None,
    source_kind="image",
    display_range=None,
    display_cmap="twilight",
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    data, header = scalar_image_from_path(source_path)
    data = np.asarray(data, dtype=float)
    height, width = data.shape
    pix_scale = pixel_scale_arcsec(header, 1.0) or 1.0
    extent = [
        -0.5 * width * pix_scale,
        0.5 * width * pix_scale,
        -0.5 * height * pix_scale,
        0.5 * height * pix_scale,
    ]
    mask = np.asarray(mask_rows if mask_rows is not None else [], dtype=float)
    if mask.shape != data.shape:
        mask = np.zeros_like(data, dtype=float)
    else:
        mask = np.flipud(np.where(mask > 0, 1, 0).astype(float))

    color_by_type = {
        "mask_1": "#9933ff",
        "mask_2": "#ff3300",
        "mask_out": "#ff4f5f",
    }
    contour_color = color_by_type.get(mask_type, "#9933ff")
    points1 = np.asarray(conjugate_points_extent_array(points1 or [], width, height, pix_scale), dtype=float).reshape(-1, 2)
    points2 = np.asarray(conjugate_points_extent_array(points2 or [], width, height, pix_scale), dtype=float).reshape(-1, 2)

    preview_dir = Path(project_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output = preview_dir / f"mask_preview_{safe_filename(mask_type)}_{safe_filename(source_kind)}.png"

    dpi = 160
    render_scale = 8
    fig = plt.figure(figsize=(max(width * render_scale, 1) / dpi, max(height * render_scale, 1) / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    preview_mtf = subtracted_display_mtf(display_mtf) if source_kind == "subtracted" else display_mtf
    preview_range = None if source_kind == "subtracted" else display_range
    preview_cmap = normalize_display_cmap(display_cmap)
    ax.imshow(
        mtf_stretch_image(data, preview_mtf, preview_range),
        extent=extent,
        cmap=preview_cmap,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    if np.any(mask > 0):
        ax.contour(
            mask,
            levels=[0.5],
            colors=contour_color,
            alpha=0.95,
            linestyles="dashed",
            origin="lower",
            extent=extent,
        )
    if points1.size:
        ax.plot(points1[:, 0], points1[:, 1], "o", color="black", markersize=1)
    if points2.size:
        ax.plot(points2[:, 0], points2[:, 1], "o", color="C1", markersize=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=4,
        width=0.8,
        colors="black",
        labelsize=7,
        pad=-12,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    fig.savefig(output, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return output


def mask_preview_payload(payload):
    import numpy as np

    project = project_payload_for_request(payload)
    project_dir = Path(project["project_folder"]).expanduser()
    source_path_value = str(payload.get("source_path") or "").strip()
    if source_path_value:
        source_path = Path(source_path_value).expanduser()
        if not source_path.is_absolute():
            source_path = resolve_workspace_path(source_path)
        source_path = source_path.resolve()
    else:
        source_path = project_data_cutout_path(project_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Mask preview source not found: {source_path}")

    mask_type = str(payload.get("mask_type") or "mask_1").strip()
    if mask_type not in {"mask_1", "mask_2", "mask_out"}:
        raise ValueError(f"Unsupported mask_type: {mask_type}")
    mask_rows = payload.get("mask")
    if mask_rows is None:
        mask_rows = np.zeros(scalar_image_from_path(source_path)[0].shape, dtype=np.uint8)
    source_kind = str(payload.get("source_kind") or "image").strip() or "image"
    display_range = project_display_range(project, payload)
    display_mtf = (
        project_lens_light_display_mtf(project, payload)
        if source_kind == "subtracted"
        else project_display_mtf(project, payload)
    )
    display_cmap = normalize_display_cmap(payload.get("display_cmap"))
    output = save_mask_matplotlib_preview(
        project_dir,
        source_path,
        mask_rows,
        mask_type,
        payload.get("conjugate_points_source1") or payload.get("conjugate_points") or [],
        payload.get("conjugate_points_source2") or [],
        display_mtf,
        source_kind,
        display_range,
        display_cmap,
    )
    return {
        "message": "Rendered matplotlib mask preview.",
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "project_folder": str(project_dir),
        "source_path": str(source_path),
        "mask_type": mask_type,
        "preview_url": relative_url_with_mtime(output),
    }


def list_projects():
    PROJECT_FOLDER_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(PROJECT_FOLDER_DIR.glob("*/project.json"))
    if PROJECT_STATE_PATH.exists():
        try:
            latest_payload = project_listing_metadata(PROJECT_STATE_PATH)
            latest_path = project_payload_path(latest_payload)
            if latest_path.exists() and latest_path not in paths:
                paths.insert(0, latest_path)
        except Exception:
            pass
    signature = project_list_signature([*paths, OVERLEAF_PAPER_PATH])
    now = time.time()
    with PROJECT_LIST_CACHE_LOCK:
        if (
            PROJECT_LIST_CACHE["signature"] == signature
            and now - float(PROJECT_LIST_CACHE.get("cached_at") or 0) < PROJECT_LIST_CACHE_TTL
        ):
            return [dict(project) for project in PROJECT_LIST_CACHE["projects"]]

        projects = []
        seen = set()
        catalog_coordinates = overleaf_catalog_coordinates()
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = project_listing_metadata(path)
            except Exception:
                continue
            project_id = payload.get("project_id", path.parent.name)
            if project_id in seen:
                continue
            seen.add(project_id)
            preview_url, preview_label = project_preview_for_listing(path.parent)
            svi_status = project_svi_listing_status(path.parent)
            script_status = project_generated_script_status(path.parent, path)
            catalog_match = overleaf_catalog_match(project_id, catalog_coordinates)
            projects.append(
                {
                    "project_id": project_id,
                    "project_name": payload.get("project_name", ""),
                    "project_folder": payload.get("project_folder", ""),
                    "comment": payload.get("comment", ""),
                    "rating": payload.get("rating", 0),
                    "saved_at": payload.get("saved_at", ""),
                    "created_at": payload.get("created_at", ""),
                    "path": payload.get("path", str(path)),
                    "preview_url": preview_url,
                    "preview_label": preview_label,
                    "in_overleaf_catalog": catalog_match is not None,
                    "overleaf_catalog_match": catalog_match,
                    **svi_status,
                    **script_status,
                }
            )
        projects.sort(
            key=lambda project: (
                float(project.get("rating") or 0),
                str(project.get("saved_at") or ""),
            ),
            reverse=True,
        )
        PROJECT_LIST_CACHE["signature"] = signature
        PROJECT_LIST_CACHE["projects"] = [dict(project) for project in projects]
        PROJECT_LIST_CACHE["cached_at"] = time.time()
        return projects


def latest_example_code_payload():
    project = project_payload_or_default()
    state = project.get("state", {})
    lens_state = state.get("lens", {}) if isinstance(state, dict) else {}
    model_config = dict(lens_state.get("model_config") or {})
    mass_prior = dict(model_config.pop("mass_prior", {}) or {})
    payload = {**model_config, **mass_prior}
    if state.get("data_folder") and not payload.get("data_folder"):
        payload["data_folder"] = state["data_folder"]
    payload["project_id"] = project.get("project_id")
    payload["project_name"] = project.get("project_name")
    payload["project_folder"] = project.get("project_folder")
    payload, artifact_manifest = standardize_project_artifacts(payload)

    job_dir = project_output_base(payload, "lens_model_runs") / "lens_svi_preview"
    generated_code, normalized_config = generate_lens_script(
        payload,
        job_dir=job_dir,
        default_data_dir=DEFAULT_DATA_DIR,
    )
    return {
        "project_id": project.get("project_id", ""),
        "saved_at": project.get("saved_at", ""),
        "created_at": project.get("created_at", ""),
        "generated_code": generated_code,
        "model_config": normalized_config,
        "project_artifacts": artifact_manifest,
        "source": "latest project model_config + current lens_code_generator.py",
    }


def gaussian_fit_cutout(path, x_display, y_display, side_length):
    import numpy as np
    from scipy.optimize import least_squares

    data, _, is_rgb = load_displayable_image(path)
    if is_rgb:
        data_fit = np.asarray(data[..., :3], dtype=float).mean(axis=2)
    else:
        data_fit = np.asarray(data, dtype=float)

    x_center, y_center, size, x0, x1, y0, y1 = cutout_bounds_from_display(
        data_fit.shape,
        x_display,
        y_display,
        side_length,
    )
    cutout = data_fit[y0:y1, x0:x1]
    finite = np.isfinite(cutout)
    if cutout.size == 0 or not np.any(finite):
        raise ValueError("Gaussian fit cutout has no finite pixels.")

    values = cutout[finite]
    background0 = float(np.nanmedian(values))
    peak_index = np.nanargmax(np.where(finite, cutout, -np.inf))
    y_peak, x_peak = np.unravel_index(peak_index, cutout.shape)
    amplitude0 = max(float(cutout[y_peak, x_peak] - background0), np.finfo(float).eps)
    sigma0 = max(size / 6.0, 1.0)
    yy, xx = np.indices(cutout.shape)

    def model(params):
        amp, x_fit, y_fit, sx, sy, theta, offset = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xp = (xx - x_fit) * cos_t + (yy - y_fit) * sin_t
        yp = -(xx - x_fit) * sin_t + (yy - y_fit) * cos_t
        return offset + amp * np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))

    def residual(params):
        return (model(params) - cutout)[finite]

    lower = [0.0, 0.0, 0.0, 0.5, 0.5, -np.pi / 2, float(np.nanmin(values))]
    upper = [
        np.inf,
        max(cutout.shape[1] - 1, 0),
        max(cutout.shape[0] - 1, 0),
        max(size, 1),
        max(size, 1),
        np.pi / 2,
        float(np.nanmax(values)),
    ]
    initial = [amplitude0, float(x_peak), float(y_peak), sigma0, sigma0, 0.0, background0]
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=500)
    amp, x_fit, y_fit, sx, sy, theta, offset = result.x

    x_data = float(x0 + x_fit)
    y_data = float(y0 + y_fit)
    height = data_fit.shape[0]
    x_display_fit = float(x_data + 0.5)
    y_display_fit = float(height - 0.5 - y_data)
    return {
        "center_display": {"x": x_display_fit, "y": y_display_fit},
        "center_data": {"x": x_data, "y": y_data},
        "initial_center_display": {"x": x_center + 0.5, "y": height - 0.5 - y_center},
        "bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "params": {
            "amplitude": float(amp),
            "sigma_x": float(sx),
            "sigma_y": float(sy),
            "theta": float(theta),
            "offset": float(offset),
            "cost": float(result.cost),
            "success": bool(result.success),
        },
    }


def gaussian_fit_fixed_fwhm(path, x_display, y_display, fwhm=1.5, side_length=4):
    import numpy as np
    from scipy.optimize import least_squares

    data, header = scalar_image_from_path(path)
    brightest = brightest_display_pixel_near(data, x_display, y_display, radius=1)
    x_center, y_center, size, x0, x1, y0, y1 = cutout_bounds_from_display(
        data.shape,
        brightest["x_display"],
        brightest["y_display"],
        side_length,
    )
    cutout = data[y0:y1, x0:x1]
    finite = np.isfinite(cutout)
    if cutout.size == 0 or not np.any(finite):
        raise ValueError("Gaussian fit cutout has no finite pixels.")

    values = cutout[finite]
    background0 = float(np.nanmedian(values))
    peak_index = np.nanargmax(np.where(finite, cutout, -np.inf))
    y_peak, x_peak = np.unravel_index(peak_index, cutout.shape)
    amplitude0 = max(float(cutout[y_peak, x_peak] - background0), np.finfo(float).eps)
    sigma = float(fwhm) / 2.354820045
    yy, xx = np.indices(cutout.shape)

    def model(params):
        amp, x_fit, y_fit, offset = params
        rr2 = (xx - x_fit) ** 2 + (yy - y_fit) ** 2
        return offset + amp * np.exp(-0.5 * rr2 / sigma**2)

    def residual(params):
        return (model(params) - cutout)[finite]

    lower = [0.0, 0.0, 0.0, float(np.nanmin(values))]
    upper = [
        np.inf,
        max(cutout.shape[1] - 1, 0),
        max(cutout.shape[0] - 1, 0),
        float(np.nanmax(values)),
    ]
    initial = [amplitude0, float(x_peak), float(y_peak), background0]
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=300)
    amp, x_fit, y_fit, offset = result.x

    x_data = float(x0 + x_fit)
    y_data = float(y0 + y_fit)
    height, width = data.shape
    x_display_fit = float(x_data + 0.5)
    y_display_fit = float(height - 0.5 - y_data)
    pix_scale = pixel_scale_arcsec(header, 1.0)
    x_arcsec = float((x_data - (width - 1) / 2.0) * pix_scale)
    y_arcsec = float((y_data - (height - 1) / 2.0) * pix_scale)
    return {
        "center_display": {"x": x_display_fit, "y": y_display_fit},
        "center_data": {"x": x_data, "y": y_data},
        "brightest_pixel_display": {"x": brightest["x_display"], "y": brightest["y_display"]},
        "brightest_pixel_data": {"x": brightest["x_data"], "y": brightest["y_data"]},
        "brightest_pixel_value": brightest["value"],
        "brightest_pixel_search_radius": brightest["radius"],
        "center_arcsec": {"x": x_arcsec, "y": y_arcsec},
        "pixel_scale_arcsec": float(pix_scale),
        "fwhm": float(fwhm),
        "bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "params": {
            "amplitude": float(amp),
            "sigma": float(sigma),
            "offset": float(offset),
            "cost": float(result.cost),
            "success": bool(result.success),
        },
    }


def clicked_conjugate_point(path, x_display, y_display):
    data, header = scalar_image_from_path(path)
    height, width = data.shape[:2]
    x_display_clamped = float(min(max(float(x_display), 0.0), float(width)))
    y_display_clamped = float(min(max(float(y_display), 0.0), float(height)))
    x_data = float(x_display_clamped - 0.5)
    y_data = float(height - 0.5 - y_display_clamped)
    pix_scale = pixel_scale_arcsec(header, 1.0)
    x_arcsec = float((x_data - (width - 1) / 2.0) * pix_scale)
    y_arcsec = float((y_data - (height - 1) / 2.0) * pix_scale)
    return {
        "method": "click",
        "center_display": {"x": x_display_clamped, "y": y_display_clamped},
        "center_data": {"x": x_data, "y": y_data},
        "center_arcsec": {"x": x_arcsec, "y": y_arcsec},
        "pixel_scale_arcsec": float(pix_scale),
        "fwhm": None,
        "params": {
            "method": "click",
            "success": True,
        },
    }


def gaussian_conjugate_point(path, x_display, y_display, side_length=9):
    data, header = scalar_image_from_path(path)
    height, width = data.shape[:2]
    fit = gaussian_fit_cutout(path, x_display, y_display, side_length)
    x_data = float(fit["center_data"]["x"])
    y_data = float(fit["center_data"]["y"])
    pix_scale = pixel_scale_arcsec(header, 1.0)
    x_arcsec = float((x_data - (width - 1) / 2.0) * pix_scale)
    y_arcsec = float((y_data - (height - 1) / 2.0) * pix_scale)
    params = dict(fit.get("params") or {})
    params["method"] = "gaussian"
    sigma_x = params.get("sigma_x")
    sigma_y = params.get("sigma_y")
    fwhm = None
    if sigma_x is not None and sigma_y is not None:
        fwhm = float(2.354820045 * (float(sigma_x) * float(sigma_y)) ** 0.5)
    fit.update(
        {
            "method": "gaussian",
            "center_arcsec": {"x": x_arcsec, "y": y_arcsec},
            "pixel_scale_arcsec": float(pix_scale),
            "fwhm": fwhm,
            "params": params,
        }
    )
    return fit


def brightest_conjugate_pixel(path, x_display, y_display):
    data, header = scalar_image_from_path(path)
    brightest = brightest_display_pixel_near(data, x_display, y_display, radius=1)
    height, width = data.shape[:2]
    x_data = float(brightest["x_data"])
    y_data = float(brightest["y_data"])
    pix_scale = pixel_scale_arcsec(header, 1.0)
    x_arcsec = float((x_data - (width - 1) / 2.0) * pix_scale)
    y_arcsec = float((y_data - (height - 1) / 2.0) * pix_scale)
    return {
        "method": "brightest",
        "center_display": {"x": brightest["x_display"], "y": brightest["y_display"]},
        "center_data": {"x": x_data, "y": y_data},
        "brightest_pixel_display": {"x": brightest["x_display"], "y": brightest["y_display"]},
        "brightest_pixel_data": {"x": x_data, "y": y_data},
        "brightest_pixel_value": brightest["value"],
        "brightest_pixel_search_radius": brightest["radius"],
        "center_arcsec": {"x": x_arcsec, "y": y_arcsec},
        "pixel_scale_arcsec": float(pix_scale),
        "fwhm": None,
        "params": {
            "method": "brightest_3x3_pixel",
            "value": brightest["value"],
            "success": True,
        },
    }


def save_cutout(path, x_display, y_display, side_length, payload=None):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

    data, header, is_rgb = load_displayable_image(path)

    x_center, y_center, size, x0, x1, y0, y1 = cutout_bounds_from_display(
        data.shape,
        x_display,
        y_display,
        side_length,
    )

    cutout = data[y0:y1, x0:x1, ...] if is_rgb else data[y0:y1, x0:x1]
    if cutout.size == 0:
        raise ValueError("Cutout is empty. Choose a center inside the image.")

    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(path.stem)}_x{x_center}_y{y_center}_s{size}"
    cutout_dir = project_subdir(payload, "cutouts")
    fits_path = cutout_dir / f"{tag}.fits"
    png_path = cutout_dir / f"{tag}.png"

    out_header = header.copy() if header is not None else fits.Header()
    out_header["CUTX"] = (x_center, "Cutout center x pixel")
    out_header["CUTY"] = (y_center, "Cutout center y pixel")
    out_header["CUTSIZE"] = (size, "Requested cutout side length")
    out_header["CUTX0"] = (x0, "Cutout x start")
    out_header["CUTY0"] = (y0, "Cutout y start")
    fits.writeto(fits_path, cutout, header=out_header, overwrite=False)

    bounds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "width": x1 - x0, "height": y1 - y0, "source_shape": data.shape[:2]}
    project_data_path = None
    project_rms_path = None
    project_dir = project_folder_from_payload(payload or {})
    if project_dir is not None and not is_rgb:
        project_data_path, project_rms_path = write_project_cutout_artifacts(project_dir, path, cutout, out_header, bounds)

    display_data = np.asarray(cutout[..., :3] if is_rgb else cutout, dtype=float)
    finite = np.isfinite(display_data)
    values = display_data[finite]
    vmin, vmax = np.nanpercentile(values, [0.5, 99.7]) if values.size else (0.0, 1.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        if values.size:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        else:
            vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    if is_rgb:
        rgb = np.clip((display_data - vmin) / max(vmax - vmin, np.finfo(float).eps), 0.0, 1.0)
        ax.imshow(rgb, origin="lower")
    else:
        positive = np.ma.array(display_data, mask=(~np.isfinite(display_data)) | (display_data <= 0))
        if positive.count():
            log_vmin = max(float(np.nanmin(positive.compressed())), np.finfo(float).tiny)
            log_vmax = max(float(np.nanmax(positive.compressed())), log_vmin * 1.01)
            norm = mcolors.LogNorm(
                vmin=log_vmin,
                vmax=log_vmax,
            )
            ax.imshow(positive, origin="lower", cmap="gray", norm=norm)
        else:
            ax.imshow(display_data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "fits_path": str(fits_path),
        "project_data_path": str(project_data_path) if project_data_path else "",
        "project_rms_path": str(project_rms_path) if project_rms_path else "",
        "preview_url": relative_url(png_path),
        "bounds": bounds,
        "center": {"x": x_center, "y": y_center},
        "shape": list(cutout.shape),
    }


def save_mask_fits(payload):
    import numpy as np
    from astropy.io import fits

    mask = np.asarray(payload.get("mask"))
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Mask must be a non-empty 2D array.")
    mask_type = str(payload.get("mask_type") or "mask_1").strip()
    if mask_type not in {"mask_1", "mask_2", "mask_out"}:
        raise ValueError(f"Unsupported mask_type: {mask_type}")
    autosave = bool(payload.get("autosave"))
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    mask_pixels = int(np.sum(mask))
    if not autosave and mask_type in {"mask_1", "mask_2"} and mask_pixels == 0:
        raise ValueError(
            f"{mask_type} has 0 selected pixels. Select Add, then paint with Brush or define "
            "a polygon with at least three Line clicks before saving."
        )
    fits_mask = np.flipud(mask)
    source_path = Path(payload.get("source_path") or "mask").expanduser()

    stem = safe_filename(source_path.stem if source_path.name else "mask")
    masks_dir = project_subdir(payload, "masks")
    fits_path = masks_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}_{mask_type}.fits"

    project_mask_path = None
    project_folder_value = str(payload.get("project_folder") or "").strip()
    if project_folder_value:
        project_dir = Path(project_folder_value).expanduser()
        if not project_dir.is_absolute():
            project_dir = resolve_workspace_path(project_dir)
        project_mask_path = str((project_dir / f"{mask_type}.fits").resolve())

    if project_mask_path:
        fits.writeto(project_mask_path, fits_mask, overwrite=True)
        fits_path_value = project_mask_path
    else:
        fits.writeto(fits_path, fits_mask, overwrite=False)
        fits_path_value = str(fits_path)
    return {
        "fits_path": fits_path_value,
        "project_mask_path": project_mask_path,
        "mask_type": mask_type,
        "shape": list(fits_mask.shape),
        "mask_pixels": mask_pixels,
        "autosave": autosave,
    }


def load_mask_fits(payload):
    import numpy as np
    from astropy.io import fits

    mask_type = str(payload.get("mask_type") or "mask_1").strip()
    if mask_type not in {"mask_1", "mask_2", "mask_out"}:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    project_folder_value = str(payload.get("project_folder") or "").strip()
    if not project_folder_value and payload.get("project_id"):
        project_folder_value = str(project_folder(payload["project_id"]))
    if not project_folder_value:
        raise ValueError("project_folder or project_id is required.")

    project_dir = resolve_project_folder_reference(project_folder_value)
    source_path = project_data_cutout_path(project_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Data_cutout.fits not found in {project_dir}")

    source_data, source_header, _ = load_displayable_image(source_path)
    source_height, source_width = source_data.shape[:2]
    source_image = image_payload_from_array(
        source_data,
        source_header,
        source_path,
        fits_zeropoints(source_header),
        band=infer_band_from_path(source_path),
        source="project",
    )
    mask_path = project_dir / f"{mask_type}.fits"
    if mask_path.exists():
        mask, header = fits.getdata(mask_path, header=True)
        mask = np.flipud(np.where(np.asarray(mask) > 0, 1, 0).astype(np.uint8))
        height, width = mask.shape
        source_label = str(source_path)
        is_generated_default = (
            mask_type in {"mask_1", "mask_2"}
            and int(header.get("DEFAULT", -1)) == 1
            and mask.size > 0
            and int(mask.min()) == 1
            and int(mask.max()) == 1
        )
        if is_generated_default:
            mask = np.zeros((height, width), dtype=np.uint8)
            fits_path = ""
            initialized = True
        elif (height, width) != (source_height, source_width):
            height, width = source_height, source_width
            header = source_header
            mask = np.zeros((height, width), dtype=np.uint8)
            fits_path = ""
            initialized = True
        else:
            fits_path = str(mask_path)
            initialized = False
    else:
        header = source_header
        height, width = source_height, source_width
        mask = np.zeros((height, width), dtype=np.uint8)
        source_label = str(source_path)
        fits_path = ""
        initialized = True

    preview_path = project_dir / "previews" / "Data_cutout_preview.png"
    if preview_path.exists():
        preview_url = relative_url_with_mtime(preview_path)
    else:
        preview_url, _ = preview_existing_path(source_path)
    bounds = {
        "x0": int(header.get("X0", 0)),
        "x1": int(header.get("X1", width)),
        "y0": int(header.get("Y0", 0)),
        "y1": int(header.get("Y1", height)),
        "width": int(header.get("WIDTH", width)),
        "height": int(header.get("HEIGHT", height)),
    }
    bounds["x1"] = bounds["x0"] + bounds["width"]
    bounds["y1"] = bounds["y0"] + bounds["height"]
    return {
        "fits_path": fits_path,
        "mask_type": mask_type,
        "bounds": bounds,
        "shape": [height, width],
        "initialized": initialized,
        "mask_pixels": int(np.sum(mask)),
        "source_path": source_label,
        "preview_url": preview_url,
        "image": source_image,
        "mask": mask.tolist(),
    }


def lens_light_int_setting(payload, keys, default, min_value=0):
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            try:
                return max(min_value, int(float(payload.get(key))))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Lens-light setting {key} must be an integer.") from exc
    return max(min_value, int(default))


def lens_light_path_float_tag(value):
    if value is None:
        return "auto"
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def default_lens_light_init_dir(n_gauss, sigma_min, sigma_max, center_max_offset):
    return (
        "lens_light_linear_mge_pyautolens_style/"
        f"gauss{int(n_gauss)}_smin{lens_light_path_float_tag(sigma_min)}_"
        f"smax{lens_light_path_float_tag(sigma_max)}_"
        f"cmax{lens_light_path_float_tag(center_max_offset)}_jaxnnls"
    )


def lens_light_settings_from_payload(payload):
    n_gauss = max(1, int(float(payload.get("n_gauss", 8) or 8)))
    constrained_svi_steps = lens_light_int_setting(
        payload,
        ("semilinear_mge_steps", "constrained_svi_steps"),
        0,
        0,
    )
    unconstrained_svi_steps = lens_light_int_setting(
        payload,
        ("unconstrained_svi_steps", "svi_steps", "init_svi_steps"),
        2000,
        0,
    )
    sigma_min = max(float(payload.get("sigma_min", 0.01) or 0.01), 1e-6)
    sigma_max_raw = str(payload.get("sigma_max", "auto") or "auto").strip().lower()
    sigma_max = None if sigma_max_raw in {"", "auto", "half", "half_image", "half image"} else max(float(sigma_max_raw), sigma_min * 1.01)
    center_max_offset_value = payload.get("center_max_offset", 0.4)
    if center_max_offset_value in ("", None):
        center_max_offset_value = 0.4
    center_max_offset = float(center_max_offset_value)
    if not math.isfinite(center_max_offset) or center_max_offset <= 0:
        raise ValueError("Lens-light Gaussian center max offset must be a positive number in arcsec.")
    init_dir = str(
        payload.get("init_dir")
        or default_lens_light_init_dir(n_gauss, sigma_min, sigma_max, center_max_offset)
    )
    background_corner_raw = str(payload.get("background_corner", "") or "").strip()
    if background_corner_raw:
        background = "corner"
        try:
            corner_size = max(1, int(float(background_corner_raw)))
        except ValueError as exc:
            raise ValueError("Lens-light subtraction BG corner px must be a number or blank.") from exc
    else:
        background = "none"
        corner_size = 0
    return {
        "n_gauss": n_gauss,
        "semilinear_mge_steps": constrained_svi_steps,
        "constrained_svi_steps": constrained_svi_steps,
        "unconstrained_svi_steps": unconstrained_svi_steps,
        "svi_steps": unconstrained_svi_steps,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "center_max_offset": center_max_offset,
        "init_dir": init_dir,
        "background": background,
        "corner_size": corner_size,
        "background_corner": background_corner_raw,
    }


def lens_light_subtraction_script_text(project_dir, gui_root, output_dir, status_path, job_id, settings=None):
    settings = lens_light_settings_from_payload(settings or {})
    template = """
    # Auto-generated constrained MGE lens-light subtraction script.
    # Source: HerculensGUI Mask stage.

    # %% Runtime setup and imports
    import os
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from pathlib import Path
    import json
    import shutil
    import sys
    import types

    PROJECT_DIR = Path(__file__).parent
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    GUI_ROOT = Path(__GUI_ROOT_LITERAL__)
    RUNNER_SOURCE = Path(__RUNNER_SOURCE_LITERAL__)
    LOCAL_RUNNER = PROJECT_DIR / "run_constrained_jax_mge_svi.py"
    if RUNNER_SOURCE.exists() and (not LOCAL_RUNNER.exists() or RUNNER_SOURCE.read_bytes() != LOCAL_RUNNER.read_bytes()):
        shutil.copy2(RUNNER_SOURCE, LOCAL_RUNNER)
    elif not LOCAL_RUNNER.exists():
        raise FileNotFoundError(
            f"Missing constrained MGE runner. Expected local copy {LOCAL_RUNNER} "
            f"or GUI source {RUNNER_SOURCE}."
        )

    # %% Paths and configuration
    JOB_ID = __JOB_ID_LITERAL__
    RUN_ID = JOB_ID
    OUTPUT_DIR = PROJECT_DIR / "__LENS_LIGHT_RESULT_DIRNAME__"
    STATUS_PATH = OUTPUT_DIR / "status.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    def write_status(state, message, fraction=0.0, **extra):
        payload = {
            "job_id": JOB_ID,
            "state": state,
            "message": message,
            "job_dir": str(OUTPUT_DIR),
            "status_path": str(STATUS_PATH),
            "script_path": str(Path(__file__)),
            "progress": {"fraction": float(fraction), "message": message},
            **extra,
        }
        STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


    try:
        from run_constrained_jax_mge_svi import LensLightMgeSVIFitter
        from astropy.io import fits
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        import numpy as np

        write_status("running", "Solving semilinear MGE lens-light model.", 0.05)
        args = types.SimpleNamespace(
            project_dir=str(PROJECT_DIR),
            init_dir=__INIT_DIR_LITERAL__,
            tag="mask_lens_light_mge_svi",
            mode="semi_linear_jaxnnls",
            n_gauss=__N_GAUSS_LITERAL__,
            sigma_min=__SIGMA_MIN_LITERAL__,
            sigma_max=__SIGMA_MAX_LITERAL__,
            steps=__CONSTRAINED_SVI_STEPS_LITERAL__,
            unconstrained_steps=__UNCONSTRAINED_SVI_STEPS_LITERAL__,
            learning_rate=0.004,
            init_scale=0.02,
            seed=100,
            num_particles=10,
            amp_delta_sigma=0.8,
            sigma_delta_sigma=0.35,
            sigma_delta_limit=1.0,
            center_delta_sigma=0.08,
            center_max_offset=__CENTER_MAX_OFFSET_LITERAL__,
            e_delta_sigma=0.12,
            e_abs_max=0.3,
            background=__BACKGROUND_LITERAL__,
            corner_size=__CORNER_SIZE_LITERAL__,
        )
        def write_svi_progress(step, total, latest_loss=None, avg_loss=None, stage=None, offset=0, total_steps=None):
            total = max(int(total), 1)
            step = min(max(int(step), 0), total)
            total_steps = max(int(total_steps or total), 1)
            completed = min(max(int(offset) + step, 0), total_steps)
            fraction = 0.05 + 0.85 * (completed / total_steps)
            if stage == "unconstrained":
                stage_label = "SVI"
            elif stage == "constrained":
                stage_label = "Semilinear MGE SVI"
            else:
                stage_label = "Semilinear MGE"
            message = f"{stage_label}: step {step}/{total} ({completed}/{total_steps})."
            if avg_loss is not None:
                message += f" avg loss {avg_loss:.4g}"
            write_status("running", message, fraction)

        result = LensLightMgeSVIFitter(args).fit_lenslight(progress_callback=write_svi_progress)
        write_status("running", "Standardizing MGE lens-light products.", 0.92)

        product_dir = Path(result["output_dir"])
        model_dst = OUTPUT_DIR / "lens_light_model.fits"
        subtracted_dst = OUTPUT_DIR / "lens_light_subtracted.fits"
        residual_dst = OUTPUT_DIR / "lens_light_fit_residual_over_rms.fits"
        fit_mask_dst = OUTPUT_DIR / "lens_light_fit_mask.fits"
        kwargs_dst = OUTPUT_DIR / "kwargs_lens_light.pkl"
        constrained_model_dst = OUTPUT_DIR / "lens_light_model_constrained.fits"
        constrained_subtracted_dst = OUTPUT_DIR / "lens_light_subtracted_constrained.fits"
        constrained_residual_dst = OUTPUT_DIR / "lens_light_fit_residual_over_rms_constrained.fits"
        constrained_kwargs_dst = OUTPUT_DIR / "kwargs_lens_light_constrained.pkl"
        shutil.copy2(result["model_fits"], model_dst)
        shutil.copy2(result["subtracted_fits"], subtracted_dst)
        shutil.copy2(result["residual_fits"], residual_dst)
        shutil.copy2(result["fit_mask_fits"], fit_mask_dst)
        shutil.copy2(result["kwargs_lens_light"], kwargs_dst)
        shutil.copy2(result["constrained_model_fits"], constrained_model_dst)
        shutil.copy2(result["constrained_subtracted_fits"], constrained_subtracted_dst)
        shutil.copy2(result["constrained_residual_fits"], constrained_residual_dst)
        shutil.copy2(result["constrained_kwargs_lens_light"], constrained_kwargs_dst)

        def write_subtracted_preview(fits_path, png_path):
            lens_light_subtracted, header = fits.getdata(fits_path, header=True)
            finite_abs = np.abs(lens_light_subtracted[np.isfinite(lens_light_subtracted)])
            finite_abs = finite_abs[finite_abs > 0]
            linthresh = float(np.nanpercentile(finite_abs, 5.0)) if finite_abs.size else 1e-6
            vmax = float(np.nanpercentile(finite_abs, 99.5)) if finite_abs.size else 1.0
            linthresh = max(linthresh, np.finfo(float).tiny)
            vmax = max(vmax, linthresh * 10.0)
            fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
            ax.imshow(
                lens_light_subtracted,
                origin="lower",
                cmap="twilight",
                norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
            )
            ax.set_axis_off()
            fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        subtracted_preview = OUTPUT_DIR / "lens_light_subtracted_preview.png"
        constrained_subtracted_preview = OUTPUT_DIR / "lens_light_subtracted_constrained_preview.png"
        write_subtracted_preview(subtracted_dst, subtracted_preview)
        write_subtracted_preview(constrained_subtracted_dst, constrained_subtracted_preview)
        semilinear_label = "Semilinear MGE SVI" if result.get("constrained_steps", 0) else "Semilinear MGE (no SVI)"
        result_variants = []
        if result.get("unconstrained_steps", 0):
            result_variants.append(
                {
                    "key": "unconstrained",
                    "label": "SVI",
                    "subtracted_fits": str(subtracted_dst),
                    "model_fits": str(model_dst),
                    "residual_fits": str(residual_dst),
                    "kwargs_lens_light": str(kwargs_dst),
                    "preview_path": str(subtracted_preview),
                }
            )
        result_variants.append(
            {
                "key": "constrained",
                "label": semilinear_label,
                "subtracted_fits": str(constrained_subtracted_dst),
                "model_fits": str(constrained_model_dst),
                "residual_fits": str(constrained_residual_dst),
                "kwargs_lens_light": str(constrained_kwargs_dst),
                "preview_path": str(constrained_subtracted_preview),
            }
        )

        summary = {
            "run_id": RUN_ID,
            "method": "semilinear_mge_jaxnnls_then_svi" if result.get("unconstrained_steps", 0) else "semilinear_mge_jaxnnls",
            "n_gauss": __N_GAUSS_LITERAL__,
            "project_dir": str(PROJECT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "source_output_dir": str(product_dir),
            "kwargs_lens_light": str(kwargs_dst),
            "semilinear_mge_steps": __CONSTRAINED_SVI_STEPS_LITERAL__,
            "constrained_svi_steps": __CONSTRAINED_SVI_STEPS_LITERAL__,
            "unconstrained_svi_steps": __UNCONSTRAINED_SVI_STEPS_LITERAL__,
            "svi_steps": __UNCONSTRAINED_SVI_STEPS_LITERAL__,
            "center_max_offset": __CENTER_MAX_OFFSET_LITERAL__,
            "init_dir": __INIT_DIR_LITERAL__,
            "background": __BACKGROUND_LITERAL__,
            "corner_size": __CORNER_SIZE_LITERAL__,
        }
        summary_path = OUTPUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_status(
            "completed",
            "Semilinear MGE lens-light subtraction completed.",
            1.0,
            preview_path=str(subtracted_preview),
            diagnostic_path=str(result["diagnostic"]),
            subtracted_fits=str(subtracted_dst),
            model_fits=str(model_dst),
            residual_fits=str(residual_dst),
            fit_mask_fits=str(fit_mask_dst),
            kwargs_lens_light=str(kwargs_dst),
            result_variants=result_variants,
            summary_path=str(summary_path),
        )
        print(f"MGE lens-light subtraction outputs written to {OUTPUT_DIR}", flush=True)
    except Exception as exc:
        write_status("failed", f"Constrained MGE lens-light subtraction failed: {exc}", 1.0)
        raise
    """
    runner_source = WEB_ROOT / LENS_LIGHT_RUNNER_NAME
    return (
        textwrap.dedent(template)
        .replace("__LENS_LIGHT_RESULT_DIRNAME__", LENS_LIGHT_RESULT_DIRNAME)
        .replace("__GUI_ROOT_LITERAL__", json.dumps(str(GUI_ROOT)))
        .replace("__RUNNER_SOURCE_LITERAL__", json.dumps(str(runner_source)))
        .replace("__JOB_ID_LITERAL__", json.dumps(str(job_id)))
        .replace("__N_GAUSS_LITERAL__", json.dumps(settings["n_gauss"]))
        .replace("__SIGMA_MIN_LITERAL__", json.dumps(settings["sigma_min"]))
        .replace("__SIGMA_MAX_LITERAL__", json.dumps(settings["sigma_max"]))
        .replace("__CENTER_MAX_OFFSET_LITERAL__", json.dumps(settings["center_max_offset"]))
        .replace("__CONSTRAINED_SVI_STEPS_LITERAL__", json.dumps(settings["constrained_svi_steps"]))
        .replace("__UNCONSTRAINED_SVI_STEPS_LITERAL__", json.dumps(settings["unconstrained_svi_steps"]))
        .replace("__TOTAL_SVI_STEPS_LITERAL__", json.dumps(settings["constrained_svi_steps"] + settings["unconstrained_svi_steps"]))
        .replace("__INIT_DIR_LITERAL__", json.dumps(settings["init_dir"]))
        .replace("__BACKGROUND_LITERAL__", json.dumps(settings["background"]))
        .replace("__CORNER_SIZE_LITERAL__", json.dumps(settings["corner_size"]))
        .strip()
        + "\n"
    )


def generate_lens_light_subtraction_script(payload):
    payload, manifest = standardize_project_artifacts(payload)
    project_dir = Path(payload["project_folder"]).expanduser()
    output_dir = project_dir / LENS_LIGHT_RESULT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d_%H%M%S")
    script_output_dir = output_dir
    script_path = project_dir / LENS_LIGHT_SCRIPT_NAME
    status_path = output_dir / "status.json"
    script_path.write_text(
        lens_light_subtraction_script_text(project_dir, GUI_ROOT, script_output_dir, status_path, job_id, payload),
        encoding="utf-8",
    )
    return {
        "message": "Generated independent lens-light subtraction SVI script.",
        "script_path": str(script_path),
        "output_dir": str(script_output_dir),
        "project_folder": str(project_dir),
        "project_artifacts": manifest,
    }


def lens_light_subtraction_status_paths(project_dir=None):
    if project_dir is not None:
        root = Path(project_dir)
        paths = list(root.glob("lens_light_subtraction/lens_light_subtraction_*/status.json"))
        fixed = root / LENS_LIGHT_RESULT_DIRNAME / "status.json"
        if fixed.exists():
            paths.append(fixed)
        return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)
    paths = list(PROJECT_FOLDER_DIR.glob("*/lens_light_subtraction/lens_light_subtraction_*/status.json"))
    paths.extend(PROJECT_FOLDER_DIR.glob(f"*/{LENS_LIGHT_RESULT_DIRNAME}/status.json"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def lens_light_job_pid(job_dir):
    pid_path = Path(job_dir) / "pid.txt"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def find_active_lens_light_subtraction_job(project_dir):
    for status_path in lens_light_subtraction_status_paths(project_dir):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("state") in LENS_LIGHT_TERMINAL_STATES:
            continue
        pid = lens_light_job_pid(status_path.parent)
        if pid is not None and pid_is_running(pid):
            return status_path
    return None


def start_lens_light_subtraction_job(payload):
    payload, manifest = standardize_project_artifacts(payload)
    settings = lens_light_settings_from_payload(payload)
    project_dir = Path(payload["project_folder"]).expanduser()
    active = find_active_lens_light_subtraction_job(project_dir)
    if active is not None:
        active_status = json.loads(active.read_text(encoding="utf-8"))
        status = read_lens_light_subtraction_job_status(active_status.get("job_id", active.parent.name.removeprefix("lens_light_subtraction_")))
        return {
            **status,
            "message": f"Lens-light subtraction job {status['job_id']} is already running.",
            "existing_job": True,
        }

    job_id = time.strftime("%Y%m%d_%H%M%S")
    job_dir = project_dir / LENS_LIGHT_RESULT_DIRNAME
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"
    log_path = job_dir / "run.log"
    script_path = project_dir / LENS_LIGHT_SCRIPT_NAME
    script_path.write_text(
        lens_light_subtraction_script_text(project_dir, GUI_ROOT, job_dir, status_path, job_id, settings),
        encoding="utf-8",
    )

    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Lens-light subtraction job queued.",
            "job_dir": str(job_dir),
            "status_path": str(status_path),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "project_folder": str(project_dir),
            "project_artifacts": manifest,
            "lens_light_settings": settings,
            "progress": {"fraction": 0.0, "message": "Lens-light subtraction job queued."},
        },
    )

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(configured_python("herculens_python")), "-u", str(script_path)],
            cwd=str(project_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["pid"] = process.pid
    status["message"] = f"Started lens-light subtraction job {job_id}."
    status["state"] = "running"
    status["progress"] = {"fraction": 0.02, "message": "Lens-light subtraction job started."}
    atomic_write_json(status_path, status)
    return status


def ensure_lens_light_runner_in_project(project_dir):
    runner_source = WEB_ROOT / LENS_LIGHT_RUNNER_NAME
    runner_target = Path(project_dir) / LENS_LIGHT_RUNNER_NAME
    if not runner_source.exists():
        raise FileNotFoundError(f"Missing lens-light runner source: {runner_source}")
    if not runner_target.exists() or runner_source.read_bytes() != runner_target.read_bytes():
        shutil.copy2(runner_source, runner_target)
    return runner_target


def lens_light_sciama_slim_files(project_dir, settings):
    slim_files = list(LENS_LIGHT_SCIAMA_SLIM_FILES)
    init_dir = str(settings.get("init_dir") or "").strip()
    if init_dir and not Path(init_dir).is_absolute():
        init_path = Path(project_dir) / init_dir
        if init_path.exists():
            slim_files.append(init_dir.rstrip("/\\"))
    seen = set()
    unique = []
    for item in slim_files:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def start_sciama_lens_light_subtraction_job(payload):
    project_dir = existing_project_dir_from_payload(payload)
    payload = dict(payload or {})
    payload["project_folder"] = str(project_dir)
    payload, manifest = standardize_project_artifacts(payload)
    settings = lens_light_settings_from_payload(payload)
    project_dir = Path(payload["project_folder"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = resolve_workspace_path(project_dir)
    required_inputs = ("Data_cutout.fits", "RMS_map.fits", "PSF_model.fits", "mask_1.fits")
    missing_inputs = [name for name in required_inputs if not (project_dir / name).exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot submit Sciama lens-light subtraction; missing required project files: "
            + ", ".join(missing_inputs)
        )
    active = find_active_lens_light_subtraction_job(project_dir)
    if active is not None:
        active_status = json.loads(active.read_text(encoding="utf-8"))
        status = read_lens_light_subtraction_job_status(active_status.get("job_id", active.parent.name.removeprefix("lens_light_subtraction_")))
        return {
            **status,
            "message": f"Lens-light subtraction job {status['job_id']} is already running.",
            "existing_job": True,
        }

    job_id = time.strftime("%Y%m%d_%H%M%S")
    job_dir = project_dir / LENS_LIGHT_RESULT_DIRNAME
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"
    log_path = job_dir / "run.log"
    config_path = job_dir / "config.json"
    script_path = project_dir / LENS_LIGHT_SCRIPT_NAME
    script_path.write_text(
        lens_light_subtraction_script_text(project_dir, GUI_ROOT, job_dir, status_path, job_id, settings),
        encoding="utf-8",
    )
    ensure_lens_light_runner_in_project(project_dir)

    project_name = payload.get("project_name") or payload.get("project_id") or project_dir.name
    safe_project = safe_sciama_name(project_name)
    remote_base = str(payload.get("sciama_remote_base") or SCIAMA_REMOTE_BASE).rstrip("/")
    remote_dir = f"{remote_base}/{safe_project}_lenslight_{job_id}"
    stage_dir = RUNS_ROOT / "_sciama_upload" / f"{safe_project}_lenslight_{job_id}"
    slurm_job_name = f"HGLT_{safe_project[:12]}_{job_id[-6:]}"
    requested_gres = str(payload.get("sciama_gres") or "").strip()
    requested_partition = str(payload.get("sciama_partition") or "gpu.q").strip() or "gpu.q"
    if requested_gres:
        gpu_candidates = [
            {
                "partition": requested_partition,
                "gres": requested_gres,
                "label": f"{requested_gres} on {requested_partition}",
            }
        ]
        candidate_mode = "custom"
    else:
        gpu_candidates = [dict(candidate) for candidate in SCIAMA_GPU_REQUEST_CANDIDATES]
        candidate_mode = "default"
    first_gpu_candidate = gpu_candidates[0]
    gres_candidates = [str(candidate["gres"]) for candidate in gpu_candidates]
    gpu_candidate_summary = ",".join(f"{candidate['partition']}:{candidate['gres']}" for candidate in gpu_candidates)
    config = {
        "job_id": job_id,
        "runner": str(SCIAMA_GPU_RUNNER),
        "runner_label": "sciama_lens_light_gpu",
        "runner_type": "sciama_gpu",
        "job_label": "lens-light subtraction",
        "project_id": payload.get("project_id"),
        "project_name": payload.get("project_name"),
        "project_dir": str(project_dir),
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "status_path": str(status_path),
        "script_path": str(script_path),
        "remote_script_name": LENS_LIGHT_SCRIPT_NAME,
        "result_dir_name": LENS_LIGHT_RESULT_DIRNAME,
        "stage_dir": str(stage_dir),
        "remote_base": remote_base,
        "remote_dir": remote_dir,
        "slim_files": lens_light_sciama_slim_files(project_dir, settings),
        "conda_activate": str(payload.get("sciama_conda_activate") or SCIAMA_CONDA_ACTIVATE),
        "partition": str(first_gpu_candidate["partition"]),
        "gres": str(first_gpu_candidate["gres"]),
        "gres_candidates": gres_candidates,
        "gpu_candidates": gpu_candidates,
        "gpu_candidate_mode": candidate_mode,
        "slurm_job_name": slurm_job_name,
        "project_artifacts": manifest,
        "lens_light_settings": settings,
        "completed_message": "Sciama GPU lens-light subtraction completed.",
        "running_message": "Running lens-light subtraction on Sciama GPU.",
        "downloading_message": "Downloading Sciama lens-light subtraction results.",
    }
    atomic_write_json(config_path, config)
    atomic_write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Sciama GPU lens-light subtraction job queued.",
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "project_folder": str(project_dir),
            "project_artifacts": manifest,
            "lens_light_settings": settings,
            "runner": "sciama_lens_light_gpu",
            "runner_type": "sciama_gpu",
            "remote_dir": remote_dir,
            "partition": str(first_gpu_candidate["partition"]),
            "gres": str(first_gpu_candidate["gres"]),
            "gres_candidates": gres_candidates,
            "gpu_candidates": gpu_candidates,
            "gpu_candidate_mode": candidate_mode,
            "progress": {"fraction": 0.0, "message": "Sciama GPU lens-light subtraction job queued."},
        },
    )
    log_path.write_text(
        f"SCIAMA_GPU_LENS_LIGHT_JOB job_id={job_id} mode={candidate_mode} remote_dir={remote_dir} gpu_candidates={gpu_candidate_summary}\n",
        encoding="utf-8",
    )
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(sys.executable), "-u", str(SCIAMA_GPU_RUNNER), str(config_path)],
            cwd=str(GUI_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    status = read_json_with_retry(status_path)
    status.update(
        {
            "pid": process.pid,
            "message": f"Started Sciama GPU lens-light subtraction job {job_id}.",
            "state": "queued",
            "progress": {
                "fraction": 0.01,
                "message": f"Sciama GPU lens-light subtraction started; trying {first_gpu_candidate['label']} first.",
            },
        }
    )
    atomic_write_json(status_path, status)
    return status


def infer_lens_light_result_variants(job_dir):
    job_dir = Path(job_dir)
    variants = []
    include_unconstrained = True
    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            unconstrained_steps = summary.get("unconstrained_svi_steps", summary.get("svi_steps"))
            if unconstrained_steps is not None:
                include_unconstrained = int(unconstrained_steps) > 0
        except Exception:
            pass
    svi = {
        "key": "unconstrained",
        "label": "SVI",
        "subtracted_fits": job_dir / "lens_light_subtracted.fits",
        "model_fits": job_dir / "lens_light_model.fits",
        "residual_fits": job_dir / "lens_light_fit_residual_over_rms.fits",
        "kwargs_lens_light": job_dir / "kwargs_lens_light.pkl",
        "preview_path": job_dir / "lens_light_subtracted_preview.png",
    }
    semilinear = {
        "key": "constrained",
        "label": "Semilinear MGE (no SVI)",
        "subtracted_fits": job_dir / "lens_light_subtracted_constrained.fits",
        "model_fits": job_dir / "lens_light_model_constrained.fits",
        "residual_fits": job_dir / "lens_light_fit_residual_over_rms_constrained.fits",
        "kwargs_lens_light": job_dir / "kwargs_lens_light_constrained.pkl",
        "preview_path": job_dir / "lens_light_subtracted_constrained_preview.png",
    }
    candidates = (svi, semilinear) if include_unconstrained else (semilinear,)
    for candidate in candidates:
        if candidate["subtracted_fits"].exists():
            variants.append({key: str(value) for key, value in candidate.items()})
    return variants


def read_lens_light_subtraction_job_status(job_id):
    matches = sorted(PROJECT_FOLDER_DIR.glob(f"*/lens_light_subtraction/lens_light_subtraction_{safe_filename(job_id)}/status.json"))
    if matches:
        status_path = matches[-1]
    else:
        status_path = None
        for candidate in lens_light_subtraction_status_paths():
            try:
                status = read_json_with_retry(candidate)
            except Exception:
                continue
            if str(status.get("job_id") or "") == str(job_id):
                status_path = candidate
                break
    if status_path is None:
        raise FileNotFoundError(f"Unknown lens-light subtraction job: {job_id}")
    status = read_json_with_retry(status_path)
    job_dir = status_path.parent
    pid = lens_light_job_pid(job_dir)
    log_path = Path(status.get("log_path") or job_dir / "run.log")

    if status.get("state") not in LENS_LIGHT_TERMINAL_STATES:
        running = pid is not None and pid_is_running(pid)
        if running:
            status["state"] = "running"
            status.setdefault("progress", {"fraction": 0.05, "message": status.get("message", "Lens-light subtraction running.")})
        else:
            text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
            preview_path = Path(status.get("preview_path") or job_dir / "lens_light_subtracted_preview.png")
            if "Traceback (most recent call last)" in text or "SyntaxError:" in text or "IndentationError:" in text:
                last_error = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "Lens-light subtraction failed.")
                status["state"] = "failed"
                status["message"] = f"Lens-light subtraction failed: {last_error}"
                status["progress"] = {"fraction": 1.0, "message": status["message"], "failed": True}
            elif preview_path.exists():
                status["state"] = "completed"
                status["message"] = "Lens-light subtraction completed."
                status["progress"] = {"fraction": 1.0, "message": status["message"]}
                status["preview_path"] = str(preview_path)
            else:
                status["state"] = "failed"
                status["message"] = "Lens-light subtraction stopped before writing preview."
                status["progress"] = {"fraction": 1.0, "message": status["message"], "failed": True}
            atomic_write_json(status_path, status)

    inferred_variants = infer_lens_light_result_variants(job_dir)
    if inferred_variants:
        existing_variants = status.get("result_variants") if isinstance(status.get("result_variants"), list) else []
        inferred_keys = {str(item.get("key") or "") for item in inferred_variants}
        if "unconstrained" not in inferred_keys:
            existing_variants = [
                item for item in existing_variants
                if not isinstance(item, dict) or str(item.get("key") or "") != "unconstrained"
            ]
        by_key = {str(item.get("key") or ""): dict(item) for item in existing_variants if isinstance(item, dict)}
        for variant in inferred_variants:
            key = str(variant.get("key") or "")
            by_key[key] = {**variant, **by_key.get(key, {})}
        status["result_variants"] = [by_key[key] for key in ("unconstrained", "constrained") if key in by_key]
        primary = status["result_variants"][0]
        status.setdefault("subtracted_fits", primary.get("subtracted_fits"))
        status.setdefault("model_fits", primary.get("model_fits"))
        status.setdefault("residual_fits", primary.get("residual_fits"))
        status.setdefault("kwargs_lens_light", primary.get("kwargs_lens_light"))
        status.setdefault("preview_path", primary.get("preview_path"))
        summary_path = job_dir / "summary.json"
        if summary_path.exists():
            status.setdefault("summary_path", str(summary_path))

    preview_path = status.get("preview_path")
    if preview_path and Path(preview_path).exists():
        status["preview_url"] = relative_url(Path(preview_path))
    if isinstance(status.get("result_variants"), list):
        variants = []
        for variant in status.get("result_variants") or []:
            if not isinstance(variant, dict):
                continue
            item = dict(variant)
            if item.get("key") == "constrained" and item.get("label") == "Semilinear MGE":
                item["label"] = "Semilinear MGE (no SVI)"
            for source_key, url_key in (
                ("preview_path", "preview_url"),
                ("subtracted_fits", "subtracted_url"),
                ("model_fits", "model_url"),
                ("residual_fits", "residual_url"),
            ):
                value = item.get(source_key)
                if value and Path(value).exists():
                    item[url_key] = relative_url(Path(value))
            variants.append(item)
        status["result_variants"] = variants
    diagnostic_path = status.get("diagnostic_path")
    if diagnostic_path and Path(diagnostic_path).exists():
        status["diagnostic_url"] = relative_url(Path(diagnostic_path))
    summary_path = status.get("summary_path")
    if summary_path and Path(summary_path).exists():
        try:
            status["summary"] = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    if not isinstance(status.get("summary"), dict):
        status["summary"] = {}
    if not status["summary"].get("n_gauss"):
        kwargs_path = status.get("kwargs_lens_light")
        if kwargs_path and Path(kwargs_path).exists():
            try:
                import pickle

                with Path(kwargs_path).open("rb") as handle:
                    kwargs_lens_light = pickle.load(handle)
                if isinstance(kwargs_lens_light, dict):
                    kwargs_lens_light = [kwargs_lens_light]
                first = kwargs_lens_light[0] if isinstance(kwargs_lens_light, (list, tuple)) and kwargs_lens_light else {}
                n_gauss = len(first.get("sigma", [])) if isinstance(first, dict) else 0
                if n_gauss:
                    status.setdefault("summary", {})["n_gauss"] = int(n_gauss)
            except Exception:
                pass
    subtracted_fits = status.get("subtracted_fits")
    if subtracted_fits and Path(subtracted_fits).exists():
        try:
            subtracted_image = fits_payload_for_gui(subtracted_fits)
            current_shape = None
            try:
                project_dir = Path(status_path).parent.parent
                current_shape = list(fits_payload_for_gui(project_data_cutout_path(project_dir)).get("shape") or [])
            except Exception:
                current_shape = None
            if current_shape and list(subtracted_image.get("shape") or []) != current_shape:
                status["subtracted_stale"] = True
                status["subtracted_stale_message"] = (
                    f"Lens-light subtraction is {subtracted_image.get('shape')}; "
                    f"current cutout is {current_shape}. Run lens-light subtraction again."
                )
                status.pop("subtracted_image", None)
            else:
                status["subtracted_image"] = subtracted_image
                status.pop("subtracted_stale", None)
                status.pop("subtracted_stale_message", None)
        except Exception:
            pass
    return status


def latest_lens_light_subtraction_payload(payload):
    project_dir = project_folder_from_payload(payload or {})
    if project_dir is None:
        raise ValueError("project_folder or project_id is required.")
    project_dir = Path(project_dir).expanduser().resolve()

    for status_path in lens_light_subtraction_status_paths(project_dir):
        try:
            status = read_json_with_retry(status_path)
        except Exception:
            continue
        job_id = str(status.get("job_id") or "").strip()
        if not job_id:
            if status_path.parent.name.startswith("lens_light_subtraction_"):
                job_id = status_path.parent.name.removeprefix("lens_light_subtraction_")
            else:
                job_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(status_path.stat().st_mtime))
            status["job_id"] = job_id
            atomic_write_json(status_path, status)
        return read_lens_light_subtraction_job_status(job_id)

    result_dir = project_dir / LENS_LIGHT_RESULT_DIRNAME
    variants = infer_lens_light_result_variants(result_dir)
    if not variants:
        raise FileNotFoundError(f"No lens-light subtraction result found in {project_dir}")

    job_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(result_dir.stat().st_mtime if result_dir.exists() else time.time()))
    status_path = result_dir / "status.json"
    primary = variants[0]
    status = {
        "job_id": job_id,
        "state": "completed",
        "message": "Loaded lens-light subtraction result from project folder.",
        "project_folder": str(project_dir),
        "output_dir": str(result_dir),
        "result_variants": variants,
        "subtracted_fits": primary.get("subtracted_fits"),
        "model_fits": primary.get("model_fits"),
        "residual_fits": primary.get("residual_fits"),
        "kwargs_lens_light": primary.get("kwargs_lens_light"),
        "preview_path": primary.get("preview_path"),
        "progress": {"fraction": 1.0, "message": "Loaded lens-light subtraction result from project folder."},
    }
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        status["summary_path"] = str(summary_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(status_path, status)
    return read_lens_light_subtraction_job_status(job_id)


def stop_lens_light_subtraction_job(payload):
    job_id = payload.get("job_id")
    if job_id:
        status_path = None
        for candidate in lens_light_subtraction_status_paths():
            try:
                status = read_json_with_retry(candidate)
            except Exception:
                continue
            if str(status.get("job_id") or "") == str(job_id):
                status_path = candidate
                break
        if status_path is None:
            raise FileNotFoundError(f"Unknown lens-light subtraction job: {job_id}")
    else:
        project_folder = str(payload.get("project_folder") or "").strip()
        project_dir = Path(project_folder).expanduser() if project_folder else None
        status_path = find_active_lens_light_subtraction_job(project_dir) if project_dir else None
        if status_path is None:
            raise FileNotFoundError("No active lens-light subtraction job to stop.")

    status = read_json_with_retry(status_path)
    job_dir = status_path.parent
    pid = lens_light_job_pid(job_dir)
    killed = False
    if pid is not None and pid_is_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            os.kill(pid, signal.SIGTERM)
        killed = True

    progress = dict(status.get("progress") or {})
    progress.update({"message": "Lens-light subtraction stopped.", "stopped": True})
    status.update(
        {
            "job_id": status.get("job_id") or job_id or job_dir.name,
            "state": "stopped",
            "message": "Lens-light subtraction stopped.",
            "job_dir": str(job_dir),
            "log_path": status.get("log_path", str(job_dir / "run.log")),
            "progress": progress,
            "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stopped_pid": pid,
        }
    )
    atomic_write_json(status_path, status)
    return {
        **status,
        "killed": killed,
        "message": "Lens-light subtraction stopped." if killed else "Lens-light subtraction job was not running.",
    }


def list_folder(path):
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Folder does not exist: {path}")

    entries = []
    parent = path.parent if path.parent != path else path
    entries.append({"name": "..", "path": str(parent), "kind": "folder", "previewable": False})

    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        is_dir = child.is_dir()
        suffix = child.suffix.lower()
        previewable = (not is_dir) and suffix in IMAGE_EXTENSIONS
        if is_dir or previewable:
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "kind": "folder" if is_dir else "image",
                    "previewable": previewable,
                }
            )
    return {"cwd": str(path), "parent": str(parent), "entries": entries}


class HerculensGuiHandler(BaseHTTPRequestHandler):
    server_version = "HerculensGui/0.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_get(parsed.path)
        return self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_post(parsed.path)
        json_response(self, {"detail": "Not found"}, status=404)

    def handle_api_get(self, path):
        if path == "/api/health":
            return json_response(
                self,
                {
                    "status": "Ready",
                    "gui_root": str(GUI_ROOT),
                    "cwd": str(SERVER_CWD),
                    "data_folder": str(DEFAULT_FILE_DIR),
                    "settings_path": str(SETTINGS_PATH),
                    "runtime_settings": load_runtime_settings(),
                },
            )

        if path == "/api/files":
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            folder = query.get("dir", [str(DEFAULT_FILE_DIR)])[0] or str(DEFAULT_FILE_DIR)
            try:
                payload = list_folder(Path(folder))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, payload)

        if path == "/api/file":
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            file_path = query.get("path", [""])[0]
            try:
                return self.serve_workspace_file(file_path)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)

        if path == "/api/project/latest":
            payload = project_payload_or_default()
            exists = bool(payload.pop("_project_exists", False))
            if exists:
                atomic_write_json(project_payload_path(payload), payload)
                atomic_write_json(PROJECT_STATE_PATH, payload)
            payload["exists"] = exists
            return json_response(self, payload)

        if path == "/api/projects":
            return json_response(self, {"projects": list_projects()})

        if path == "/api/settings":
            return json_response(
                self,
                {
                    "settings": load_runtime_settings(),
                    "settings_path": str(SETTINGS_PATH),
                    "backend_python": sys.executable,
                },
            )

        if path == "/api/example-code/latest":
            try:
                return json_response(self, latest_example_code_payload())
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)

        if path.startswith("/api/euclid-cutout/job/"):
            job_id = path.rsplit("/", 1)[-1]
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            project_folder_value = query.get("project_folder", [""])[0]
            try:
                return json_response(self, read_euclid_job_status(job_id, project_folder=project_folder_value))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)

        if path == "/api/image-preprocess/latest":
            return json_response(
                self,
                {
                    "message": "No image preprocess product is registered yet.",
                },
            )

        if path == "/api/psf-fit/latest":
            return json_response(
                self,
                {
                    "message": "No PSF fit product is registered yet.",
                },
            )

        if path.startswith("/api/psf-fit/job/"):
            job_id = path.rsplit("/", 1)[-1]
            matches = sorted(RUNS_ROOT.glob(f"**/psf_fit_{safe_filename(job_id)}_*/status.json"))
            if not matches:
                return json_response(self, {"detail": f"Unknown PSF job: {job_id}"}, status=404)
            try:
                return json_response(self, json.loads(matches[-1].read_text(encoding="utf-8")))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)

        if path == "/api/lens-model/latest":
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                project_id = query.get("project_id", [""])[0] or None
                include_panels = query.get("include_panels", ["0"])[0] in {"1", "true", "True", "yes"}
                return json_response(
                    self,
                    latest_lens_model_payload(project_id=project_id, include_panels=include_panels),
                )
            except Exception as exc:
                return json_response(self, {"message": str(exc)}, status=404)

        if path.startswith("/api/lens-model/job/"):
            job_id = path.rsplit("/", 1)[-1]
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                include_panels = query.get("include_panels", ["0"])[0] in {"1", "true", "True", "yes"}
                return json_response(self, read_lens_job_status(job_id, include_panels=include_panels))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)

        if path.startswith("/api/mask/lens-light-subtraction-job/"):
            job_id = path.rsplit("/", 1)[-1]
            try:
                return json_response(self, read_lens_light_subtraction_job_status(job_id))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)

        if path == "/api/mask/lens-light-subtraction-latest":
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                payload = {
                    "project_id": query.get("project_id", [""])[0],
                    "project_folder": query.get("project_folder", [""])[0],
                }
                return json_response(self, latest_lens_light_subtraction_payload(payload))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)

        return json_response(self, {"detail": "Unknown API endpoint"}, status=404)

    def handle_api_post(self, path):
        content_type = self.headers.get("Content-Type", "")
        if path == "/api/image-preprocess/run" and content_type.startswith("multipart/form-data"):
            try:
                upload_path, upload_payload = save_uploaded_file(self)
                preview_path, image_shape = save_preview_png(upload_path, upload_payload)
                metadata = image_preview_metadata(upload_path)
                artifacts = register_project_original_file(upload_path, upload_payload, make_default_cutout=True)
                image_payload = fits_payload_for_gui(artifacts.get("original_path") or upload_path)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Loaded image preview: {upload_path.name}",
                    "preview_url": relative_url(preview_path),
                    "uploaded_path": str(upload_path),
                    "source_path": artifacts.get("original_path", str(upload_path)),
                    "original_path": artifacts.get("original_path", ""),
                    "original_source_path": artifacts.get("original_source_path", str(upload_path)),
                    "data_cutout_path": artifacts.get("data_cutout_path", ""),
                    "data_cutout_preview_url": artifacts.get("data_cutout_preview_url", ""),
                    "data_cutout_bounds": artifacts.get("data_cutout_bounds"),
                    "data_cutout_shape": artifacts.get("data_cutout_shape"),
                    "psf_source_path": artifacts.get("original_path", str(upload_path)),
                    "image_shape": list(image_shape),
                    "image_payload": image_payload,
                    "pixel_scale_arcsec": metadata.get("pixel_scale_arcsec"),
                },
            )

        if path == "/api/image-preprocess/rms" and content_type.startswith("multipart/form-data"):
            try:
                upload_path, upload_payload = save_uploaded_rms(self)
                result = register_rms_request_payload(upload_path, upload_payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/psf-fit/input-preview" and content_type.startswith("multipart/form-data"):
            try:
                upload_path, upload_payload = save_uploaded_psf(self)
                preview = psf_preview_from_path(upload_path, upload_payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Loaded input PSF: {upload_path.name}",
                    **preview,
                },
            )

        try:
            payload = read_json_body(self)
        except Exception as exc:
            return json_response(self, {"detail": f"Invalid JSON: {exc}"}, status=400)

        if path == "/api/euclid-cutout/run":
            try:
                result = start_euclid_cutout_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/euclid-cutout/load-fits":
            try:
                image = fits_payload_for_gui(payload.get("path"))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Loaded Euclid FITS cutout.",
                    "image": image,
                },
            )

        if path == "/api/legacy-cutout/run":
            try:
                result = download_legacy_cutout(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/photoz/measure":
            try:
                result = measure_photoz_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/fundamental-plane/run":
            try:
                result = fundamental_plane_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/image-preprocess/rms":
            selected_path = payload.get("path") or payload.get("rms_path")
            if not selected_path:
                return json_response(self, {"detail": "Choose an RMS FITS file first."}, status=400)
            try:
                result = register_rms_request_payload(selected_path, payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/image-preprocess/run":
            selected_path = payload.get("path")
            if selected_path:
                try:
                    source_path = resolved_existing_or_workspace_path(selected_path)
                    project_dir = project_folder_from_payload(payload or {})
                    source_resolution = {}
                    if project_dir is not None:
                        source_path, source_resolution = project_preprocess_original_source(
                            source_path,
                            project_dir,
                            payload,
                            prefer_project_original=payload.get("prefer_project_original", True) is not False,
                        )
                    preview_url, image_shape = preview_existing_path(source_path, payload)
                    metadata = image_preview_metadata(source_path)
                    register_artifacts = payload.get("register_artifacts", True) is not False
                    register_default_cutout = payload.get("register_default_cutout", True) is not False
                    artifacts = (
                        register_project_original_file(source_path, payload, make_default_cutout=register_default_cutout)
                        if register_artifacts
                        else {}
                    )
                    image_payload = fits_payload_for_gui(artifacts.get("original_path") or source_path)
                except Exception as exc:
                    return json_response(self, {"detail": str(exc)}, status=400)
                response = {
                    "message": f"Loaded image preview: {source_path.name}",
                    "preview_url": preview_url,
                    "source_path": artifacts.get("original_path", str(source_path)),
                    "original_path": artifacts.get("original_path", ""),
                    "original_source_path": artifacts.get("original_source_path", str(source_path)),
                    "data_cutout_path": artifacts.get("data_cutout_path", ""),
                    "data_cutout_preview_url": artifacts.get("data_cutout_preview_url", ""),
                    "data_cutout_bounds": artifacts.get("data_cutout_bounds"),
                    "data_cutout_shape": artifacts.get("data_cutout_shape"),
                    "psf_source_path": artifacts.get("original_path", str(source_path)),
                    "image_payload": image_payload,
                    "pixel_scale_arcsec": metadata.get("pixel_scale_arcsec"),
                    **source_resolution,
                }
                if image_shape is not None:
                    response["image_shape"] = list(image_shape)
                return json_response(self, response)

            return json_response(
                self,
                {
                    "message": "Image preprocess request received. Connect this endpoint to the preprocess script next.",
                    "payload": payload,
                },
            )

        if path == "/api/project/save":
            payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            payload = compact_project_payload_for_save(payload)
            payload = normalize_project_payload(payload, create_folder=True)
            atomic_write_json(project_payload_path(payload), payload)
            atomic_write_json(PROJECT_STATE_PATH, payload)
            return json_response(
                self,
                {
                    "message": "Project saved.",
                    "project_path": str(PROJECT_STATE_PATH),
                    "project_id": payload.get("project_id", ""),
                    "project_name": payload.get("project_name", ""),
                    "project_folder": payload.get("project_folder", ""),
                    "rating": payload.get("rating", 0),
                    "saved_at": payload.get("saved_at", ""),
                    "created_at": payload.get("created_at", ""),
                },
            )

        if path == "/api/project/thumbnail":
            try:
                result = save_project_thumbnail(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/cutout-preview":
            try:
                result = save_project_cutout_preview(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/rename":
            try:
                payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if "state" not in payload and not str(payload.get("project_folder") or "").strip():
                    existing = read_project_payload(payload.get("project_id"))
                    existing["project_name"] = payload.get("project_name")
                    existing["saved_at"] = payload["saved_at"]
                    payload = existing
                payload = rename_project_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Project renamed.",
                    "project_path": str(PROJECT_STATE_PATH),
                    **payload,
                },
            )

        if path == "/api/project/delete":
            try:
                result = delete_project_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/copy":
            try:
                result = copy_project_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/comment":
            try:
                result = save_project_comment(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/rating":
            try:
                result = save_project_rating(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/project/new":
            saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
            requested_name = str(payload.get("project_name") or "").strip()
            project_id = unique_project_id(requested_name or saved_at.replace("-", "").replace(":", "").replace(" ", "_"))
            project_name = requested_name or project_id
            new_project = {
                "project_id": project_id,
                "project_name": project_name,
                "created_at": saved_at,
                "saved_at": saved_at,
                "state": {},
            }
            new_project = normalize_project_payload(new_project, create_folder=True)
            atomic_write_json(PROJECT_STATE_PATH, new_project)
            atomic_write_json(project_payload_path(new_project), new_project)
            return json_response(
                self,
                {
                    "message": "Started new project.",
                    "project_path": str(PROJECT_STATE_PATH),
                    **new_project,
                },
            )

        if path == "/api/project/load":
            project_id = payload.get("project_id")
            if not project_id:
                return json_response(self, {"detail": "project_id is required"}, status=400)
            try:
                project = normalize_project_payload(read_project_payload(project_id), create_folder=True)
            except FileNotFoundError:
                return json_response(self, {"detail": f"Project not found: {project_id}"}, status=404)
            atomic_write_json(project_payload_path(project), project)
            atomic_write_json(PROJECT_STATE_PATH, project)
            return json_response(self, {"message": f"Loaded project {project_id}.", **project})

        if path == "/api/settings/save":
            try:
                settings = save_runtime_settings(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Settings saved.",
                    "settings": settings,
                    "settings_path": str(SETTINGS_PATH),
                },
            )

        if path == "/api/settings/check":
            try:
                result = check_python_runtime(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            status = 200 if result.get("ok") else 400
            return json_response(self, result, status=status)

        if path == "/api/image-preprocess/cutout":
            try:
                source_path = Path(payload["path"]).expanduser().resolve()
                cutout = save_cutout(source_path, payload["x"], payload["y"], payload["size"], payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Saved cutout: {Path(cutout['fits_path']).name}",
                    **cutout,
                },
            )

        if path == "/api/image-preprocess/gaussian-fit":
            try:
                source_path = Path(payload["path"]).expanduser().resolve()
                fit = gaussian_fit_cutout(source_path, payload["x"], payload["y"], payload["size"])
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Gaussian fit completed.",
                    **fit,
                },
            )

        if path == "/api/mask/save":
            try:
                mask_result = save_mask_fits(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Saved mask: {Path(mask_result['fits_path']).name}",
                    **mask_result,
                },
            )

        if path == "/api/mask/load":
            try:
                mask_result = load_mask_fits(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=404)
            if mask_result.get("fits_path"):
                message = f"Loaded mask: {Path(mask_result['fits_path']).name}"
            else:
                message = f"Initialized {mask_result['mask_type']} from project image."
            return json_response(
                self,
                {
                    "message": message,
                    **mask_result,
                },
            )

        if path == "/api/mask/preview":
            try:
                preview_result = mask_preview_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, preview_result)

        if path == "/api/mask/conjugate-point":
            try:
                source_path = payload.get("path")
                if not source_path:
                    raise ValueError("path is required.")
                method = str(payload.get("method") or "brightest").strip().lower()
                if method == "click":
                    fit = clicked_conjugate_point(source_path, payload["x"], payload["y"])
                    message = "Selected clicked pixel."
                elif method == "gaussian":
                    fit = gaussian_conjugate_point(
                        source_path,
                        payload["x"],
                        payload["y"],
                        payload.get("size") or 9,
                    )
                    message = "Selected Gaussian-fit center."
                else:
                    fit = brightest_conjugate_pixel(
                        source_path,
                        payload["x"],
                        payload["y"],
                    )
                    message = "Selected brightest pixel in the clicked 3x3 region."
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": message,
                    **fit,
                },
            )

        if path == "/api/mask/lens-light-subtraction-script":
            try:
                result = generate_lens_light_subtraction_script(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/mask/lens-light-subtraction-run":
            try:
                result = start_lens_light_subtraction_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/mask/lens-light-subtraction-run-gpu":
            try:
                result = start_sciama_lens_light_subtraction_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/mask/lens-light-subtraction-stop":
            try:
                result = stop_lens_light_subtraction_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/psf-fit/input-preview":
            try:
                source_path = Path(payload.get("psf_path") or payload.get("path")).expanduser().resolve()
                preview = psf_preview_from_path(source_path, payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Loaded input PSF: {source_path.name}",
                    **preview,
                },
            )

        if path == "/api/psf-fit/default-euclid":
            try:
                preview = load_default_euclid_psf(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Loaded default Euclid PSF.",
                    **preview,
                },
            )

        if path == "/api/psf-fit/default-euclid-h":
            try:
                preview = load_default_euclid_h_psf(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Loaded default Euclid H-band PSF.",
                    **preview,
                },
            )

        if path in ("/api/psf-fit/default-euclid-y", "/api/psf-fit/default-euclid-j"):
            band = "Y" if path.endswith("-y") else "J"
            try:
                preview = load_default_euclid_nisp_psf(payload, band)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Loaded default Euclid {band}-band PSF.",
                    **preview,
                },
            )

        if path == "/api/psf-fit/auto-stars":
            try:
                source_path = Path(payload.get("image_path") or payload.get("path")).expanduser().resolve()
                result = find_psf_stars(
                    source_path,
                    payload.get("kernel_size", 31),
                    threshold_sigma=payload.get("threshold_sigma", 5.0),
                    fwhm=payload.get("fwhm", 1.6),
                    payload=payload,
                )
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Found {len(result['stars'])} PSF candidates.",
                    **result,
                },
            )

        if path == "/api/psf-fit/clear-output":
            try:
                result = clear_psf_outputs(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/psf-fit/stop":
            try:
                result = stop_psf_fit_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/psf-fit/run":
            try:
                mode = payload.get("mode", "input")
                if mode == "auto":
                    result = start_psf_fit_job(payload)
                    message = f"Started PSF fit job {result['job_id']}."
                else:
                    source_path = Path(payload.get("psf_path") or payload.get("psf_input")).expanduser().resolve()
                    result = fit_psf_from_input(source_path, payload)
                    message = "Fitted PSF from input kernel."
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": message,
                    **result,
                },
            )

        if path == "/api/lens-model/generate-code":
            try:
                payload, artifact_manifest = standardize_project_artifacts(payload)
                project_dir = Path(payload["project_folder"]).expanduser()
                preview_dir = project_dir / LENS_MODEL_RESULT_DIRNAME
                preview_dir.mkdir(parents=True, exist_ok=True)
                generated_code, model_config = generate_lens_script(
                    payload | {"output_dir": str(preview_dir)},
                    job_dir=preview_dir,
                    default_data_dir=DEFAULT_DATA_DIR,
                )
                script_path = project_dir / LENS_MODEL_SCRIPT_NAME
                script_path.write_text(generated_code, encoding="utf-8")
                hmc_script_path = None
                hmc_generated_code = ""
                if isinstance(model_config, dict) and (
                    model_config.get("mode") == "double_source_plane" or model_config.get("dspl_enabled")
                ):
                    hmc_generated_code, _ = generate_lens_hmc_script(
                        payload | {"output_dir": str(preview_dir)},
                        job_dir=preview_dir,
                        default_data_dir=DEFAULT_DATA_DIR,
                    )
                    hmc_script_path = project_dir / LENS_HMC_SCRIPT_NAME
                    hmc_script_path.write_text(hmc_generated_code, encoding="utf-8")
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": "Generated lens model code.",
                    "generated_code": generated_code,
                    "script_path": str(script_path),
                    "hmc_script_path": str(hmc_script_path) if hmc_script_path else "",
                    "hmc_generated": bool(hmc_script_path),
                    "model_config": model_config,
                    "project_artifacts": artifact_manifest,
                },
            )

        if path == "/api/lens-model/cutout-overlay":
            try:
                result = project_cutout_overlay_payload(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/lens-model/run":
            try:
                result = start_lens_svi_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/lens-model/run-gpu":
            try:
                result = start_sciama_lens_svi_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/lens-model/run-all-gpu":
            try:
                result = start_sciama_lens_svi_job(payload, use_all_gpu_candidates=True)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        if path == "/api/lens-model/stop":
            try:
                result = stop_lens_svi_job(payload)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, result)

        return json_response(self, {"detail": "Unknown API endpoint"}, status=404)

    def serve_static(self, path):
        request_path = "/web_gui/index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        candidate = (GUI_ROOT / request_path).resolve()
        try:
            candidate.relative_to(GUI_ROOT)
        except ValueError:
            return self.send_error(403)

        if candidate.is_dir():
            candidate = candidate / "index.html"

        if not candidate.exists() or not candidate.is_file():
            return self.send_error(404)

        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_workspace_file(self, file_path):
        candidate = Path(file_path).expanduser().resolve()
        try:
            candidate.relative_to(WORKSPACE_ROOT)
        except ValueError:
            return self.send_error(403)
        if not candidate.exists() or not candidate.is_file():
            return self.send_error(404)
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Serve the Herculens HTML GUI and lightweight API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), HerculensGuiHandler)
    print(f"Serving Herculens GUI at http://{args.host}:{args.port}/web_gui/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
