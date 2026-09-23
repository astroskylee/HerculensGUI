# AGENTS.md - HerculensGUI Quick Guide

For the generated lens-model script's annotated execution flow, read
`LENS_MODELING_SCRIPT_GUIDE.md` before changing SVI/HMC generation or mask logic.

This file is for AI agents working inside `D:\lensing\Herculens\HerculensGUI`.
Parent workspace rules in `D:\lensing\AGENTS.md` still apply.

## What This Project Is

HerculensGUI is a local web GUI for a lens-modeling workflow:

1. download or load a science image,
2. choose a cutout,
3. fit or load a PSF,
4. draw masks and optional conjugate points,
5. generate and run a Herculens/NumPyro SVI script,
6. inspect result previews and saved products.

It is not a packaged web app. It is a small static frontend plus a Python
standard-library HTTP server, with scientific jobs launched as subprocesses.

## Quick Run

From this folder:

```powershell
/home/skylee/anaconda3/envs/herculens/bin/python web_gui/backend_server.py --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/web_gui/
```

The backend server itself should be launched with the `herculens` conda
environment. Do not use the system `python3` for backend restarts; it can miss
scientific dependencies such as `matplotlib` and break project preview loading.

The GUI depends on configured Python runtimes for heavy work. Runtime settings
are saved in `web_gui_settings.local.json`; environment overrides are:

- `HERCULENS_PYTHON`
- `EUCLID_PYTHON`
- `PHOSPHOROS_PYTHON`
- `PHOSPHOROS_ROOT`

Do not launch full SVI, Euclid downloads, Phosphoros runs, or other expensive
jobs unless the user explicitly asks for that run.

## Main Files

- `web_gui/index.html` - the whole DOM for the five workflow pages: Euclid cutout, image preprocess, PSF fit, mask, lens model.
- `web_gui/app.js` - all client-side state, canvas rendering, project restore/save, API calls, polling, and event handlers.
- `web_gui/styles.css` - layout, light/dark theme, responsive behavior, preview panels, mask/PSF/lens UI styling.
- `web_gui/backend_server.py` - static file server and JSON API. This is the backend hub.
- `web_gui/lens_code_generator.py` - converts GUI/project payloads into generated `run_lens_svi.py` scripts.
- `semilinear_solver.py` - independent fixed-model source inversion used after single-plane pixelated SVI.
- `web_gui/psf_svi_runner.py` - background PSF SVI runner used by auto PSF fitting.
- `web_gui/psf_joint_nonlinear_fit.py` - deterministic/linear least-squares PSF experiment; not the main backend default.
- `web_gui/euclid_cutout_runner.py` - wrapper that calls workspace Euclid helper scripts.
- `web_gui/defaults/Euclid_default_PSF.fits` - tracked default Euclid PSF.
- `Tian_infra.py` - local scientific helper layer used by generated scripts; key classes include `Plot`, `Geometry`, `Mass`, `PowerSpectrum`, `Light`, `ModelHelper`, `SVI`, and `GUIStatus`.
- `runs/` - generated project folders and job outputs. It is ignored by git.
- `SliceLens_1chain_parametric_pixelated_svi.ipynb` - exploratory/reference notebook, not the GUI entrypoint.

## Data And Project Layout

Projects live under:

```text
runs/<project_id>/
```

Important files in a project folder:

- `project.json` - saved GUI state.
- `project_artifacts.json` - manifest of standardized files.
- `Data_cutout.fits` - standard lens-model input image.
- `RMS_map.fits` - optional/standard RMS map.
- `PSF_model.fits` - standard PSF used by lens modeling.
- `mask_1.fits` - source-plane 1 arc mask.
- `mask_2.fits` - source-plane 2 arc mask for DSPL.
- `mask_out.fits` - pixels excluded from fitting.
- `run_lens_svi.py` - generated runnable lens-model script.
- `Tian_infra.py` - copied into the project folder so the generated script can run locally.
- `semilinear_solver.py` - copied into the project folder with `Tian_infra.py` for local and Sciama runs.
- `lens_model_result/` - lens SVI `status.json`, `run.log`, previews, summaries, losses, pickles, and FITS panels.
- `lens_light_subtraction_result/` - optional lens-light subtraction output.
- `psf/jobs/psf_fit_<job_id>_<image>/` - auto PSF job config/status/log/products.
- `cutouts/`, `masks/`, `previews/`, `data/` - intermediate GUI products.

`runs/.latest_project.json` stores the currently selected/restorable project.

## Workflow Data Flow

1. Project management starts from `/api/project/latest`, `/api/projects`, `/api/project/new`, `/api/project/save`, and `/api/project/load`.
2. Euclid cutout starts a background job, writes under the active project, and can register RMS/PSF bundles.
3. Image preprocess loads a FITS/NPY/image path or upload, makes PNG previews, and saves a chosen cutout as `Data_cutout.fits`.
4. PSF can be loaded directly, taken from the default Euclid PSF, or fitted from selected stars. The standard result is copied to `PSF_model.fits`.
5. Mask editing saves native-orientation FITS masks into the project root.
6. Lens-light subtraction is optional and writes its own generated script/output; it does not replace `Data_cutout.fits`.
7. Lens model generation calls `standardize_project_artifacts()`, then `generate_lens_script()`, writes `run_lens_svi.py`, and launches it with the configured Herculens Python.
8. A single-plane run selects the best pixelated SVI chain, solves a gradient-regularized semilinear source with fixed mass and lens light, and saves both the source grid and a second model preview.
9. The frontend polls job endpoints and renders `status.json`, previews, chain selectors, and mass summaries.

## Backend API Map

Core GET endpoints:

- `/api/health`
- `/api/files?dir=...`
- `/api/settings`
- `/api/project/latest`
- `/api/projects`
- `/api/example-code/latest`
- `/api/image-preprocess/latest`
- `/api/psf-fit/latest`
- `/api/lens-model/latest?project_id=...`
- `/api/euclid-cutout/job/<job_id>`
- `/api/psf-fit/job/<job_id>`
- `/api/lens-model/job/<job_id>`
- `/api/mask/lens-light-subtraction-job/<job_id>`

Core POST endpoints:

- `/api/settings/save`, `/api/settings/check`
- `/api/project/new`, `/api/project/save`, `/api/project/load`, `/api/project/rename`, `/api/project/delete`, `/api/project/thumbnail`
- `/api/euclid-cutout/run`, `/api/euclid-cutout/load-fits`
- `/api/legacy-cutout/run`
- `/api/photoz/measure`
- `/api/fundamental-plane/run`
- `/api/image-preprocess/run`, `/api/image-preprocess/cutout`, `/api/image-preprocess/gaussian-fit`
- `/api/psf-fit/input-preview`, `/api/psf-fit/default-euclid`, `/api/psf-fit/auto-stars`, `/api/psf-fit/run`, `/api/psf-fit/stop`, `/api/psf-fit/clear-output`
- `/api/mask/save`, `/api/mask/load`, `/api/mask/conjugate-point`, `/api/mask/lens-light-subtraction-run`
- `/api/lens-model/generate-code`, `/api/lens-model/cutout-overlay`, `/api/lens-model/run`, `/api/lens-model/stop`

When adding an endpoint, keep the pattern simple: route in `handle_api_get()` or
`handle_api_post()`, return through `json_response()`, and keep long work in a
subprocess with `config.json`, `status.json`, `run.log`, and `pid.txt`.

## Frontend Mental Model

`app.js` is intentionally one file. Important state objects near the top:

- `projectState` - current project id/name/folder and save timers.
- `euclidState` - current Euclid/DESI image payload and photometry state.
- `imageView` - loaded preprocess image, zoom/pan, cutout center, background box.
- `cutoutState` - saved cutout paths and bounds.
- `psfState` - PSF mode, selected star IDs, preview/job status.
- `maskState` - mask pixels, tool/mode/type, conjugate points, lens-light job state.
- `lensState` - generated code, model config, lens job, chain previews.

HTML ids in `index.html` must stay in sync with `document.querySelector(...)`
calls in `app.js`. There is no build step or frontend framework.

## Path Notes

This project is usually edited from Windows paths like `D:\lensing\...`, but
some backend defaults still reference WSL/Linux paths such as `/mnt/d/lensing`,
`/home/skylee/...`, and workspace helper scripts under
`/mnt/d/lensing/codex_agent/user_script/`. Be careful when changing path logic:
support both absolute project paths and GUI-relative paths where possible.

## Safety Notes

- `runs/` contains generated science products and is git-ignored; avoid editing result files in place unless requested.
- Project deletion uses a recycle destination under `/mnt/d/lensing/RecycleBin`, but some cleanup helpers remove generated PSF outputs inside `runs/`; do not broaden delete scopes.
- Euclid credentials, if entered, are written only into a job-local `credentials.txt`; never log, commit, or copy credentials.
- Keep `HDF5_USE_FILE_LOCKING=FALSE` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` early in runner scripts.
- Generated scripts are meant to run from the project folder. If you change `Tian_infra.py` or generated-code assumptions, test with a tiny/smoke path before any real SVI run.

## Good First Checks

For light validation after documentation or syntax-only edits:

```powershell
python -m py_compile .\web_gui\backend_server.py .\web_gui\lens_code_generator.py .\web_gui\euclid_cutout_runner.py
```

For UI/API behavior, start the server and use `/api/health`. Do not start heavy
modeling jobs as a casual smoke test.
