import argparse
import cgi
import json
import mimetypes
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


GUI_ROOT = Path(__file__).resolve().parents[1]
SERVER_CWD = Path.cwd().resolve()
WEB_ROOT = Path(__file__).resolve().parent
LATEST_RUN_DIR = GUI_ROOT / "runs" / "slicelens_svi_2k"
LATEST_LENS_FIGURE = LATEST_RUN_DIR / "20260425_144805_four_panel_comparison.png"
UPLOAD_DIR = GUI_ROOT / "runs" / "web_gui_uploads"
PREVIEW_DIR = GUI_ROOT / "runs" / "web_gui_previews"
HEADER_DIR = GUI_ROOT / "runs" / "web_gui_headers"
CUTOUT_DIR = GUI_ROOT / "runs" / "web_gui_cutouts"
IMAGE_EXTENSIONS = {".fits", ".fit", ".fts", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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


def relative_url(path):
    return "/" + path.resolve().relative_to(GUI_ROOT).as_posix()


def safe_filename(name):
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)
    return clean or "uploaded_image"


def save_uploaded_file(handler):
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
        },
    )
    field = form["image_file"] if "image_file" in form else None
    if field is None or not getattr(field, "filename", ""):
        raise ValueError("No image file was uploaded.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(field.filename)}"
    out_path = UPLOAD_DIR / suffix
    with out_path.open("wb") as out:
        shutil.copyfileobj(field.file, out)
    return out_path


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


def save_header_metadata(upload_path, header):
    if header is None:
        return None
    HEADER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HEADER_DIR / f"{upload_path.stem}_header.json"
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


def save_preview_png(upload_path):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    raw_data, header = load_science_image(upload_path)
    save_header_metadata(upload_path, header)
    data = np.asarray(raw_data, dtype=float)
    data = np.squeeze(data)
    is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    if data.ndim > 2 and not is_rgb:
        data = data[0]
    if data.ndim != 2 and not is_rgb:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

    display_data = data[..., :3] if is_rgb else data
    finite = np.isfinite(display_data)
    if not np.any(finite):
        raise ValueError("Image contains no finite values.")

    values = display_data[finite]
    vmin, vmax = np.nanpercentile(values, [0.5, 99.7])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / f"{upload_path.stem}_preview.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    if is_rgb:
        rgb = np.clip((display_data - vmin) / max(vmax - vmin, np.finfo(float).eps), 0.0, 1.0)
        ax.imshow(rgb, origin="lower")
    else:
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path, display_data.shape


def preview_existing_path(path):
    preview_path, image_shape = save_preview_png(path)
    return relative_url(preview_path), image_shape


def save_cutout(path, x_display, y_display, side_length):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

    raw_data, header = load_science_image(path)
    data = np.asarray(raw_data)
    data = np.squeeze(data)
    is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    if data.ndim > 2 and not is_rgb:
        data = data[0]
    if data.ndim != 2 and not is_rgb:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

    height, width = data.shape[:2]
    x_center = int(round(float(x_display)))
    y_display = int(round(float(y_display)))
    y_center = height - 1 - y_display
    size = max(1, int(round(float(side_length))))
    half = size // 2

    x0 = max(0, x_center - half)
    x1 = min(width, x0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y_center - half)
    y1 = min(height, y0 + size)
    y0 = max(0, y1 - size)

    cutout = data[y0:y1, x0:x1, ...] if is_rgb else data[y0:y1, x0:x1]
    if cutout.size == 0:
        raise ValueError("Cutout is empty. Choose a center inside the image.")

    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(path.stem)}_x{x_center}_y{y_center}_s{size}"
    fits_path = CUTOUT_DIR / f"{tag}.fits"
    png_path = CUTOUT_DIR / f"{tag}.png"

    out_header = header.copy() if header is not None else fits.Header()
    out_header["CUTX"] = (x_center, "Cutout center x pixel")
    out_header["CUTY"] = (y_center, "Cutout center y pixel")
    out_header["CUTSIZE"] = (size, "Requested cutout side length")
    out_header["CUTX0"] = (x0, "Cutout x start")
    out_header["CUTY0"] = (y0, "Cutout y start")
    fits.writeto(fits_path, cutout, header=out_header, overwrite=False)

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
        ax.imshow(display_data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "fits_path": str(fits_path),
        "preview_url": relative_url(png_path),
        "bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "center": {"x": x_center, "y": y_center},
        "shape": list(cutout.shape),
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
            return json_response(self, {"status": "Ready", "gui_root": str(GUI_ROOT), "cwd": str(SERVER_CWD)})

        if path == "/api/files":
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            folder = query.get("dir", [str(SERVER_CWD)])[0] or str(SERVER_CWD)
            try:
                payload = list_folder(Path(folder))
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(self, payload)

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

        if path == "/api/lens-model/latest":
            if LATEST_LENS_FIGURE.exists():
                return json_response(
                    self,
                    {
                        "message": "Loaded latest lens model visualization.",
                        "preview_url": relative_url(LATEST_LENS_FIGURE),
                    },
                )
            return json_response(self, {"message": "No lens model figure found."}, status=404)

        return json_response(self, {"detail": "Unknown API endpoint"}, status=404)

    def handle_api_post(self, path):
        content_type = self.headers.get("Content-Type", "")
        if path == "/api/image-preprocess/run" and content_type.startswith("multipart/form-data"):
            try:
                upload_path = save_uploaded_file(self)
                preview_path, image_shape = save_preview_png(upload_path)
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Loaded image preview: {upload_path.name}",
                    "preview_url": relative_url(preview_path),
                    "uploaded_path": str(upload_path),
                    "image_shape": list(image_shape),
                },
            )

        try:
            payload = read_json_body(self)
        except Exception as exc:
            return json_response(self, {"detail": f"Invalid JSON: {exc}"}, status=400)

        if path == "/api/image-preprocess/run":
            selected_path = payload.get("path")
            if selected_path:
                try:
                    source_path = Path(selected_path).expanduser().resolve()
                    preview_url, image_shape = preview_existing_path(source_path)
                except Exception as exc:
                    return json_response(self, {"detail": str(exc)}, status=400)
                response = {
                    "message": f"Loaded image preview: {source_path.name}",
                    "preview_url": preview_url,
                    "source_path": str(source_path),
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

        if path == "/api/image-preprocess/cutout":
            try:
                source_path = Path(payload["path"]).expanduser().resolve()
                cutout = save_cutout(source_path, payload["x"], payload["y"], payload["size"])
            except Exception as exc:
                return json_response(self, {"detail": str(exc)}, status=400)
            return json_response(
                self,
                {
                    "message": f"Saved cutout: {Path(cutout['fits_path']).name}",
                    **cutout,
                },
            )

        if path == "/api/psf-fit/run":
            return json_response(
                self,
                {
                    "message": "PSF fit request received. Connect this endpoint to the PSF script next.",
                    "payload": payload,
                },
            )

        if path == "/api/lens-model/run":
            response = {
                "message": "Lens model request received. Connect this endpoint to the SVI runner script next.",
                "payload": payload,
            }
            if LATEST_LENS_FIGURE.exists():
                response["preview_url"] = relative_url(LATEST_LENS_FIGURE)
            return json_response(self, response)

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
