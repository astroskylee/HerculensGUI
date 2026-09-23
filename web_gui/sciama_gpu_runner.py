import json
import os
import pty
import re
import select
import shlex
import shutil
import sys
import time
from pathlib import Path


TERMINAL_STATES = {"completed", "failed", "stopped"}
LOCAL_SCIAMA_SETTINGS = Path.home() / ".config" / "herculensgui" / "sciama.json"


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_log(log_path, text):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8", errors="ignore") as handle:
        handle.write(text)
        handle.flush()


def python_started_in_attempt(attempt_log):
    """Return True only when the remote shell printed an allocated node."""
    for line in attempt_log.replace("\r", "").splitlines():
        stripped = line.strip()
        if stripped.startswith("SCIAMA_GPU_NODE=") and "$(" not in stripped:
            return True
    return False


def update_status(config, state, message, *, fraction=None, extra=None):
    runner_label = str(config.get("runner_label") or "sciama_gpu")
    progress = {"message": message}
    if fraction is not None:
        progress["fraction"] = fraction
    status = {
        "job_id": config["job_id"],
        "state": state,
        "message": message,
        "job_dir": config["job_dir"],
        "config_path": config["config_path"],
        "log_path": config["log_path"],
        "script_path": config["script_path"],
        "runner": runner_label,
        "runner_type": "sciama_gpu",
        "remote_dir": config["remote_dir"],
        "gpu_candidate_mode": config.get("gpu_candidate_mode", "default"),
        "progress": progress,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        status.update(extra)
    atomic_write_json(config["status_path"], status)


def safe_name(value):
    text = str(value or "project").strip() or "project"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]


def load_passphrase(config):
    configured_path = config.get("passphrase_file") or os.getenv("HGUI_SCIAMA_PASSPHRASE_FILE")
    if not configured_path and LOCAL_SCIAMA_SETTINGS.exists():
        local_settings = json.loads(LOCAL_SCIAMA_SETTINGS.read_text(encoding="utf-8"))
        configured_path = local_settings.get("passphrase_file")
    if not configured_path:
        return None
    passphrase_file = Path(configured_path).expanduser()
    if not passphrase_file.exists():
        return None
    return passphrase_file.read_text(encoding="utf-8").strip()


def format_command(argv):
    return " ".join(shlex.quote(str(item)) for item in argv)


def run_pty(argv, log_path, *, passphrase=None, cwd=None):
    append_log(log_path, f"\n$ {format_command(argv)}\n")
    pid, fd = pty.fork()
    if pid == 0:
        if cwd:
            os.chdir(cwd)
        os.execvp(str(argv[0]), [str(item) for item in argv])

    prompt_buffer = ""
    returncode = None
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if fd in ready:
                try:
                    chunk = os.read(fd, 8192)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="ignore")
                append_log(log_path, text)
                prompt_buffer = (prompt_buffer + text)[-2000:]
                lower = prompt_buffer.lower()
                if "are you sure you want to continue connecting" in lower:
                    os.write(fd, b"yes\n")
                    prompt_buffer = ""
                elif passphrase and ("passphrase" in lower or "password:" in lower):
                    os.write(fd, (passphrase + "\n").encode("utf-8"))
                    prompt_buffer = ""

            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                if os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    returncode = 128 + os.WTERMSIG(status)
                break
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if returncode is None:
        _, status = os.waitpid(pid, 0)
        returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
    append_log(log_path, f"\n[exit {returncode}] {format_command(argv)}\n")
    return returncode


def require_success(returncode, label):
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")


def gpu_candidates(config):
    candidates = config.get("gpu_candidates")
    cleaned = []
    seen = set()
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                gres = str(item.get("gres") or "").strip()
                partition = str(item.get("partition") or config.get("partition") or "gpu.q").strip() or "gpu.q"
                label = str(item.get("label") or (f"{gres} on {partition}" if gres else f"default GPU request on {partition}")).strip()
            else:
                gres = str(item or "").strip()
                partition = str(config.get("partition") or "gpu.q").strip() or "gpu.q"
                label = f"{gres} on {partition}" if gres else f"default GPU request on {partition}"
            key = (partition, gres)
            if key not in seen:
                cleaned.append({"partition": partition, "gres": gres, "label": label})
                seen.add(key)
    if not cleaned:
        legacy_candidates = config.get("gres_candidates")
        if not isinstance(legacy_candidates, list) or not legacy_candidates:
            legacy_candidates = [config.get("gres")]
        for item in legacy_candidates:
            gres = str(item or "").strip()
            partition = str(config.get("partition") or "gpu.q").strip() or "gpu.q"
            key = (partition, gres)
            if gres and key not in seen:
                cleaned.append({"partition": partition, "gres": gres, "label": f"{gres} on {partition}"})
                seen.add(key)
    if not cleaned:
        cleaned.append({"partition": "gpu.q", "gres": "gpu:A100:1", "label": "A100 full GPU on gpu.q"})
    return cleaned


def copy_file_for_staging(source, target):
    """Copy a WSL project file, falling back when DrvFS sendfile is unavailable."""
    source = Path(source)
    target = Path(target)
    try:
        return Path(shutil.copy2(source, target))
    except OSError as copy_error:
        try:
            with source.open("rb") as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        except Exception as fallback_error:
            raise copy_error from fallback_error
        return target


def stage_slim_project(config):
    project_dir = Path(config["project_dir"]).expanduser()
    stage_dir = Path(config["stage_dir"]).expanduser()
    stage_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for name in config.get("slim_files") or []:
        source = project_dir / name
        if not source.exists():
            missing.append(name)
            continue
        target = stage_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                copy_function=copy_file_for_staging,
            )
        else:
            copy_file_for_staging(source, target)
        copied.append(name)
    if not copied:
        raise FileNotFoundError("No slim project files were available to upload.")
    return copied, missing


def merge_final_status(config, state, message, *, returncode=None):
    runner_label = str(config.get("runner_label") or "sciama_gpu")
    status_path = Path(config["status_path"])
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {}
    if status.get("state") not in TERMINAL_STATES:
        status["state"] = state
    status.setdefault("message", message)
    status.update(
        {
            "job_id": config["job_id"],
            "job_dir": config["job_dir"],
            "config_path": config["config_path"],
            "log_path": config["log_path"],
            "script_path": config["script_path"],
            "runner": runner_label,
            "runner_type": "sciama_gpu",
            "remote_dir": config["remote_dir"],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if returncode is not None:
        status["remote_returncode"] = returncode
    if status.get("state") == "completed":
        status["message"] = status.get("message") or str(config.get("completed_message") or "Sciama GPU job completed.")
    elif state == "failed":
        status["state"] = "failed"
        status["message"] = message
    atomic_write_json(status_path, status)


def main():
    if os.name != "posix":
        raise RuntimeError("Sciama GPU runner must be started from WSL/Linux because it uses SSH through a pseudo-terminal.")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: sciama_gpu_runner.py <config.json>")

    config_path = Path(sys.argv[1]).expanduser()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    log_path = Path(config["log_path"])
    passphrase = load_passphrase(config)

    try:
        job_label = str(config.get("job_label") or "Lens SVI")
        remote_script_name = str(config.get("remote_script_name") or "run_lens_svi.py")
        result_dir_name = str(config.get("result_dir_name") or "lens_model_result").strip("/ ")
        hmc_netcdf_root = str(
            config.get("hmc_netcdf_root") or "/mnt/lustre/tianli/HerculensGUI_hmc"
        )
        completed_message = str(config.get("completed_message") or f"Sciama GPU {job_label} completed.")
        running_message = str(config.get("running_message") or f"Running {job_label} on Sciama GPU.")
        downloading_message = str(config.get("downloading_message") or f"Downloading Sciama {job_label} results.")
        append_log(log_path, f"SCIAMA_RUNNER_START job_id={config['job_id']}\n")
        update_status(config, "uploading", "Staging slim project for Sciama upload.", fraction=0.02)
        copied, missing = stage_slim_project(config)
        append_log(log_path, f"SCIAMA_STAGE upload copied={','.join(copied)} missing={','.join(missing)}\n")

        remote_dir = config["remote_dir"]
        mkdir_cmd = f"mkdir -p {shlex.quote(remote_dir)}"
        require_success(run_pty(["ssh", "sciama2", mkdir_cmd], log_path, passphrase=passphrase), "Remote directory creation")

        stage_dir = Path(config["stage_dir"]).expanduser()
        upload_files = [f"{stage_dir}/."]
        update_status(config, "uploading", "Uploading slim project to Sciama.", fraction=0.08)
        require_success(
            run_pty(["scp", "-rp", *upload_files, f"sciama2:{remote_dir}/"], log_path, passphrase=passphrase),
            "Slim project upload",
        )

        update_status(config, "running", running_message, fraction=0.12)
        append_log(log_path, f"SCIAMA_REMOTE_DIR={remote_dir}\n")
        append_log(log_path, "SCIAMA_STAGE run\n")
        remote_returncode = 1
        chosen_candidate = None
        python_started_candidate = None
        candidates = gpu_candidates(config)
        candidate_labels = [str(candidate.get("label") or f"{candidate['gres']} on {candidate['partition']}") for candidate in candidates]
        append_log(log_path, f"SCIAMA_GPU_CANDIDATES={','.join(candidate_labels)}\n")
        for index, candidate in enumerate(candidates, start=1):
            partition = str(candidate["partition"])
            gres = str(candidate["gres"])
            label = str(candidate.get("label") or f"{gres} on {partition}")
            append_log(log_path, f"SCIAMA_GPU_ATTEMPT {index}/{len(candidates)} partition={partition} gres={gres} label={label}\n")
            update_status(
                config,
                "running",
                f"{running_message} ({label}).",
                fraction=0.12,
                extra={"partition": partition, "gres": gres, "gpu_candidates": candidates},
            )
            inner = (
                f"set -euo pipefail; cd {shlex.quote(remote_dir)}; "
                f"source {shlex.quote(config['conda_activate'])}; "
                "export HDF5_USE_FILE_LOCKING=FALSE; "
                "export XLA_PYTHON_CLIENT_PREALLOCATE=false; "
                f"export HERCULENS_HMC_NETCDF_ROOT={shlex.quote(hmc_netcdf_root)}; "
                f"export HERCULENS_GPU_LABEL={shlex.quote(label)}; "
                f"export HERCULENS_GPU_GRES={shlex.quote(gres)}; "
                f"export HERCULENS_GPU_PARTITION={shlex.quote(partition)}; "
                "echo SCIAMA_GPU_NODE=$(hostname); "
                f"echo SCIAMA_GPU_PARTITION={shlex.quote(partition)}; "
                f"echo SCIAMA_GPU_GRES={shlex.quote(gres)}; "
                f"echo SCIAMA_GPU_LABEL={shlex.quote(label)}; "
                f"python -u {shlex.quote(remote_script_name)}"
            )
            remote_run = (
                f"cd {shlex.quote(remote_dir)} && "
                "export HDF5_USE_FILE_LOCKING=FALSE && "
                "export XLA_PYTHON_CLIENT_PREALLOCATE=false && "
                f"srun --immediate=60 --pty "
                f"{f'--gres={shlex.quote(gres)} ' if gres else ''}"
                f"-J {shlex.quote(config['slurm_job_name'])} "
                f"-p {shlex.quote(partition)} /bin/bash -lc {shlex.quote(inner)}"
            )
            before_size = Path(log_path).stat().st_size if Path(log_path).exists() else 0
            remote_returncode = run_pty(["ssh", "sciama2", remote_run], log_path, passphrase=passphrase)
            try:
                with Path(log_path).open("r", encoding="utf-8", errors="ignore") as handle:
                    handle.seek(before_size)
                    attempt_log = handle.read()
            except Exception:
                attempt_log = ""
            if remote_returncode == 0:
                chosen_candidate = candidate
                append_log(log_path, f"SCIAMA_GPU_SELECTED partition={partition} gres={gres} label={label}\n")
                break
            append_log(log_path, f"SCIAMA_GPU_FAILED partition={partition} gres={gres} returncode={remote_returncode}\n")
            if python_started_in_attempt(attempt_log):
                python_started_candidate = candidate
                append_log(log_path, "SCIAMA_GPU_NO_FALLBACK reason=python_started\n")
                break

        update_status(config, "downloading", downloading_message, fraction=0.95)
        append_log(log_path, "SCIAMA_STAGE download\n")
        Path(config["job_dir"]).mkdir(parents=True, exist_ok=True)
        download_code = run_pty(
            ["scp", "-rp", f"sciama2:{remote_dir}/{result_dir_name}/.", config["job_dir"]],
            log_path,
            passphrase=passphrase,
        )
        if remote_returncode != 0:
            if python_started_candidate:
                label = str(python_started_candidate.get("label") or "selected GPU")
                script_failure = (
                    "run_lens_svi.py failed"
                    if remote_script_name == "run_lens_svi.py"
                    else f"{remote_script_name} failed"
                )
                raise RuntimeError(
                    f"Sciama GPU was allocated ({label}), but {script_failure} after Python started; "
                    f"exit code {remote_returncode}. Check run.log for the model traceback."
                )
            raise RuntimeError(
                f"Sciama GPU resources were not allocated for any candidate ({', '.join(candidate_labels)}); "
                f"last exit code {remote_returncode}"
            )
        require_success(download_code, f"{job_label} result download")
        if chosen_candidate:
            update_status(
                config,
                "completed",
                completed_message,
                fraction=1.0,
                extra={
                    "partition": chosen_candidate["partition"],
                    "gres": chosen_candidate["gres"],
                    "gpu_candidates": candidates,
                },
            )
        merge_final_status(config, "completed", completed_message, returncode=remote_returncode)
        append_log(log_path, "SCIAMA_RUNNER_DONE\n")
    except Exception as exc:
        message = f"Sciama GPU job failed: {exc}"
        append_log(log_path, f"\nSCIAMA_RUNNER_FAILED {message}\n")
        update_status(config, "failed", message, fraction=1.0)
        raise


if __name__ == "__main__":
    main()
