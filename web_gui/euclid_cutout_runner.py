from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download Euclid cutouts plus VIS RMS map.")
    parser.add_argument("--single-script", required=True)
    parser.add_argument("--bundle-script", required=True)
    parser.add_argument("--target-ra", required=True)
    parser.add_argument("--target-dec", required=True)
    parser.add_argument("--credentials-file", default="")
    parser.add_argument("--radius-arcsec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--env", default="IDR")
    return parser.parse_args()


def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    bundle_dir = output_root / "vis_rms_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    single_command = [
        sys.executable,
        args.single_script,
        "--target-ra",
        args.target_ra,
        "--target-dec",
        args.target_dec,
        "--env",
        args.env,
        "--radius-arcsec",
        args.radius_arcsec,
        "--output-root",
        str(output_root),
        "--no-show",
    ]
    if args.credentials_file:
        single_command.extend(["--credentials-file", args.credentials_file])
    run_command(single_command)

    bundle_command = [
        sys.executable,
        args.bundle_script,
        "--ra",
        args.target_ra,
        "--dec",
        args.target_dec,
        "--env",
        args.env,
        "--radius-arcsec",
        args.radius_arcsec,
        "--output-dir",
        str(bundle_dir),
    ]
    if args.credentials_file:
        bundle_command.extend(["--credentials-file", args.credentials_file])
    run_command(bundle_command)


if __name__ == "__main__":
    main()
