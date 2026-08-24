#!/usr/bin/env python3
"""Copy NGC + W&B keys from Desktop/env.txt into gitignored .env.lambda. Prints no secrets."""

from __future__ import annotations

from pathlib import Path

SRC = Path("/Users/avinashnandyala/Desktop/env.txt")
DST = Path(__file__).resolve().parents[2] / ".env.lambda"


def _value_after_colon(line: str) -> str:
    return line.split(":", 1)[1].strip().strip("'\"")


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Missing {SRC}")

    ngc = wandb = None
    for raw in SRC.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("nvidi") and "api" in low and ":" in line:
            ngc = _value_after_colon(line)
        elif low.startswith("want") and "api" in low and ":" in line:
            wandb = _value_after_colon(line)
        elif low.startswith("wandb") and "api" in low and ":" in line:
            wandb = _value_after_colon(line)

    if not ngc:
        raise SystemExit("NGC/NVIDIA API key label not found in env.txt")
    if not wandb:
        raise SystemExit("W&B API key label not found in env.txt")

    body = (
        "# Generated locally. Do not commit.\n"
        f"NGC_API_KEY={ngc}\n"
        f"WANDB_API_KEY={wandb}\n"
        "WANDB_PROJECT=beamdojo\n"
        "BEAMDOJO_LOG_ROOT=/lambda/nfs/beamdojo/logs\n"
    )
    DST.write_text(body)
    DST.chmod(0o600)
    print(f"Wrote {DST} (NGC + W&B only). File is gitignored.")


if __name__ == "__main__":
    main()
