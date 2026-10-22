"""Command-line entry point.
    python -m executor --config configs/nicu.yaml
    python -m executor --config configs/nicu.yaml --set training.epochs.64=20 --set runtime.seed=1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from executor.config import ConfigError, load_config
from utils.log import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mocostorm", description="MoCo-SToRM motion-compensated reconstruction")
    p.add_argument("--config", "-c", required=True, type=Path, help="YAML config (or legacy JSON)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value; may be repeated")
    p.add_argument("--data-dir", help="shortcut for --set data.data_dir=...")
    p.add_argument("--output-dir", help="shortcut for --set data.output_dir=...")
    p.add_argument("--run-name", help="shortcut for --set runtime.run_name=...")
    p.add_argument("--dry-run", action="store_true", help="validate and print the resolved config, then exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    for flag, key in ((args.data_dir, "data.data_dir"), (args.output_dir, "data.output_dir"),
                      (args.run_name, "runtime.run_name")):
        if flag:
            overrides.append(f"{key}={flag}")

    try:
        cfg = load_config(args.config, overrides)
    except (ConfigError, OSError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(cfg.to_yaml())
        return 0

    output_dir = cfg.resolve_output_dir(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "recon.log", cfg.runtime.verbose)
    (output_dir / "config.yaml").write_text(cfg.to_yaml())
    logger.info("Config %s -> %s (SGE job %s)", args.config, output_dir, os.environ.get("JOB_ID", "-"))

    # Imported late so --dry-run and --help work without CUDA/sigpy.
    from executor.pipeline import run_reconstruction

    try:
        run_reconstruction(cfg, output_dir)
    except Exception:
        logger.exception("Reconstruction failed")
        return 1
    logger.info("Finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
