"""
STEP 1 - Run the frozen pipeline on every CT volume in a folder.

Example (MedSeg COVID-19, 9 volumes):
    python scripts/01_run_segmentation.py --images data/medseg/rp_im --out outputs/medseg

Freeze first:
    python scripts/01_run_segmentation.py --write-config frozen_config.json
    (edit nothing after you have looked at MedSeg results)
    python scripts/01_run_segmentation.py --images ... --out ... --config frozen_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lungseg.config import Config          # noqa: E402
from lungseg.pipeline import run_case      # noqa: E402


def case_id(p: Path) -> str:
    return p.name.replace(".nii.gz", "").replace(".nii", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pattern", default="*.nii*")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--write-config", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="case ids to run")
    ap.add_argument("--final-only", action="store_true", help="do not save intermediate stages")
    args = ap.parse_args()

    if args.write_config:
        Config().to_json(args.write_config)
        print(f"wrote {args.write_config}")
        return

    cfg = Config.from_json(args.config) if args.config else Config()
    files = sorted(args.images.glob(args.pattern))
    if args.only:
        files = [f for f in files if case_id(f) in set(args.only)]
    if not files:
        sys.exit(f"no images matching {args.pattern} in {args.images}")

    for f in files:
        cid = case_id(f)
        print(f"[{cid}] running ...", flush=True)
        meta = run_case(f, args.out / cid, cfg, save_stages=not args.final_only)
        print(f"[{cid}] done in {meta['timing_s']['total_s']:.1f} s  "
              f"{json.dumps({k: round(v, 1) for k, v in meta['timing_s'].items()})}")


if __name__ == "__main__":
    main()
