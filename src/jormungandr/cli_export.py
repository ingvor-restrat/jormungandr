#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from jormungandr.export import (
    compare_checkpoints,
    export_inference_bundle,
    export_torchscript_from_checkpoint,
    inspect_checkpoint,
    load_obs_json,
)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Jormungandr checkpoint inspection, comparison, and inference export utility."
        )
    )

    # Compatibility mode for the original no-subcommand exporter.
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--module", choices=["auto", "policy", "q", "heads"], default="auto")
    ap.add_argument("--metadata-out", default="")

    sub = ap.add_subparsers(dest="cmd")

    sp_inspect = sub.add_parser("inspect", help="Inspect a checkpoint payload and print model/settings metadata.")
    sp_inspect.add_argument("--checkpoint", required=True)

    sp_bundle = sub.add_parser(
        "export-bundle",
        help="Export a self-describing inference bundle (torchscript + manifest + C++ header).",
    )
    sp_bundle.add_argument("--checkpoint", required=True)
    sp_bundle.add_argument("--bundle-dir", required=True)
    sp_bundle.add_argument("--module", choices=["auto", "policy", "q", "heads"], default="auto")
    sp_bundle.add_argument("--torchscript-name", default="policy.ts.pt")
    sp_bundle.add_argument("--manifest-name", default="manifest.json")
    sp_bundle.add_argument("--header-name", default="jormungandr_model_spec.hpp")

    sp_compare = sub.add_parser(
        "compare",
        help="Compare two checkpoints on the same observation batch.",
    )
    sp_compare.add_argument("--left", required=True, help="Path to left checkpoint")
    sp_compare.add_argument("--right", required=True, help="Path to right checkpoint")
    sp_compare.add_argument("--obs-json", default="", help="Optional JSON file containing obs array or {\"obs\": ...}")
    sp_compare.add_argument("--num-obs", type=int, default=64, help="Synthetic obs rows when --obs-json is omitted")
    sp_compare.add_argument("--seed", type=int, default=7)

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    if args.cmd == "inspect":
        out = inspect_checkpoint(checkpoint_path=args.checkpoint)
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "export-bundle":
        out = export_inference_bundle(
            checkpoint_path=args.checkpoint,
            bundle_dir=args.bundle_dir,
            module=args.module,
            torchscript_name=args.torchscript_name,
            manifest_name=args.manifest_name,
            header_name=args.header_name,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "compare":
        obs = load_obs_json(args.obs_json) if args.obs_json else None
        out = compare_checkpoints(
            left_checkpoint=args.left,
            right_checkpoint=args.right,
            obs=obs,
            num_obs=args.num_obs,
            seed=args.seed,
        )
        print(json.dumps(out, indent=2))
        return

    # Legacy default: checkpoint -> torchscript export
    if args.checkpoint and args.out:
        info = export_torchscript_from_checkpoint(
            checkpoint_path=args.checkpoint,
            out_path=args.out,
            module=args.module,
            metadata_path=args.metadata_out or None,
        )
        print(json.dumps(info, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
