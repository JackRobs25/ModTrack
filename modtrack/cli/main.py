"""ModTrack command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser and subcommand tree."""
    parser = argparse.ArgumentParser(prog="modtrack", description="ModTrack evaluation and optional finetuning")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run evaluation")
    eval_p.add_argument("--dataset", required=True, choices=["wildtrack", "multiviewx", "radarscenes"])
    eval_p.add_argument("--mode", required=True, choices=["spatial", "semantic", "joint"])

    ft = sub.add_parser("finetune", help="Run optional finetuning workflows")
    ft_sub = ft.add_subparsers(dest="component", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", required=True, choices=["wildtrack", "multiviewx"])
        p.add_argument("--data-root", required=True, type=Path)
        p.add_argument("--output-dir", required=True, type=Path)

    add_common(ft_sub.add_parser("yolo", help="Finetune YOLO detector"))
    add_common(ft_sub.add_parser("lift", help="Finetune Lift depth model"))
    add_common(ft_sub.add_parser("semantics", help="Finetune OSNet semantics model"))
    add_common(ft_sub.add_parser("semantics-prepare", help="Prepare person crops for semantics finetune"))

    return parser


def main() -> None:
    """Parse command-line arguments and dispatch to the selected workflow."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "eval":
            # Import lazily so `modtrack --help` does not import heavy runtime deps.
            from modtrack.core.run_eval import run_eval

            run_eval(dataset=args.dataset, mode=args.mode)
            return

        if args.component == "yolo":
            from modtrack.finetune.workflows import run_finetune_yolo

            run_finetune_yolo(args.dataset, args.data_root, args.output_dir)
        elif args.component == "lift":
            from modtrack.finetune.workflows import run_finetune_lift

            run_finetune_lift(args.dataset, args.data_root, args.output_dir)
        elif args.component == "semantics":
            from modtrack.finetune.workflows import run_finetune_semantics

            run_finetune_semantics(args.dataset, args.data_root, args.output_dir)
        elif args.component == "semantics-prepare":
            from modtrack.finetune.workflows import run_semantics_prepare

            run_semantics_prepare(args.dataset, args.data_root, args.output_dir)
        else:
            parser.error(f"Unknown finetune component: {args.component}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ModTrack] Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
