"""Small command line entry: generate synthetic data, check provider status.

    python -m hindsight.cli gen [--minutes 30] [--render]
    python -m hindsight.cli status
"""

from __future__ import annotations

import argparse

from .config import settings
from .ingest import synthetic as S


def _gen(args) -> None:
    root = settings.data_dir / "synthetic"
    specs = [("train_01", 7, args.minutes, False),
             ("val_01", 11, args.minutes, False),
             ("demo_01", 23, min(args.minutes, 20), True)]
    for vid, seed, minutes, render in specs:
        v = S.generate(vid, minutes=minutes, seed=seed, render=render or args.render)
        v.save(root, render=render or args.render)
        print(f"{vid:10s} frames={v.n_frames} dets={len(v.dets)} putdowns={len(v.putdowns)} "
              f"segments={len(v.segments)} queries={len(v.queries)} -> {root/vid}")


def _status(_args) -> None:
    from .providers import provider_status

    for k, v in provider_status().items():
        print(f"{k:12s} {v}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="hindsight")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="materialize synthetic datasets")
    g.add_argument("--minutes", type=float, default=30.0)
    g.add_argument("--render", action="store_true", help="also write PNG frames")
    g.set_defaults(func=_gen)

    s = sub.add_parser("status", help="show active providers")
    s.set_defaults(func=_status)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
