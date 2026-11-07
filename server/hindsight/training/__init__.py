# eval_all is imported as a submodule (hindsight.training.eval_all) rather than
# re-exported here, so `python -m hindsight.training.eval_all` doesn't trip the
# runpy "already imported" warning.
from .datasets import load, load_split, synthetic_root

__all__ = ["load", "load_split", "synthetic_root"]
