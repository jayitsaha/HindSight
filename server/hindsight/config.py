from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _parse_keys() -> list[str]:
    raw = os.environ.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY", "")
    out, seen = [], set()
    for tok in raw.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.startswith("nvapi-") and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


@dataclass
class Settings:
    provider: str = os.environ.get("HINDSIGHT_PROVIDER", "auto")

    nim_base: str = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    embed_model: str = os.environ.get("NIM_EMBED_MODEL", "nvidia/nvclip")
    vlm_model: str = os.environ.get("NIM_VLM_MODEL", "meta/llama-3.2-90b-vision-instruct")
    grounder_model: str = os.environ.get("NIM_GROUNDER_MODEL", "microsoft/kosmos-2")

    # retrieval
    pca_dim: int = int(os.environ.get("HINDSIGHT_PCA_DIM", "128"))

    # repetition: same action within repeat_window is a repeat; past repeat_gap is a clear non-repeat.
    repeat_window_s: float = 600.0
    repeat_gap_s: float = 1800.0

    data_dir: Path = Path(os.environ.get("HINDSIGHT_DATA", str(ROOT / "server" / "data")))
    model_dir: Path = Path(os.environ.get("HINDSIGHT_MODELS", str(ROOT / "server" / "artifacts")))

    api_keys: list[str] = field(default_factory=_parse_keys)

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_keys(self) -> bool:
        return bool(self.api_keys)


settings = Settings()
