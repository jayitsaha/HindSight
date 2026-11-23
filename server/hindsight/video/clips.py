"""
Frame extraction from long session videos around evidence time spans.
"""

from typing import Dict, List
import os


def clip_frame_times(start: float, end: float, n: int, pad: float = 5.0, duration: float = None) -> List[float]:
    """n evenly spaced timestamps (seconds) over [start - pad, end + pad], clamped to the video."""
    lo = max(0.0, start - pad)
    hi = end + pad
    if duration is not None:
        hi = min(hi, max(0.0, duration - 0.5))
        lo = min(lo, hi)
    if n <= 1 or hi <= lo:
        return [(lo + hi) / 2]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def extract_frames(video_path: str, times: List[float], out_paths: List[str], max_side: int = 768) -> None:
    """Seeks to each timestamp with PyAV and writes the nearest following frame as a JPEG."""
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        for t, out in zip(times, out_paths):
            container.seek(int(t / tb), stream=stream, backward=True)
            image = None
            for frame in container.decode(stream):
                if frame.time is None or frame.time >= t - 0.05:
                    image = frame.to_image()
                    break
                image = frame.to_image()  # keep the last decoded frame if seek landed before t
            if image is None:
                raise RuntimeError(f"No frame decoded at t={t:.1f}s in {video_path}")
            image.thumbnail((max_side, max_side))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            image.save(out, quality=90)


def video_duration(video_path: str) -> float:
    import av

    with av.open(video_path) as container:
        if container.duration:
            return container.duration / 1e6
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)
