"""
Utility to generate synthetic egocentric video (.mp4) for testing and demonstration.
Useful when full video datasets are not yet downloaded.
"""

from typing import Optional, List
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_demo_video(
    output_path: str = "demo_video.mp4",
    duration_seconds: float = 2.0,
    fps: int = 30,
    object_label: str = "keys",
    location_label: str = "wooden console table in entryway",
    action_label: str = "placed on table"
) -> str:
    """
    Creates a realistic synthetic egocentric video file (.mp4) simulating
    a Meta Aria glasses recording of a person placing an object down.

    Args:
        output_path: File path to save the .mp4 video.
        duration_seconds: Duration of the demo video in seconds.
        fps: Frames per second (default: 30).
        object_label: Name of the object interacting in the video.
        location_label: Name of the location where object is placed.
        action_label: Action description.

    Returns:
        Absolute path to the created .mp4 video file.
    """
    total_frames = int(duration_seconds * fps)
    frames_pil: List[Image.Image] = []

    width, height = 640, 480

    for i in range(total_frames):
        t = i / fps
        # Progress: 0.0 to 1.0
        progress = i / max(1, total_frames - 1)

        # Create room background (gradient or dark interior)
        bg_color = (40 + int(10 * progress), 45, 55)
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw simulated table surface
        table_top_y = 300
        draw.rectangle([(0, table_top_y), (width, height)], fill=(90, 60, 40))  # wooden table
        draw.line([(0, table_top_y), (width, table_top_y)], fill=(120, 85, 55), width=3)

        # Animate object moving in and being placed onto the table
        # Object starts high left, moves down onto table at t > 1.0
        if progress < 0.6:
            obj_x = int(150 + progress * 200)
            obj_y = int(120 + progress * 250)
            status = f"Moving {object_label} toward table"
        else:
            obj_x = int(150 + 0.6 * 200)
            obj_y = int(120 + 0.6 * 250)
            status = f"Placed {object_label} onto {location_label}"

        # Draw simulated object (e.g. golden keys or mug)
        obj_w, obj_h = 50, 35
        draw.rectangle([(obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h)], fill=(218, 165, 32), outline=(255, 255, 255), width=2)

        # Draw HUD / Aria Glasses metadata
        draw.rectangle([(10, 10), (width - 10, 85)], fill=(0, 0, 0, 160), outline=(100, 100, 100))
        draw.text((20, 15), f"Meta Aria Glasses - Egocentric Stream [Session 01 | {fps} FPS]", fill=(200, 255, 200))
        draw.text((20, 35), f"Time: {t:05.2f}s | Frame: {i:04d} | Status: {status}", fill=(255, 255, 255))
        draw.text((20, 55), f"Target Object: '{object_label}' at '{location_label}'", fill=(255, 215, 0))

        # Draw detected object bounding box
        draw.rectangle([(obj_x - 5, obj_y - 5), (obj_x + obj_w + 5, obj_y + obj_h + 5)], outline=(0, 255, 127), width=2)
        draw.text((obj_x - 5, obj_y - 20), f"{object_label} (det: 0.96)", fill=(0, 255, 127))

        frames_pil.append(img)

    # Save to MP4 using PyAV if available, then OpenCV. A broken OpenCV install
    # (for example, a NumPy ABI mismatch) should not prevent video generation.
    saved = False
    try:
        import av

        container = av.open(output_path, mode="w")
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for f in frames_pil:
            arr = np.array(f)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        saved = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        try:
            container.close()
        except Exception:
            pass
        saved = False

    if not saved:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for image in frames_pil:
                rgb = np.asarray(image)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out.write(bgr)
            out.release()
            saved = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception:
            saved = False

    if not saved:
        # Fallback: Save as animated GIF or folder of frames
        gif_path = output_path.replace(".mp4", ".gif")
        frames_pil[0].save(
            gif_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0
        )
        output_path = gif_path

    print(f"[DemoVideo] Generated {duration_seconds}s demo egocentric video at: {output_path}")
    return os.path.abspath(output_path)
