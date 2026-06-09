"""
VoiceScript MCP Server
======================
Exposes ffmpeg/ffprobe audio analysis tools via Model Context Protocol (FastMCP).

Tools:
    - get_audio_metadata   : Extract duration, bitrate, sample rate, channels, codec
    - detect_silence       : Detect silence segments using ffmpeg silencedetect
    - detect_clipping      : Detect volume levels and audio clipping

Usage:
    python mcp_server.py

Connect via MCP client or Claude Desktop:
    {
        "mcpServers": {
            "voicescript": {
                "command": "python",
                "args": ["/path/to/mcp_server.py"]
            }
        }
    }
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ── Initialize MCP server ────────────────────────────────
mcp = FastMCP(
    name="VoiceScript Audio Analysis",
    instructions=(
        "Audio analysis tools for court deposition recordings. "
        "Use get_audio_metadata first to understand the file, "
        "then detect_silence and detect_clipping to assess quality."
    ),
)


# ── Helper ────────────────────────────────────────────────
def _validate_file(file_path: str) -> Path:
    """Validate that the audio file exists and is accessible."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    return path


# ── Tool 1: get_audio_metadata ───────────────────────────
@mcp.tool()
def get_audio_metadata(file_path: str) -> dict:
    """
    Extract audio metadata from a file using ffprobe.

    Returns duration, bitrate, sample rate, channels, codec, and file size.
    This should be the first tool called when analysing an audio file.

    Args:
        file_path: Absolute or relative path to the audio file (.mp3, .wav, .m4a, etc.)

    Returns:
        dict with keys:
            file_name       : str   — filename only
            duration_seconds: float — total duration in seconds
            bitrate_kbps    : float — bitrate in kbps (None if unavailable)
            sample_rate_hz  : int   — sample rate in Hz (None if unavailable)
            channels        : int   — number of audio channels (None if unavailable)
            codec           : str   — audio codec name (None if unavailable)
            file_size_mb    : float — file size in MB rounded to 2 decimal places
    """
    _validate_file(file_path)

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        {},
    )
    fmt = data.get("format", {})

    duration  = float(fmt.get("duration", 0))
    bitrate   = float(fmt.get("bit_rate", 0)) / 1000 if fmt.get("bit_rate") else None
    file_size = round(float(fmt.get("size", 0)) / (1024 * 1024), 2) if fmt.get("size") else None

    return {
        "file_name":        Path(file_path).name,
        "duration_seconds": duration,
        "bitrate_kbps":     bitrate,
        "sample_rate_hz":   int(audio_stream.get("sample_rate", 0)) or None,
        "channels":         audio_stream.get("channels"),
        "codec":            audio_stream.get("codec_name"),
        "file_size_mb":     file_size,
    }


# ── Tool 2: detect_silence ───────────────────────────────
@mcp.tool()
def detect_silence(
    file_path: str,
    silence_thresh_db: float = -40.0,
    min_silence_duration: float = 2.0,
) -> dict:
    """
    Detect silence segments in an audio file using ffmpeg silencedetect filter.

    Args:
        file_path            : Path to the audio file
        silence_thresh_db    : Volume threshold for silence in dB (default: -40.0)
        min_silence_duration : Minimum silence duration in seconds to detect (default: 2.0)

    Returns:
        dict with keys:
            segment_count         : int   — number of silence segments found
            total_silence_seconds : float — total silence duration in seconds
            silence_ratio         : float — ratio of silence to total duration (0.0–1.0)
            segments              : list  — list of dicts with start, end, duration, label
    """
    _validate_file(file_path)

    # Get duration for ratio calculation
    meta = get_audio_metadata(file_path)
    total_duration = meta["duration_seconds"]

    cmd = [
        "ffmpeg", "-i", file_path,
        "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stderr = result.stderr

    silence_starts    = re.findall(r"silence_start: ([\d.]+)", stderr)
    silence_ends      = re.findall(r"silence_end: ([\d.]+)", stderr)
    silence_durations = re.findall(r"silence_duration: ([\d.]+)", stderr)

    segments = []
    for i, (start, end, dur) in enumerate(
        zip(silence_starts, silence_ends, silence_durations)
    ):
        segments.append({
            "start":    float(start),
            "end":      float(end),
            "duration": float(dur),
            "label":    f"Silence {i + 1}: {float(start):.1f}s – {float(end):.1f}s ({float(dur):.1f}s)",
        })

    total_silence = sum(float(d) for d in silence_durations)
    silence_ratio = total_silence / total_duration if total_duration > 0 else 0.0

    return {
        "segment_count":         len(segments),
        "total_silence_seconds": round(total_silence, 2),
        "silence_ratio":         round(silence_ratio, 4),
        "segments":              segments,
    }


# ── Tool 3: detect_clipping ──────────────────────────────
@mcp.tool()
def detect_clipping(file_path: str) -> dict:
    """
    Detect audio volume levels and clipping using ffmpeg volumedetect filter.

    Clipping is defined as max_volume >= -1.0 dB, which typically causes
    audible distortion and degrades transcription accuracy.

    Args:
        file_path: Path to the audio file

    Returns:
        dict with keys:
            avg_volume_db     : float — mean volume in dB (None if unavailable)
            max_volume_db     : float — peak volume in dB (None if unavailable)
            clipping_detected : bool  — True if max_volume >= -1.0 dB
            noise_level       : str   — 'low' / 'medium' / 'high' / 'unknown'
                                        low    : avg < -35 dB
                                        medium : -35 dB <= avg <= -20 dB
                                        high   : avg > -20 dB
    """
    _validate_file(file_path)

    cmd = [
        "ffmpeg", "-i", file_path,
        "-af", "volumedetect",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stderr = result.stderr

    mean_match = re.search(r"mean_volume: ([\-\d.]+) dB", stderr)
    max_match  = re.search(r"max_volume: ([\-\d.]+) dB", stderr)

    avg_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1))  if max_match  else None

    clipping = max_db is not None and max_db >= -1.0

    if avg_db is None:
        noise_level = "unknown"
    elif avg_db > -20:
        noise_level = "high"
    elif avg_db > -35:
        noise_level = "medium"
    else:
        noise_level = "low"

    return {
        "avg_volume_db":     avg_db,
        "max_volume_db":     max_db,
        "clipping_detected": clipping,
        "noise_level":       noise_level,
    }


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
