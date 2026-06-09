# 🎙️ VoiceScript — Audio Analysis Agent

**AI Engineer Technical Assessment**  
**Author:** Dhany Delio Achmad  
**Stack:** Python · ffmpeg/ffprobe · Whisper · LangChain · Groq · Pydantic · Tenacity · FastMCP

---

## Overview

An AI-powered audio analysis pipeline that processes court deposition recordings and produces structured JSON reports. The system combines low-level signal analysis via `ffmpeg`/`ffprobe`, real language detection via **OpenAI Whisper**, and a **multi-agent LLM architecture** (LangChain + Groq) to deliver both machine-readable metrics and human-readable insights.

---

## Architecture

```
audio_file.mp3 / .wav
        │
        ▼
┌─────────────────────────────────────────────────────┐
│               ffmpeg / ffprobe Tools                │
│                                                     │
│  Tool 1: get_audio_metadata()   → ffprobe           │
│          duration, bitrate, sample_rate, channels   │
│                                                     │
│  Tool 2: detect_silence()       → silencedetect     │
│          silence segments, ratio, total duration    │
│                                                     │
│  Tool 3: detect_volume_and_clipping() → volumedetect│
│          avg_db, max_db, clipping, noise_level      │
│                                                     │
│  Tool 4: detect_language_whisper()  → Whisper tiny  │
│          ISO language code, confidence, bilingual   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            Python Infrastructure Rules              │
│  Deterministic checks — no LLM involved             │
│  • silence_ratio > 20%                              │
│  • clipping (max_vol >= -1 dB)                      │
│  • noise level medium/high                          │
│  • bitrate < 64 kbps                                │
│  • avg_volume_db < -40 dB                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            Multi-Agent LLM Pipeline                 │
│                                                     │
│  Agent 1 — Acoustic Expert                          │
│  Model  : llama-3.1-8b-instant  (fast, cheap)       │
│  Input  : volume_data, silence_data, metadata       │
│  Output : overall_usability, acoustic_issues        │
│                                                     │
│  Agent 2 — Linguistic Expert                        │
│  Model  : llama-3.1-8b-instant  (fast, cheap)       │
│  Input  : file_name, silence_data + Whisper result  │
│  Output : detected_languages, linguistic_issues     │
│                                                     │
│  Agent 3 — Manager / Reconciler                     │
│  Model  : llama-3.3-70b-versatile  (strong reasoning)│
│  Input  : Agent 1 + Agent 2 findings + all issues   │
│  Output : final summary + prioritised recommendations│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Python Sanitizer — Step 7                  │
│  Pass 1: Drops LLM issues contradicting raw numbers │
│  Pass 2: Semantic dedup — Python issues authoritative│
│  LLM-only insights (no Python equivalent) kept      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
         Pydantic AudioAnalysisReport
                       │
                       ▼
           JSON Report → ./reports/
```

---

## Design Decisions

### 1. Multi-Agent with Tiered Models
Rather than a single LLM call, the pipeline uses three specialised agents. Agent 1 and Agent 2 use `llama-3.1-8b-instant` (~12× cheaper than 70B) for focused, structured JSON tasks. Agent 3 uses `llama-3.3-70b-versatile` only where strong reasoning is needed — synthesising findings from two agents into a coherent professional narrative. This delivers multi-agent depth at near single-agent cost.

### 2. Whisper for Accurate Language Detection
Agent 2 (Linguistic Expert) uses OpenAI Whisper `tiny` to sample the first 30 seconds of audio and detect the spoken language from the actual signal — not from filename guesses. Whisper data is passed as ground truth into the LLM context, with a Python-level override that enforces Whisper's answer if the LLM returns "unknown". The `tiny` model loads once and is reused across files.

### 3. LLM as Interpreter, Not Calculator
`ffmpeg` computes all numeric metrics deterministically. The LLM never does math — it only interprets results. A Python sanitiser in Step 7 runs two passes before issues are finalised:
- **Pass 1 — Math guard:** drops any LLM issue that contradicts raw measured numbers (e.g., claiming bitrate is low when it is 128 kbps)
- **Pass 2 — Semantic deduplication:** if an LLM issue covers the same topic as a Python infra issue (clipping, noise, bitrate, silence, volume), the Python version is kept as authoritative and the LLM duplicate is dropped. LLM-only insights with no Python equivalent are kept as genuine added value.

### 4. Pydantic for Structured Output
All reports are validated through Pydantic models before being written to disk, guaranteeing a consistent JSON schema regardless of LLM response variation.

### 5. Fault Tolerance via Tenacity
Each LLM call is wrapped with a `tenacity` retry decorator targeting `RateLimitError` and `httpx.TimeoutException`:
- **Strategy:** exponential backoff 2s → 4s → 8s, max 3 attempts
- **Per-agent fallback:** if all retries fail, each agent returns a valid static object — rule-based verdict for Agent 1, Whisper result for Agent 2, templated summary for Agent 3 — so the pipeline never crashes

---

## Requirements

### System Dependencies
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

Verify:
```bash
ffmpeg -version && ffprobe -version
```

### Python Dependencies
```bash
pip install langchain langchain-groq pydantic python-dotenv groq tenacity httpx openai-whisper fastmcp
```

> The first run will automatically download the Whisper `tiny` model (~75 MB). Subsequent runs use the cached model.

**Python 3.9+** required.

---

## Setup

### 1. Project Structure

```
voicescript-pipeline/
├── voicescript_assessment.ipynb   # Main notebook — run this
├── mcp_server.py                  # MCP Server — exposes tools via FastMCP
├── .env                           # API keys (create manually, not committed)
├── .gitignore
├── README.md
├── audio_samples/                 # Test audio files
│   ├── bad_audio.mp3              # Low quality — clipping + high noise
│   └── moonlight-plaza.mp3        # Medium noise
└── reports/                       # Output JSON reports
    ├── report_bad_audio_mp3.json
    └── report_moonlight-plaza_mp3.json
```

### 2. Configure API Key

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

### 3. Run

Open `voicescript_assessment.ipynb` in Jupyter or VS Code, then:

```
Kernel → Restart & Run All
```

Reports are automatically saved to `./reports/` after the final cell.

---

## Output Schema

```json
{
  "file_name": "deposition_001.wav",
  "duration_seconds": 3600.0,
  "audio_quality": {
    "silence_ratio": 0.12,
    "clipping_detected": false,
    "avg_volume_db": -18.5
  },
  "issues": [
    "Long silence detected between 1200–1500s"
  ],
  "llm_summary": "Human-readable synthesis from the Manager agent.",
  "recommendations": [
    "Trim the 5-minute silence segment between 20–25 minutes.",
    "Apply noise reduction before transcription."
  ],
  "overall_usability": "partially_usable",
  "detected_languages": ["en"]
}
```

**`overall_usability` values:**

| Value | Meaning |
|---|---|
| `usable` | Clean audio, ready for legal transcription as-is |
| `partially_usable` | Requires pre-processing (noise reduction, silence trimming) |
| `unusable` | Severe clipping + high noise — not suitable for transcription |

---

## Example Outputs

### `bad_audio.mp3` — Unusable

> Low-quality recording with audio clipping, high background noise, and low bitrate.

```json
{
  "file_name": "bad_audio.mp3",
  "duration_seconds": 121.11,
  "audio_quality": {
    "silence_ratio": 0.1341,
    "clipping_detected": true,
    "avg_volume_db": -16.6
  },
  "issues": [
    "Audio clipping detected — may cause distortion",
    "High background noise detected — may reduce transcription accuracy",
    "Low bitrate: 47 kbps — may affect audio quality"
  ],
  "llm_summary": "The audio file bad_audio.mp3 has been deemed unusable due to significant acoustic issues, including clipping at max volume, high background noise, and a low bitrate of 47 kbps below the recommended 64 kbps threshold. These combined issues cause distortion and reduce transcription accuracy.",
  "recommendations": [
    "Increase the bitrate to at least 64 kbps to improve audio quality.",
    "Apply noise reduction techniques to minimise high background noise.",
    "Re-record the audio to avoid clipping and ensure a higher quality signal.",
    "Consider using a lossless codec to preserve original audio quality.",
    "Implement audio normalisation to ensure consistent volume levels."
  ],
  "overall_usability": "unusable",
  "detected_languages": ["en"]
}
```

---

### `moonlight-plaza.mp3` — Partially Usable

> Longer recording with acceptable bitrate and silence ratio, but medium background noise.

```json
{
  "file_name": "moonlight-plaza.mp3",
  "duration_seconds": 854.53,
  "audio_quality": {
    "silence_ratio": 0.0116,
    "clipping_detected": false,
    "avg_volume_db": -25.3
  },
  "issues": [
    "Medium background noise detected — may reduce transcription accuracy"
  ],
  "llm_summary": "The audio file moonlight-plaza.mp3 is partially usable, with an average volume of -25.3 dB and a low silence ratio of 1.16%, indicating generally clear audio. No clipping detected and bitrate is 128 kbps — well above threshold. However, medium background noise may reduce transcription accuracy and should be addressed before legal use.",
  "recommendations": [
    "Apply noise reduction to minimise medium background noise before transcription.",
    "Verify transcription output for errors caused by background noise.",
    "Consider re-recording in a quieter environment for optimal quality.",
    "Use noise-cancelling software to enhance the signal-to-noise ratio.",
    "Review sections with higher noise for potential re-recording."
  ],
  "overall_usability": "partially_usable",
  "detected_languages": ["en"]
}
```

---

## Notebook Structure

| Cell | Section | Description |
|---|---|---|
| 1 | Install Dependencies | pip install all required packages |
| 2 | Setup & Imports | Load `.env`, configure API keys |
| 3 | Pydantic Schema | `AudioQuality`, `AudioMetadata`, `AudioAnalysisReport` |
| 4 | Tool 1 | `get_audio_metadata()` via ffprobe |
| 5 | Tool 2 | `detect_silence()` via ffmpeg silencedetect |
| 6 | Tool 3 | `detect_volume_and_clipping()` via ffmpeg volumedetect |
| 6b | Tool 4 | `detect_language_whisper()` via OpenAI Whisper tiny |
| 7 | Multi-Agent LLM | Agent 1 · Agent 2 (+ Whisper) · Agent 3 with fault tolerance |
| 8 | Main Orchestrator | `analyze_audio()` — full 9-step pipeline with sanitizer |
| 9 | Run Analysis | Analyze `bad_audio.mp3` + `moonlight-plaza.mp3` |
| 10 | Comparison | Side-by-side results table + LLM summaries |
| 11 | Batch Processing | `batch_analyze()` for entire directories |
| 12 | Save Reports | Auto-save all reports to `./reports/` |

---

## MCP Server (FastMCP)

The three core ffmpeg tools are also exposed as an MCP (Model Context Protocol) server via `mcp_server.py`, allowing any MCP-compatible agent or client to call them directly.

**Tools exposed:**

| Tool | Description |
|---|---|
| `get_audio_metadata` | Extract duration, bitrate, sample rate, channels, codec via ffprobe |
| `detect_silence` | Detect silence segments with configurable threshold and min duration |
| `detect_clipping` | Detect volume levels, clipping, and noise level classification |

**Run the server:**

```bash
python mcp_server.py
```

**Connect from Claude Desktop** — add to your `mcp.json`:

```json
{
  "mcpServers": {
    "voicescript": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

---

## Extensibility

- **New ffmpeg tools** — add a `detect_*()` function following the same pattern and wire it into Step 4
- **New LLM agents** — add a specialist (e.g. speaker diarisation) between Agent 2 and Agent 3
- **Batch at scale** — replace sequential agent calls with `asyncio.gather()` for parallel processing
- **OpenAI fallback** — swap `ChatGroq` with `ChatOpenAI` by changing one import and one env variable
