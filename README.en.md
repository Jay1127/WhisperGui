# Whisper GUI

A small local desktop app for testing Whisper speech-to-text options.

## Setup

Create and activate a virtual environment first.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Local Whisper execution requires FFmpeg.

```powershell
choco install ffmpeg
```

If you use Scoop, install it like this.

```powershell
scoop install ffmpeg
```

## Install

Install the Whisper package.

```powershell
pip install openai-whisper faster-whisper
```

Or install from `requirements.txt`.

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

## Test items

- Local Whisper transcription
- `openai-whisper` and `faster-whisper` engine selection
- `transcribe` and `translate` tasks
- Local model selection
- Output format selection
- Language option
- Prompt option
- Conversion progress display
- Word timestamps
- Subtitle line options
- Subtitle punctuation options
- Decoding options
- faster-whisper `Compute type`, `VAD filter`, `Hotwords`, and `Batch size` options

`Highlight words`, subtitle line options, punctuation options, and the silence threshold are available only when `Word timestamps` is enabled.

`Compute type`, `VAD filter`, `Hotwords`, and `Batch size` are available only when `Engine` is set to `faster-whisper`.

When `Output format` is `all`, the app writes multiple output files in the selected output folder using the input file name.

The app saves the converted text to the selected output path or folder and shows a preview in the window.
