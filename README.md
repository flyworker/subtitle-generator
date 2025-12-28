# Multi-Language Subtitle Generator

Automatically generate and translate subtitles from videos in any language to any other language using your local GPU (RTX 3070).

## Features

- **Multi-Language Support**: Translate from any language to any language (200+ languages via NLLB)
- **GPU-Accelerated**: Utilizes your RTX 3070 for fast processing
- **Offline Translation**: Uses local AI models (no API keys needed)
- **High-Quality**: OpenAI Whisper for transcription + NLLB/Qwen for translation
- **VLC Compatible**: Generates standard .srt files that work with VLC player
- **Free**: No API costs or usage limits
- **Multiple Models**: Supports NLLB models (default) and Qwen via Ollama

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (RTX 3070)
- FFmpeg installed on your system

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

**macOS:**
```bash
brew install ffmpeg
```

## Installation

1. **Clone or navigate to this directory:**
```bash
cd /home/flyworker/Documents/projects/subtittle
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch with CUDA support:**
```bash
# For CUDA 11.8 (check your CUDA version with: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install other dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate translated subtitles from a video (default: Japanese → Chinese):

```bash
# Using default NLLB model
python main.py /path/to/your/video.mp4

# Using recommended GGUF model (best quality and performance)
python main.py /path/to/your/video.mp4 --translation-model qwen3-gguf-q5ks
```

This will create a file named `video.zh.srt` in the same directory as your video.

### Language Selection

Specify source and target languages using ISO 639-1 language codes:

```bash
# Japanese to Chinese (default)
python main.py video.mp4 -s ja -t zh

# English to Spanish
python main.py video.mp4 -s en -t es

# French to German
python main.py video.mp4 -s fr -t de

# Korean to English
python main.py video.mp4 -s ko -t en

# Spanish to Portuguese
python main.py video.mp4 -s es -t pt

# Arabic to English
python main.py video.mp4 -s ar -t en
```

### Supported Language Codes

Common language codes (ISO 639-1):
- `ja` / `jpn` - Japanese
- `zh` / `zho` - Chinese (Simplified)
- `zh-tw` / `zh-hant` - Chinese (Traditional)
- `en` / `eng` - English
- `es` / `spa` - Spanish
- `fr` / `fra` - French
- `de` / `deu` - German
- `it` / `ita` - Italian
- `pt` / `por` - Portuguese
- `ru` / `rus` - Russian
- `ko` / `kor` - Korean
- `ar` / `arb` - Arabic
- `th` / `tha` - Thai
- `vi` / `vie` - Vietnamese
- `hi` / `hin` - Hindi
- `id` / `ind` - Indonesian
- `ms` / `zsm` - Malay
- And 180+ more languages supported by NLLB!

### Advanced Options

```bash
# Specify custom output path
python main.py video.mp4 -s ja -t zh -o my_subtitles.srt

# Use larger Whisper model for better accuracy
python main.py video.mp4 -s en -t es --whisper-model large

# Use smaller translation model for faster processing (less VRAM)
python main.py video.mp4 -s fr -t de --translation-model facebook/nllb-200-1.3B

# Use fastest model (least VRAM)
python main.py video.mp4 -s ko -t en --translation-model facebook/nllb-200-distilled-600M

# Keep both source and target language subtitles (for language learning)
python main.py video.mp4 -s ja -t zh --keep-original

# Use HuggingFace model (recommended - loads directly, no Ollama needed)
python main.py video.mp4 -s ja -t zh --translation-model huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2

# Use Ollama model (requires Ollama v0.12.7+ installed and model pulled)
python main.py video.mp4 -s ja -t zh --translation-model huihui_ai/qwen2.5-vl-abliterated:7b

# Use newer Ollama model
python main.py video.mp4 -s ja -t zh --translation-model huihui_ai/qwen3-abliterated:8b-v2

# Use GGUF model (recommended - best quality and performance)
python main.py video.mp4 -s ja -t zh --translation-model qwen3-gguf-q5ks
```

### Available Whisper Models

- `tiny` - Fastest, lowest accuracy (~1GB VRAM)
- `base` - Fast, decent accuracy (~1GB VRAM)
- `small` - Balanced (~2GB VRAM)
- `medium` - **Recommended** - Good balance (~5GB VRAM)
- `large` - Best accuracy, slower (~10GB VRAM)

### Available Translation Models

**NLLB Models (Direct Loading, supports 200+ languages):**
- `facebook/nllb-200-3.3B` - **Default** - Best quality (~13GB VRAM)
- `facebook/nllb-200-1.3B` - Good quality, faster (~5GB VRAM)
- `facebook/nllb-200-distilled-600M` - Fastest, good quality (~2.5GB)

**HuggingFace Models (Direct Loading):**
- `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2` - Highest quality, full precision (~14GB VRAM, direct from HuggingFace)
- `Qwen/Qwen2.5-7B-Instruct` - High quality, general purpose (~14GB VRAM)

**HuggingFace GGUF Models (Quantized - Best Performance - RECOMMENDED):**
- `qwen3-gguf-q5ks` - **⭐ Recommended** - Q5_K_S quantization (5.7GB), imatrix optimized for highest quality. Already installed and ready to use! Best balance of quality and speed.
- Source repository: [mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF) - Multiple quantization levels (2.2-6.8GB)
  - Alternative quantizations: `Q4_K_S` (4.8GB) or `Q4_K_M` (5.1GB) for best speed/quality balance
  - See [GGUF_SETUP.md](GGUF_SETUP.md) for detailed setup instructions

**Ollama Models (Alternative - Requires Ollama v0.12.7+, supports many languages):**
- `huihui_ai/qwen3-abliterated:8b-v2` - High quality, optimized for translation (~5GB with Ollama)
- `huihui_ai/qwen2.5-vl-abliterated:7b` - High quality, optimized for translation (~7GB with Ollama)
- `huihui_ai/qwen3-vl-abliterated:8b-instruct` - High quality, optimized for translation (~8GB with Ollama)
- `qwen2.5:7b-instruct` - High quality, general purpose (~7GB with Ollama)

**Note**: 
- **GGUF models** (like `qwen3-gguf-q5ks`) are the **recommended** option - best quality and performance. Already installed and ready to use!
- **HuggingFace models** (like `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`) are loaded directly - no setup needed, just use the model name.
- **Ollama models** use format `namespace/model:tag`. To use an Ollama model, ensure Ollama v0.12.7+ is installed and the model is pulled.
- **📊 See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for detailed comparison between models.**

### Setting Up Models

```bash
# Install/update Ollama to v0.12.7+
curl -fsSL https://ollama.com/install.sh | sh

# Pull an Ollama model (example)
ollama pull huihui_ai/qwen3-abliterated:8b-v2
# Or use other models:
# ollama pull huihui_ai/qwen2.5-vl-abliterated:7b

# For HuggingFace GGUF models, import into Ollama first:
# 1. Download the GGUF file (e.g., Q4_K_S quantization ~4.8GB)
# 2. Create Ollama model from GGUF:
ollama create qwen3-gguf -f Modelfile
# Where Modelfile contains:
# FROM /path/to/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF.Q4_K_S.gguf
# TEMPLATE """{{ .System }}{{ .Prompt }}"""
# SYSTEM "You are a professional translator."

# Then use it directly
python main.py video.mp4 --translation-model huihui_ai/qwen3-abliterated:8b-v2

# Or use the installed GGUF model (Q5_K_S quantization, 5.7GB) - RECOMMENDED:
python main.py video.mp4 --translation-model qwen3-gguf-q5ks
```

### Benchmarking Model Speed

To compare translation speed between different models:

```bash
# Compare Ollama vs HuggingFace models
python3 benchmark_speed.py
```

This will test both models with the same subtitle segments and show detailed performance metrics.

## Using Subtitles in VLC Player

### Method 1: Automatic Loading (Recommended)

1. Make sure the subtitle file has the same name as the video:
   ```
   movie.mp4
   movie.zh.srt
   ```
2. Place both files in the same folder
3. Open the video in VLC - subtitles will load automatically!

### Method 2: Manual Loading

1. Open your video in VLC
2. Go to: **Subtitle → Add Subtitle File...**
3. Select your `.srt` file
4. Subtitles will appear immediately

### Adjust Subtitle Settings in VLC

- **Position/Size**: Tools → Preferences → Subtitles/OSD
- **Timing**: Press `H` to delay or `G` to advance subtitle timing

## Project Structure

```
subtittle/
├── main.py              # Main pipeline script
├── transcribe.py        # Whisper-based multi-language transcription
├── translate.py         # NLLB/Qwen multi-language translation
├── srt_generator.py     # SRT subtitle file generator
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Audio Extraction & Transcription**: Whisper extracts audio from your video and transcribes speech to text with timestamps (supports 100+ languages)
2. **Translation**: NLLB or Qwen model translates text segments from source language to target language
3. **SRT Generation**: Creates a standard .srt subtitle file with proper formatting

## Performance Notes

With your RTX 3070 (8GB VRAM):
- **Recommended**: `medium` Whisper + `qwen3-gguf-q5ks` (~40-60 min for a 2-hour movie, best quality)
- **Alternative**: `medium` Whisper + `1.3B` NLLB (~60-90 min for a 2-hour movie)
- **High Quality**: `large` Whisper + `3.3B` NLLB (~90-120 min for a 2-hour movie)
- **Fast**: `medium` Whisper + `600M` NLLB (~30-60 min for a 2-hour movie)

First run will download models (~5-10GB total), but subsequent runs will be faster.

## Examples

### Example 1: Japanese to Chinese (Original Use Case)
```bash
python main.py japanese_movie.mp4 -s ja -t zh
# Creates: japanese_movie.zh.srt
```

### Example 2: English to Spanish
```bash
python main.py english_video.mp4 -s en -t es
# Creates: english_video.es.srt
```

### Example 3: French to German with Dual Subtitles
```bash
python main.py french_movie.mp4 -s fr -t de --keep-original
# Creates: french_movie.de.srt (with both French and German)
```

### Example 4: Korean to English (Fast Mode)
```bash
python main.py korean_show.mp4 -s ko -t en \
  --whisper-model small \
  --translation-model facebook/nllb-200-distilled-600M
# Creates: korean_show.en.srt
```

## Troubleshooting

### CUDA Out of Memory

If you get CUDA out of memory errors:
- Use smaller models: `--whisper-model small`
- Use smaller translation model: `--translation-model facebook/nllb-200-distilled-600M`
- Close other GPU applications
- Process shorter video segments

### Subtitle Not Showing in VLC

- Ensure filename matches video (e.g., `movie.mp4` and `movie.zh.srt`)
- Check file encoding is UTF-8
- Manually load: Subtitle → Add Subtitle File

### Poor Translation Quality

- Try larger translation model: `--translation-model facebook/nllb-200-3.3B`
- Check that source audio is clear (Whisper accuracy affects translation)
- For better quality, use `--translation-model facebook/nllb-200-1.3B` or the default `3.3B` model
- Some language pairs may work better with specific models

### Language Not Recognized

- Use ISO 639-1 language codes (e.g., `ja`, `en`, `es`, `fr`)
- For NLLB, the system will automatically convert to NLLB format
- If a language code doesn't work, try the 3-letter ISO 639-3 code (e.g., `jpn`, `eng`, `spa`)

## License

This project uses open-source models and libraries. Please respect the licenses of:
- OpenAI Whisper (MIT)
- Meta NLLB (CC-BY-NC 4.0)
- Transformers (Apache 2.0)

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Meta NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) - Multi-language translation
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Model framework
