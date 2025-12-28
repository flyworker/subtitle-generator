# Ollama Setup Guide

This guide shows you how to set up Ollama to run Qwen2.5-7B-Instruct as a local service, which is more efficient than loading the model directly each time.

## Benefits of Using Ollama

- ✅ Model stays loaded in memory (faster subsequent translations)
- ✅ No need to download via HuggingFace (Ollama handles downloads)
- ✅ Better memory management
- ✅ Can be used by multiple applications
- ✅ Automatic GPU acceleration

## Installation

### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or download from: https://ollama.com/download

**Verify installation:**
```bash
ollama --version
```

### 2. Download Qwen2.5-7B-Instruct Model

```bash
ollama pull qwen2.5:7b-instruct
```

This will download the model (~4.7GB with quantization). The download progress will be shown.

**Note:** Ollama uses quantization (4-bit) to reduce model size while maintaining quality. The full precision model is ~13-14GB, but Ollama's quantized version is much smaller and faster.

**Alternative smaller models:**
```bash
# 3B model (faster, less VRAM, ~2-3GB with quantization)
ollama pull qwen2.5:3b-instruct

# 14B model (better quality, more VRAM, ~8-9GB with quantization)
ollama pull qwen2.5:14b-instruct
```

### 3. Start Ollama Service

Ollama runs as a service automatically after installation. Verify it's running:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response with available models.

### 4. Test the Setup

```bash
ollama run qwen2.5:7b-instruct "Translate to Chinese: こんにちは"
```

## Usage with This Project

The translation script will **automatically detect** Ollama if it's running and use it. No code changes needed!

If Ollama is not available, it will fall back to direct model loading.

### Manual Control

You can explicitly enable/disable Ollama:

```python
# Force use Ollama
translator = JapaneseToChinese(use_ollama=True)

# Force direct loading (skip Ollama)
translator = JapaneseToChinese(use_ollama=False)

# Use different Ollama model
translator = JapaneseToChinese(
    use_ollama=True,
    ollama_model="qwen2.5:3b-instruct"
)
```

## Troubleshooting

### Ollama not starting

```bash
# Check if Ollama is running
systemctl status ollama

# Start Ollama service
sudo systemctl start ollama

# Or run manually
ollama serve
```

### Model not found

```bash
# List available models
ollama list

# Pull the model again
ollama pull qwen2.5:7b-instruct
```

### Connection refused

Make sure Ollama is running on port 11434:
```bash
curl http://localhost:11434/api/tags
```

If using a different port or remote server:
```python
translator = JapaneseToChinese(
    use_ollama=True,
    ollama_url="http://your-server:11434"
)
```

## Performance Tips

1. **Keep Ollama running**: Don't stop the Ollama service between runs
2. **Use GPU**: Ollama automatically uses GPU if available
3. **Model size**: 3B is faster, 7B is better quality, 14B is best quality
4. **Memory**: 7B needs ~8GB VRAM, 3B needs ~4GB VRAM

## Next Steps

Once Ollama is set up, just run your script normally:

```bash
python main.py your_video.mp4
```

The script will automatically use Ollama if available!

