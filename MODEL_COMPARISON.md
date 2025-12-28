# Model Comparison: Qwen3 (Ollama) vs Qwen2.5 (HuggingFace)

## Quick Comparison

| Feature | `huihui_ai/qwen3-abliterated:8b-v2` (Ollama) | `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2` (HuggingFace) |
|---------|----------------------------------------------|--------------------------------------------------------------|
| **Model Size** | ~5GB (quantized 4-bit) | ~14GB (full precision) |
| **VRAM Usage** | ~5-6GB | ~14-16GB |
| **Loading Method** | Ollama API (model stays in memory) | Direct HuggingFace loading |
| **Setup Required** | Ollama v0.12.7+ installed, model pulled | None (auto-downloads) |
| **First Load Time** | Fast (model already in Ollama cache) | Slow (~30-60s to load from disk) |
| **Subsequent Requests** | Very fast (model in GPU memory) | Fast (model in GPU memory) |
| **Batch Processing** | Excellent (keep_alive keeps model loaded) | Good (model stays loaded during session) |
| **Quality** | High (quantized but optimized) | Highest (full precision) |
| **Speed** | Fast (optimized quantization) | Moderate (full precision) |
| **Memory Efficiency** | Excellent | Good |
| **Multi-app Usage** | Yes (shared Ollama service) | No (per-process) |

## Detailed Analysis

### 1. Memory Usage

**Ollama Model (`huihui_ai/qwen3-abliterated:8b-v2`):**
- Quantized to 4-bit precision
- ~5GB disk space
- ~5-6GB VRAM when loaded
- Model stays in memory between requests (with `keep_alive`)

**HuggingFace Model (`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`):**
- Full 16-bit (or bfloat16) precision
- ~14GB disk space
- ~14-16GB VRAM when loaded
- Model loaded per Python process

**Winner:** Ollama model (3x less VRAM)

### 2. Speed Performance

**Ollama Model:**
- ✅ Model stays in GPU memory between requests
- ✅ Optimized quantization for inference
- ✅ `keep_alive` parameter keeps model hot
- ✅ Dynamic `num_predict` (64-512 tokens) for short segments
- ✅ Lower temperature (0.05) for faster, deterministic output
- ⚠️ API overhead (HTTP requests)

**HuggingFace Model:**
- ✅ Direct GPU access (no API overhead)
- ✅ Full precision (better quality)
- ⚠️ Must reload if process restarts
- ⚠️ Fixed `max_new_tokens=512` (may be overkill for short segments)

**Winner:** Ollama model (for batch processing), HuggingFace (for single requests)

### 3. Quality

**Ollama Model:**
- Quantized to 4-bit but optimized for translation
- Slightly lower precision but optimized architecture
- Good for subtitle translation (short segments)

**HuggingFace Model:**
- Full precision (16-bit/bfloat16)
- Highest quality possible
- Better for complex/long translations

**Winner:** HuggingFace model (slight edge in quality)

### 4. Setup & Ease of Use

**Ollama Model:**
```bash
# Requires Ollama installation
curl -fsSL https://ollama.com/install.sh | sh

# Pull model (one-time)
ollama pull huihui_ai/qwen3-abliterated:8b-v2

# Then use directly
python main.py video.mp4 -m huihui_ai/qwen3-abliterated:8b-v2
```

**HuggingFace Model:**
```bash
# No setup needed - just use it
python main.py video.mp4 -m huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2
# Model auto-downloads on first use
```

**Winner:** HuggingFace model (zero setup)

### 5. Use Cases

**Choose Ollama Model (`huihui_ai/qwen3-abliterated:8b-v2`) when:**
- ✅ Limited VRAM (< 16GB)
- ✅ Processing many videos in batch
- ✅ Want model to stay loaded between runs
- ✅ Multiple applications need translation
- ✅ Want faster inference on short segments
- ✅ Don't mind Ollama setup

**Choose HuggingFace Model (`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`) when:**
- ✅ Have plenty of VRAM (≥ 16GB)
- ✅ Want highest quality
- ✅ Want zero setup (just works)
- ✅ Processing single videos
- ✅ Don't want to install Ollama
- ✅ Need full precision

## Performance Benchmarks

### Translation Speed (typical subtitle segment ~20-50 chars)

**Ollama Model:**
- First request: ~0.5-1.0s (model loading)
- Subsequent requests: ~0.2-0.5s (model in memory)
- Batch of 100 segments: ~20-50s total

**HuggingFace Model:**
- First request: ~1.0-2.0s (model loading)
- Subsequent requests: ~0.3-0.8s (model in memory)
- Batch of 100 segments: ~30-80s total

### Run Your Own Benchmark

To get actual speed measurements on your hardware:

```bash
# Run the speed comparison benchmark
python3 benchmark_speed.py
```

This will:
- Test both models with the same subtitle segments
- Measure average, median, and percentile times
- Calculate throughput (chars/sec)
- Show detailed comparison and recommendations
- Display time saved for batch processing

**Note:** First run will be slower due to model loading. The benchmark includes warmup to get accurate measurements.

### Memory Footprint

**Ollama Model:**
- Base: ~5GB VRAM
- Peak: ~6GB VRAM
- Can share with other Ollama models

**HuggingFace Model:**
- Base: ~14GB VRAM
- Peak: ~16GB VRAM
- Per-process (can't share easily)

## Recommendations

### For Most Users:
**Use `huihui_ai/qwen3-abliterated:8b-v2` (Ollama)** if:
- You have < 16GB VRAM
- You process multiple videos
- You want faster batch processing
- You're okay with Ollama setup

### For Maximum Quality:
**Use `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2` (HuggingFace)** if:
- You have ≥ 16GB VRAM
- Quality is top priority
- You want zero setup
- You process single videos occasionally

## Code Optimizations Applied

Both models benefit from recent optimizations:

1. **Dynamic token limits**: Adjusts `num_predict`/`max_new_tokens` based on text length
2. **Optimized temperature**: Lower for qwen3 (0.05), standard for qwen2.5 (0.1)
3. **Keep alive**: Ollama model stays in memory (5 minutes)
4. **Quality parameters**: `top_p=0.9`, `repeat_penalty=1.1`

## Conclusion

**For subtitle translation:**
- **Ollama model is recommended** for most users due to:
  - 3x less VRAM usage
  - Faster batch processing
  - Better memory efficiency
  - Quality is still excellent for subtitles

- **HuggingFace model is better** if:
  - You have high-end GPU (≥ 16GB VRAM)
  - Quality is absolutely critical
  - You want zero setup

Both models produce high-quality translations suitable for subtitles. The choice mainly depends on your hardware and workflow needs.

