# Setting Up HuggingFace GGUF Models with Ollama

The HuggingFace model `mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF` provides optimized quantized versions that can offer better performance than the standard Ollama models.

## Available Quantizations

From the [model page](https://huggingface.co/mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF):

| Quantization | Size | Quality | Recommended For |
|-------------|------|---------|----------------|
| `Q4_K_S` | 4.8 GB | Optimal size/speed/quality | **Recommended** - Best balance |
| `Q4_K_M` | 5.1 GB | Fast, recommended | High quality, good speed |
| `Q5_K_M` | 5.85 GB | Higher quality | Maximum quality with good speed |
| `Q3_K_M` | 4.12 GB | Smaller, faster | Limited VRAM |
| `IQ4_XS` | 4.56 GB | Imatrix quant | Alternative to Q4_K_S |

## Setup Instructions

### Option 1: Download and Import into Ollama (Recommended)

1. **Download the GGUF file** from HuggingFace:
   ```bash
   # Install huggingface-cli if needed
   pip install huggingface-hub
   
   # Download the recommended Q4_K_S quantization
   huggingface-cli download mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF \
     Huihui-Qwen3-8B-abliterated-v2.i1-Q4_K_S.gguf \
     --local-dir ./models \
     --local-dir-use-symlinks False
   ```

2. **Create a Modelfile** for Ollama:
   ```bash
   cat > Modelfile << 'EOF'
   FROM ./models/Huihui-Qwen3-8B-abliterated-v2.i1-Q4_K_S.gguf
   TEMPLATE """{{ .System }}{{ .Prompt }}"""
   SYSTEM "You are a professional translator."
   PARAMETER temperature 0.05
   PARAMETER top_p 0.9
   PARAMETER repeat_penalty 1.1
   EOF
   ```

3. **Import into Ollama**:
   ```bash
   ollama create qwen3-gguf-q4ks -f Modelfile
   ```

4. **Use the model**:
   ```bash
   python main.py video.mp4 -m qwen3-gguf-q4ks
   ```

### Option 2: Direct HuggingFace Download Script

Create a script to automate the process:

```bash
#!/bin/bash
# setup_gguf_model.sh

MODEL_NAME="mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF"
QUANT="Q4_K_S"
GGUF_FILE="Huihui-Qwen3-8B-abliterated-v2.i1-${QUANT}.gguf"
OLLAMA_MODEL_NAME="qwen3-gguf-q4ks"

echo "Downloading ${GGUF_FILE}..."
huggingface-cli download ${MODEL_NAME} ${GGUF_FILE} --local-dir ./models

echo "Creating Ollama model..."
cat > Modelfile << EOF
FROM ./models/${GGUF_FILE}
TEMPLATE """{{ .System }}{{ .Prompt }}"""
SYSTEM "You are a professional translator."
PARAMETER temperature 0.05
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
EOF

ollama create ${OLLAMA_MODEL_NAME} -f Modelfile

echo "✓ Model ready! Use with: python main.py video.mp4 -m ${OLLAMA_MODEL_NAME}"
```

## Why Use GGUF Models?

1. **Better Quantization**: Imatrix quants (IQ*) are optimized for better quality at the same size
2. **More Options**: Choose the exact quantization level for your hardware
3. **Performance**: Often faster than standard Ollama quantizations
4. **Quality**: Better quality retention at lower bit levels

## Comparison with Standard Ollama Model

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `huihui_ai/qwen3-abliterated:8b-v2` (Ollama) | ~5GB | Fast | High |
| `qwen3-gguf-q4ks` (GGUF Q4_K_S) | 4.8GB | Faster | Higher |
| `qwen3-gguf-q5km` (GGUF Q5_K_M) | 5.85GB | Fast | Highest |

## Usage

After setup, use the model just like any Ollama model:

```bash
# In your translation command
python main.py video.mp4 -m qwen3-gguf-q4ks

# Or in Python code
from translate import SubtitleTranslator
translator = SubtitleTranslator(
    source_lang="ja",
    target_lang="zh",
    model_name="qwen3-gguf-q4ks"
)
```

## Troubleshooting

**Issue: Model not found**
- Ensure you've created the Ollama model: `ollama list`
- Check the model name matches what you created

**Issue: Slow performance**
- Try a smaller quantization (Q3_K_M or Q4_K_S)
- Ensure GPU is being used: `nvidia-smi`

**Issue: Download fails**
- Check internet connection
- Try downloading manually from HuggingFace web interface
- Use `huggingface-cli login` if authentication is required



