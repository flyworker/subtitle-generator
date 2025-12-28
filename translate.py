"""
Multi-language subtitle translation using local NLLB or Qwen model with GPU acceleration.
Supports both Ollama API (for Qwen) and direct model loading (NLLB or Qwen).
"""
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import os
import time
import requests
import json


# NLLB language code mapping (ISO 639-1/639-3 to NLLB codes)
# Common languages supported by NLLB
NLLB_LANGUAGE_CODES = {
    # Asian languages
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan", "japanese": "jpn_Jpan",
    "zh": "zho_Hans", "zho": "zho_Hans", "chinese": "zho_Hans", "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", "zh-hant": "zho_Hant",
    "ko": "kor_Hang", "kor": "kor_Hang", "korean": "kor_Hang",
    "th": "tha_Thai", "tha": "tha_Thai", "thai": "tha_Thai",
    "vi": "vie_Latn", "vie": "vie_Latn", "vietnamese": "vie_Latn",
    "id": "ind_Latn", "ind": "ind_Latn", "indonesian": "ind_Latn",
    "ms": "zsm_Latn", "zsm": "zsm_Latn", "malay": "zsm_Latn",
    "hi": "hin_Deva", "hin": "hin_Deva", "hindi": "hin_Deva",
    "ar": "arb_Arab", "arb": "arb_Arab", "arabic": "arb_Arab",
    
    # European languages
    "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
    "es": "spa_Latn", "spa": "spa_Latn", "spanish": "spa_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn", "french": "fra_Latn",
    "de": "deu_Latn", "deu": "deu_Latn", "german": "deu_Latn",
    "it": "ita_Latn", "ita": "ita_Latn", "italian": "ita_Latn",
    "pt": "por_Latn", "por": "por_Latn", "portuguese": "por_Latn",
    "ru": "rus_Cyrl", "rus": "rus_Cyrl", "russian": "rus_Cyrl",
    "pl": "pol_Latn", "pol": "pol_Latn", "polish": "pol_Latn",
    "nl": "nld_Latn", "nld": "nld_Latn", "dutch": "nld_Latn",
    "tr": "tur_Latn", "tur": "tur_Latn", "turkish": "tur_Latn",
    "el": "ell_Grek", "ell": "ell_Grek", "greek": "ell_Grek",
    "cs": "ces_Latn", "ces": "ces_Latn", "czech": "ces_Latn",
    "sv": "swe_Latn", "swe": "swe_Latn", "swedish": "swe_Latn",
    "no": "nob_Latn", "nob": "nob_Latn", "norwegian": "nob_Latn",
    "fi": "fin_Latn", "fin": "fin_Latn", "finnish": "fin_Latn",
    "da": "dan_Latn", "dan": "dan_Latn", "danish": "dan_Latn",
    "hu": "hun_Latn", "hun": "hun_Latn", "hungarian": "hun_Latn",
    "ro": "ron_Latn", "ron": "ron_Latn", "romanian": "ron_Latn",
    "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl", "ukrainian": "ukr_Cyrl",
    
    # Other languages
    "he": "heb_Hebr", "heb": "heb_Hebr", "hebrew": "heb_Hebr",
    "fa": "pes_Arab", "pes": "pes_Arab", "persian": "pes_Arab", "farsi": "pes_Arab",
    "bn": "ben_Beng", "ben": "ben_Beng", "bengali": "ben_Beng",
    "ta": "tam_Taml", "tam": "tam_Taml", "tamil": "tam_Taml",
    "te": "tel_Telu", "tel": "tel_Telu", "telugu": "tel_Telu",
    "mr": "mar_Deva", "mar": "mar_Deva", "marathi": "mar_Deva",
    "ur": "urd_Arab", "urd": "urd_Arab", "urdu": "urd_Arab",
}

# Language names for display
LANGUAGE_NAMES = {
    "ja": "Japanese", "jpn": "Japanese",
    "zh": "Chinese", "zho": "Chinese", "zh-cn": "Chinese",
    "zh-tw": "Traditional Chinese", "zh-hant": "Traditional Chinese",
    "ko": "Korean", "kor": "Korean",
    "en": "English", "eng": "English",
    "es": "Spanish", "spa": "Spanish",
    "fr": "French", "fra": "French",
    "de": "German", "deu": "German",
    "it": "Italian", "ita": "Italian",
    "pt": "Portuguese", "por": "Portuguese",
    "ru": "Russian", "rus": "Russian",
    "ar": "Arabic", "arb": "Arabic",
    "th": "Thai", "tha": "Thai",
    "vi": "Vietnamese", "vie": "Vietnamese",
    "hi": "Hindi", "hin": "Hindi",
    "id": "Indonesian", "ind": "Indonesian",
    "ms": "Malay", "zsm": "Malay",
}


def get_nllb_code(lang_code: str) -> str:
    """Convert language code to NLLB format."""
    lang_lower = lang_code.lower().strip()
    if lang_lower in NLLB_LANGUAGE_CODES:
        return NLLB_LANGUAGE_CODES[lang_lower]
    # If already in NLLB format, return as-is
    if "_" in lang_code:
        return lang_code
    # Try to find by partial match
    for key, value in NLLB_LANGUAGE_CODES.items():
        if lang_lower in key or key in lang_lower:
            return value
    # Default: assume it's already in NLLB format or return as-is
    return lang_code


def get_language_name(lang_code: str) -> str:
    """Get human-readable language name."""
    lang_lower = lang_code.lower().strip()
    if lang_lower in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[lang_lower]
    # Try to find by partial match
    for key, value in LANGUAGE_NAMES.items():
        if lang_lower in key or key in lang_lower:
            return value
    return lang_code.title()


class SubtitleTranslator:
    """Translate subtitles between any two languages using NLLB or Qwen model."""

    @staticmethod
    def unload_ollama(ollama_url: str = "http://localhost:11434"):
        """
        Unload all Ollama models from GPU memory to free up VRAM.
        
        This is useful when you need to free GPU memory for other tasks (e.g., Whisper transcription).
        The model will be automatically reloaded when the next translation request is made.
        
        Args:
            ollama_url: Ollama API endpoint (default: http://localhost:11434)
        """
        try:
            # Get list of available models
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Unload each model by sending a request with keep_alive: 0
                for model in models:
                    model_name = model.get("name", "")
                    if model_name:
                        try:
                            requests.post(
                                f"{ollama_url}/api/generate",
                                json={"model": model_name, "prompt": "", "keep_alive": 0},
                                timeout=5
                            )
                        except:
                            pass  # Ignore errors for individual models
                print("✓ Ollama models unloaded from GPU memory")
        except:
            pass  # Silently fail if Ollama is not running or not available

    def __init__(
        self, 
        source_lang: str = "ja",
        target_lang: str = "zh",
        model_name: str = "facebook/nllb-200-3.3B",
        use_ollama: Optional[bool] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b-instruct"
    ):
        """
        Initialize the translation model.

        Args:
            source_lang: Source language code (e.g., "ja", "en", "es", "fr")
            target_lang: Target language code (e.g., "zh", "en", "es", "fr")
            model_name: Model name (HuggingFace or Ollama format)
                       HuggingFace models:
                       - facebook/nllb-200-3.3B (default, best quality, ~13GB)
                       - facebook/nllb-200-1.3B (good quality, ~5GB)
                       - facebook/nllb-200-distilled-600M (fast, ~2.5GB)
                       - Qwen/Qwen2.5-7B-Instruct (alternative, high quality)
                       HuggingFace models (direct loading):
                       - huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2 (recommended, ~14GB VRAM)
                       Ollama models (format: namespace/model:tag):
                       - huihui_ai/qwen3-abliterated:8b-v2 (via Ollama, ~5GB)
                       - huihui_ai/qwen2.5-vl-abliterated:7b (via Ollama, ~7GB)
                       - huihui_ai/qwen3-vl-abliterated:8b-instruct (via Ollama, ~8GB)
                       - qwen2.5:7b-instruct (via Ollama, ~7GB)
                       HuggingFace GGUF models (quantized, better performance):
                       - mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF (recommended, various quantizations 2.2-6.8GB, see https://huggingface.co/mradermacher/Huihui-Qwen3-8B-abliterated-v2-i1-GGUF)
            use_ollama: If True, use Ollama API. If None, auto-detect based on model_name format
            ollama_url: Ollama API endpoint (default: http://localhost:11434)
            ollama_model: Ollama model name (default: qwen2.5:7b-instruct, ignored if model_name is Ollama format)
        """
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self.model_name = model_name
        self.is_nllb = "nllb" in model_name.lower()
        
        # Get language names for display
        self.source_lang_name = get_language_name(source_lang)
        self.target_lang_name = get_language_name(target_lang)
        
        # Detect if model_name is an Ollama model format
        # Ollama models typically have format: namespace/model:tag
        # Examples: "huihui_ai/qwen3-vl-abliterated:8b-instruct", "qwen2.5:7b-instruct"
        # Note: HuggingFace uses "huihui-ai" (hyphen), Ollama uses "huihui_ai" (underscore)
        has_colon = ":" in model_name
        has_slash = "/" in model_name
        # Check for known Ollama namespaces (Ollama uses underscore, not hyphen)
        # Ollama namespaces: huihui_ai (underscore), qwen2.5, qwen
        known_ollama_namespaces = ["huihui_ai", "qwen2.5", "qwen"]
        is_known_ollama = any(model_name.startswith(f"{ns}/") for ns in known_ollama_namespaces)
        # Ollama format: must have colon and slash (namespace/model:tag format)
        # OR has slash and matches known Ollama namespace (for models without explicit tag)
        is_ollama_model_format = (has_colon and has_slash) or (has_slash and is_known_ollama and not model_name.startswith("huihui-ai/"))
        
        # Get NLLB language codes if using NLLB
        if self.is_nllb:
            self.source_nllb_code = get_nllb_code(source_lang)
            self.target_nllb_code = get_nllb_code(target_lang)
            use_ollama = False  # NLLB models don't support Ollama
        elif is_ollama_model_format:
            # Model name is in Ollama format, use it directly
            use_ollama = True
            ollama_model = model_name  # Use the model_name as the Ollama model
        else:
            # HuggingFace model - don't use Ollama unless explicitly requested
            # Explicit HuggingFace models (with / but not Ollama format) should load directly
            if has_slash and not model_name.startswith("facebook/"):
                # This is a HuggingFace model, load directly (not via Ollama)
                use_ollama = False if use_ollama is None else use_ollama
            else:
                # Auto-detect Ollama if not specified (only for non-NLLB, non-HuggingFace models)
                if use_ollama is None:
                    use_ollama = self._check_ollama_available(ollama_url)
        
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        # Use model_name as ollama_model if it's in Ollama format, otherwise use provided ollama_model
        self.ollama_model = model_name if is_ollama_model_format else ollama_model
        
        if self.use_ollama:
            print(f"Using Ollama API for translation (model: {self.ollama_model})")
            print(f"Ollama endpoint: {ollama_url}")
            if not self._verify_ollama_model(self.ollama_model):
                print(f"\n⚠ Model '{self.ollama_model}' not found in Ollama.")
                print(f"Available models:")
                try:
                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        for m in models[:5]:  # Show first 5 models
                            print(f"  - {m.get('name', 'unknown')}")
                        if len(models) > 5:
                            print(f"  ... and {len(models) - 5} more")
                except:
                    pass
                print(f"\nPlease run: ollama pull {self.ollama_model}")
                raise RuntimeError(f"Ollama model '{self.ollama_model}' not available. Run 'ollama pull {self.ollama_model}' to install it.")
            print("✓ Ollama connection verified")
        else:
            # Direct model loading
            self._load_model_directly(model_name)

    def _check_ollama_available(self, ollama_url: str) -> bool:
        """Check if Ollama is available and running."""
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _verify_ollama_model(self, model_name: str) -> bool:
        """Verify that the specified model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Check for exact match or if model name starts with the provided name
                # This handles both "model:tag" format and partial matches
                for m in models:
                    model_full_name = m.get("name", "")
                    if model_full_name == model_name or model_full_name.startswith(model_name):
                        return True
                return False
            return False
        except:
            return False
    
    def _check_model_cache(self, model_name: str) -> Tuple[bool, str, float]:
        """
        Check if model is cached locally.
        
        Returns:
            (is_cached, cache_path, size_gb)
        """
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Convert model name to cache directory format
        # e.g., "facebook/nllb-200-3.3B" -> "models--facebook--nllb-200-3.3B"
        cache_model_name = f"models--{model_name.replace('/', '--')}"
        model_cache_dir = cache_dir / cache_model_name
        
        if not model_cache_dir.exists():
            return False, str(cache_dir), 0.0
        
        # Find the snapshot directory
        snapshots_dir = model_cache_dir / "snapshots"
        
        if not snapshots_dir.exists():
            return False, str(cache_dir), 0.0
        
        # Get the first snapshot (usually there's only one)
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return False, str(cache_dir), 0.0
        
        snapshot_dir = snapshots[0]
        
        # Check for model weight files (these are the large files)
        model_files = list(snapshot_dir.glob("*.bin")) + list(snapshot_dir.glob("*.safetensors"))
        
        # Also check for sharded models (model-*.bin or model-*.safetensors)
        if not model_files:
            model_files = list(snapshot_dir.glob("model-*.bin")) + list(snapshot_dir.glob("model-*.safetensors"))
        
        # If still no model files, check if it's a very small placeholder (incomplete download)
        if model_files:
            # Check if files are actual model weights (not just placeholders)
            total_size = sum(f.stat().st_size for f in model_files)
            size_gb = total_size / (1024**3)
            
            # If total size is less than 1MB, it's likely incomplete
            if size_gb < 0.001:
                return False, str(cache_dir), 0.0
            
            return True, str(snapshot_dir), size_gb
        
        return False, str(cache_dir), 0.0
    
    def _load_model_directly(self, model_name: str):
        """Load model directly using transformers."""
        import torch
        from pathlib import Path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.is_nllb:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_class = AutoModelForSeq2SeqLM
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_class = AutoModelForCausalLM
        
        print(f"Loading translation model: {model_name}")
        if self.is_nllb:
            print(f"Using NLLB model for {self.source_lang_name} → {self.target_lang_name} translation")
        
        # Check cache status
        is_cached, cache_path, cache_size = self._check_model_cache(model_name)
        cache_dir = Path.home() / ".cache" / "huggingface"
        
        if is_cached:
            print(f"✓ Model found in cache ({cache_size:.2f}GB)")
            print(f"  Cache location: {cache_path}")
        else:
            print(f"⚠ Model not in cache - will download")
            print(f"  Cache location: {cache_dir}")
            # Estimate model size
            if "3.3B" in model_name:
                print(f"  Estimated size: ~13GB")
                print(f"  Download time: ~5-15 minutes (depending on connection)")
            elif "1.3B" in model_name:
                print(f"  Estimated size: ~5GB")
                print(f"  Download time: ~2-8 minutes (depending on connection)")
            elif "600M" in model_name or "distilled" in model_name:
                print(f"  Estimated size: ~2.5GB")
                print(f"  Download time: ~1-5 minutes (depending on connection)")
            else:
                print(f"  Download time: ~5-15 minutes (depending on connection)")
            print(f"  Note: Progress bar may take a moment to start updating...")
        print()
        
        # Enable verbose progress for downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Load tokenizer
        start_time = time.time()
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_cached)
            tokenizer_time = time.time() - start_time
            
            if is_cached and tokenizer_time < 1.0:
                print(f"✓ Tokenizer loaded from cache ({tokenizer_time:.1f}s)")
            elif tokenizer_time < 1.0:
                print(f"✓ Tokenizer downloaded and cached ({tokenizer_time:.1f}s)")
            else:
                print(f"✓ Tokenizer loaded ({tokenizer_time:.1f}s)")
        except Exception as e:
            if "not found" in str(e).lower() and is_cached:
                # Try downloading if cache check was wrong
                print("  Tokenizer not in cache, downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer_time = time.time() - start_time
                print(f"✓ Tokenizer downloaded and cached ({tokenizer_time:.1f}s)")
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                print(f"\n❌ Network error downloading tokenizer: {e}")
                print("Please check your internet connection and try again.")
                print("You may need to configure a proxy or use a VPN if HuggingFace is blocked.")
                raise
            else:
                raise
        print()
        
        # Try to load on GPU, fallback to CPU if OOM
        try:
            if self.device == "cuda":
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    if free_memory < 5 * 1024**3:
                        print(f"⚠ GPU memory low ({free_memory / 1024**3:.2f}GB free), using CPU for translation")
                        self.device = "cpu"
                    else:
                        print(f"✓ Using device: {self.device} (GPU acceleration enabled)")
                else:
                    print(f"⚠ Using device: {self.device} (CPU mode - slower)")
            else:
                print(f"⚠ Using device: {self.device} (CPU mode - slower)")
            
            # Load model weights
            if is_cached:
                print("Loading model weights from cache...")
            else:
                print("Downloading model weights...")
                print("  This may take several minutes. Progress will be shown below...")
            
            model_start_time = time.time()
            
            try:
                if self.device == "cuda":
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    self.model = model_class.from_pretrained(
                            model_name,
                        dtype=dtype,
                        device_map="auto",
                        local_files_only=is_cached
                    )
                else:
                    self.model = model_class.from_pretrained(
                        model_name,
                        local_files_only=is_cached
                    ).to(self.device)
                
                model_time = time.time() - model_start_time
                
                if is_cached:
                    print(f"✓ Model weights loaded from cache ({model_time:.1f}s)")
                else:
                    print(f"✓ Model weights downloaded and cached ({model_time:.1f}s)")
            except Exception as e:
                error_msg = str(e).lower()
                if ("not found" in error_msg or "file" in error_msg) and is_cached:
                    # Cache was incomplete, try downloading
                    print("  Model weights incomplete in cache, downloading missing files...")
                    if self.device == "cuda":
                        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        self.model = model_class.from_pretrained(
                            model_name,
                            dtype=dtype,
                            device_map="auto"
                        )
                    else:
                        self.model = model_class.from_pretrained(model_name).to(self.device)
                    model_time = time.time() - model_start_time
                    print(f"✓ Model weights downloaded and cached ({model_time:.1f}s)")
                elif "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                    print(f"\n❌ Network error downloading model weights: {e}")
                    print("\nTroubleshooting steps:")
                    print("1. Check internet connection: curl -I https://huggingface.co")
                    print("2. If behind a proxy, set: export HF_ENDPOINT=https://hf-mirror.com")
                    raise
                else:
                    raise
            
            print("✓ Translation model ready!")
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                print(f"⚠ GPU out of memory, falling back to CPU for translation")
                self.device = "cpu"
                self.model = model_class.from_pretrained(model_name).to(self.device)
                print("✓ Translation model loaded on CPU successfully!")
            else:
                raise

    def translate_text(self, text: str) -> str:
        """
        Translate a single text from source language to target language.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        if not text.strip():
            return ""

        if self.use_ollama:
            return self._translate_via_ollama(text)
        else:
            if self.is_nllb:
                return self._translate_with_nllb(text)
            else:
                return self._translate_directly(text)
    
    def _translate_via_ollama(self, text: str) -> str:
        """Translate using Ollama API."""
        # Create language-agnostic prompt
        prompt = f"Translate the following {self.source_lang_name} text to {self.target_lang_name}:\n{text}"
        
        # Optimize num_predict based on text length (subtitle segments are usually short)
        # Estimate max tokens needed: source text + translation (usually 1.5-2x source length)
        estimated_tokens = len(text.split()) * 2  # Rough estimate
        num_predict = min(max(estimated_tokens + 50, 64), 512)  # Between 64 and 512 tokens
        
        # Use lower temperature for more deterministic translation
        # qwen2.5 models work well with 0.1, qwen3 might need slightly different settings
        temperature = 0.05 if "qwen3" in self.ollama_model.lower() else 0.1
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": f"You are a professional translator. Translate {self.source_lang_name} text to {self.target_lang_name} accurately and naturally.",
            "stream": False,
            "keep_alive": "5m",  # Keep model in memory for 5 minutes to speed up subsequent requests
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": 0.9,  # Nucleus sampling for better quality
                "repeat_penalty": 1.1  # Reduce repetition
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"⚠ Ollama API error: {e}")
            raise
    
    def _translate_with_nllb(self, text: str) -> str:
        """Translate using NLLB model."""
        import torch
        
        # Set source and target languages for NLLB
        self.tokenizer.src_lang = self.source_nllb_code
        
        # Get target language token ID
        # NLLB uses language codes as special tokens
        try:
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_nllb_code)
            if tgt_lang_id == self.tokenizer.unk_token_id:
                # Fallback: try to get from tokenizer's language codes
                print(f"⚠ Warning: Could not find language code {self.target_nllb_code}, using default")
                # Try common alternatives
                if "zho" in self.target_nllb_code.lower():
                    tgt_lang_id = self.tokenizer.convert_tokens_to_ids("zho_Hans")
                elif "eng" in self.target_nllb_code.lower():
                    tgt_lang_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
                else:
                    # Use the tokenizer's default method
                    tgt_lang_id = None
        except:
            tgt_lang_id = None
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_length": 512,
                "num_beams": 3,
                "early_stopping": True
            }
            if tgt_lang_id is not None:
                generate_kwargs["forced_bos_token_id"] = tgt_lang_id
            
            translated_tokens = self.model.generate(**generate_kwargs)
        
        # Decode translation
        translation = self.tokenizer.batch_decode(
            translated_tokens, 
            skip_special_tokens=True
        )[0]
        
        return translation.strip()
    
    def _translate_directly(self, text: str) -> str:
        """Translate using Qwen model (direct loading)."""
        # Create language-agnostic translation prompt
        messages = [
            {
                "role": "system",
                "content": f"You are a professional translator. Translate {self.source_lang_name} text to {self.target_lang_name} accurately and naturally."
            },
            {
                "role": "user",
                "content": f"Translate the following {self.source_lang_name} text to {self.target_lang_name}:\n{text}"
            }
        ]

        # Apply chat template
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        # Generate translation
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Translate a list of subtitle segments.

        Args:
            segments: List of segments with 'text', 'start', 'end' keys

        Returns:
            List of segments with translated text
        """
        print(f"Translating {len(segments)} segments from {self.source_lang_name} to {self.target_lang_name}...")

        translated_segments = []
        for segment in tqdm(segments, desc="Translating"):
            translated_text = self.translate_text(segment["text"])
            translated_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text
            })

        print("Translation complete!")
        return translated_segments


# Backward compatibility alias
JapaneseToChinese = SubtitleTranslator


if __name__ == "__main__":
    # Example usage
    translator = SubtitleTranslator(source_lang="ja", target_lang="zh")

    # Test translation
    test_text = "こんにちは、世界！"
    result = translator.translate_text(test_text)
    print(f"Japanese: {test_text}")
    print(f"Chinese: {result}")
