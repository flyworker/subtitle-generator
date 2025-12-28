"""
Multi-language audio transcription using OpenAI Whisper with GPU acceleration.
Uses local Whisper models - no API calls, fully offline.
"""
import whisper
import torch
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class Transcriber:
    """
    Transcribe audio to text using local Whisper model.
    
    This class uses the openai-whisper library which runs models locally
    on your machine. Models are downloaded once and cached locally.
    No API keys or internet connection required after initial download.
    """

    def __init__(self, model_size: str = "medium", language: Optional[str] = None):
        """
        Initialize the local Whisper model.

        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
                       medium is a good balance for most languages
            language: Language code (e.g., "ja", "en", "es", "fr"). If None, auto-detect.
                       
        Note: Models are downloaded and cached locally on first use.
              Subsequent runs will use the cached model.
        """
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ Using device: {self.device} (GPU acceleration enabled)")
        else:
            print(f"⚠ Using device: {self.device} (CPU mode - slower)")
        
        print("Loading Whisper model...")
        print(f"Model: {model_size}")
        
        # Check if model is cached
        import os
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "whisper"
        model_file = cache_dir / f"{model_size}.pt"
        
        if model_file.exists():
            file_size = model_file.stat().st_size / (1024**3)  # Size in GB
            print(f"✓ Model found in cache ({file_size:.2f}GB)")
            print(f"  Cache location: {cache_dir}")
        else:
            print(f"⚠ Model not in cache - will download on first use")
            print(f"  Cache location: {cache_dir}")
            print(f"  Note: Download may take a few minutes depending on connection speed")
        
        # whisper.load_model downloads and caches the model locally if not present
        # All processing happens on your local machine/GPU
        import time
        start_time = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - start_time
        
        if load_time < 2.0:
            print(f"✓ Model loaded from cache ({load_time:.1f}s)")
        else:
            print(f"✓ Model loaded ({load_time:.1f}s)")

    def transcribe(self, video_path: str, language: Optional[str] = None) -> List[Dict]:
        """
        Transcribe audio from video file using local Whisper model.

        Args:
            video_path: Path to video file
            language: Language code (e.g., "ja", "en", "es", "fr"). 
                     If None, uses the language set during initialization or auto-detects.

        Returns:
            List of subtitle segments with text, start, and end times
            
        Note: All processing happens locally - no API calls or internet required.
        """
        # Use provided language, instance language, or None (auto-detect)
        transcribe_language = language or self.language
        
        if transcribe_language:
            print(f"Transcribing {video_path} in {transcribe_language}...")
        else:
            print(f"Transcribing {video_path} (auto-detecting language)...")

        # Local transcription - runs entirely on your machine/GPU
        result = self.model.transcribe(
            video_path,
            language=transcribe_language,
            task="transcribe",
            verbose=False
        )

        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })

        print(f"Transcription complete! Found {len(segments)} segments.")
        return segments
    
    def cleanup(self):
        """
        Free GPU memory by deleting the model and clearing cache.
        Call this after transcription is complete to free up GPU for other models.
        """
        if hasattr(self, 'model'):
            del self.model
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ Whisper model unloaded, GPU memory freed")


# Backward compatibility alias
JapaneseTranscriber = Transcriber


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_file> [language_code]")
        sys.exit(1)

    video_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    transcriber = Transcriber(language=language)
    segments = transcriber.transcribe(video_file)

    for seg in segments[:5]:  # Print first 5 segments
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
