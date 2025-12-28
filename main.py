#!/usr/bin/env python3
"""
Main script for generating Chinese subtitles from Japanese videos.

This script:
1. Transcribes Japanese audio from video using Whisper
2. Translates Japanese text to Chinese using NLLB
3. Generates .srt subtitle file for VLC player
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from transcribe import JapaneseTranscriber
from translate import JapaneseToChinese
from srt_generator import generate_srt


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chinese subtitles from Japanese video"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to Japanese video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .srt file path (default: same as video with .zh.srt extension)"
    )
    parser.add_argument(
        "-w", "--whisper-model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "-t", "--translation-model",
        type=str,
        default="facebook/nllb-200-3.3B",
        help="Translation model name (default: facebook/nllb-200-3.3B)"
    )
    parser.add_argument(
        "--keep-japanese",
        action="store_true",
        help="Keep original Japanese text in output SRT file"
    )
    parser.add_argument(
        "--force-retranscribe",
        action="store_true",
        help="Force re-transcription even if cached transcription exists"
    )

    args = parser.parse_args()

    # Validate video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        video_path = Path(args.video_path)
        output_path = str(video_path.with_suffix('')) + '.zh.srt'
    else:
        output_path = args.output

    print("=" * 60)
    print("Japanese Video → Chinese Subtitle Generator")
    print("=" * 60)
    print(f"Video: {args.video_path}")
    print(f"Output: {output_path}")
    print(f"Whisper Model: {args.whisper_model}")
    print(f"Translation Model: {args.translation_model}")
    print("=" * 60)
    print()
    
    # Start total timer
    total_start_time = time.time()

    # Determine temp file path for transcription cache
    video_path_obj = Path(args.video_path)
    temp_file = video_path_obj.parent / f".{video_path_obj.stem}.ja.{args.whisper_model}.transcription.json"
    
    # Always check for cached transcription first
    japanese_segments = None
    cache_found = False
    step1_time = 0.0
    
    if temp_file.exists() and not args.force_retranscribe:
        try:
            print(f"Checking for cached transcription: {temp_file.name}")
            cache_start_time = time.time()
            with open(temp_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache: check if video path and settings match
            if (cache_data.get('video_path') == args.video_path and
                cache_data.get('whisper_model') == args.whisper_model):
                japanese_segments = cache_data.get('segments', [])
                cache_found = True
                step1_time = time.time() - cache_start_time
                print(f"✓ Found cached transcription ({len(japanese_segments)} segments)")
                print(f"  Cache file: {temp_file}")
                print(f"  Cache load time: {step1_time:.1f}s")
                print(f"  Skipping Step 1 (transcription), starting from Step 2 (translation)")
                print(f"  To force re-transcription, use --force-retranscribe")
            else:
                print(f"⚠ Cache mismatch (different video/settings), will re-transcribe")
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            print(f"  Will re-transcribe")
    
    # Step 1: Transcribe Japanese audio (only if no valid cache found)
    if not cache_found:
        print("STEP 1: Transcribing Japanese audio...")
        print("-" * 60)
        step1_start_time = time.time()
        
        # Free GPU memory from Ollama before transcription
        print("Freeing GPU memory from Ollama...")
        JapaneseToChinese.unload_ollama()
        print()
        
        transcriber = JapaneseTranscriber(model_size=args.whisper_model, language="ja")
        japanese_segments = transcriber.transcribe(args.video_path, language="ja")
        
        # Save transcription to temp file
        try:
            cache_data = {
                'video_path': args.video_path,
                'whisper_model': args.whisper_model,
                'segments': japanese_segments
            }
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Transcription saved to cache: {temp_file.name}")
        except Exception as e:
            print(f"⚠ Warning: Could not save transcription cache: {e}")
        
        # Free GPU memory before loading translation model
        transcriber.cleanup()
        del transcriber
        
        step1_time = time.time() - step1_start_time
        print(f"✓ Step 1 completed in {step1_time:.1f}s ({step1_time/60:.1f} minutes)")
        print()
    
    print()

    # Step 2: Translate to Chinese
    print("STEP 2: Translating to Chinese...")
    print("-" * 60)
    step2_start_time = time.time()
    translator = None
    try:
        translator = JapaneseToChinese(model_name=args.translation_model)
        chinese_segments = translator.translate_segments(japanese_segments)
        step2_time = time.time() - step2_start_time
        print(f"✓ Step 2 completed in {step2_time:.1f}s ({step2_time/60:.1f} minutes)")
        print()

        # Step 3: Generate SRT file
        print("STEP 3: Generating subtitle file...")
        print("-" * 60)
        step3_start_time = time.time()

        if args.keep_japanese:
            from srt_generator import generate_dual_srt
            generate_dual_srt(japanese_segments, chinese_segments, output_path)
        else:
            generate_srt(chinese_segments, output_path)
        
        step3_time = time.time() - step3_start_time
        print(f"✓ Step 3 completed in {step3_time:.1f}s")
        print()

        # Calculate total time
        total_time = time.time() - total_start_time
        
        print()
        print("=" * 60)
        print("✓ SUCCESS!")
        print("=" * 60)
        print(f"Chinese subtitle file created: {output_path}")
        print()
        print("Time Summary:")
        print(f"  Step 1 (Transcription): {step1_time:.1f}s ({step1_time/60:.1f} min)")
        print(f"  Step 2 (Translation):   {step2_time:.1f}s ({step2_time/60:.1f} min)")
        print(f"  Step 3 (SRT Generation): {step3_time:.1f}s")
        print(f"  Total Time:             {total_time:.1f}s ({total_time/60:.1f} min)")
        print()
        print("To use with VLC player:")
        print(f"1. Place the .srt file in the same folder as your video")
        print(f"2. Make sure the .srt filename matches the video filename")
        print(f"3. Open the video in VLC - subtitles will load automatically!")
        print()
        print("Alternatively, in VLC:")
        print(f"  Subtitle → Add Subtitle File... → Select {output_path}")
        print("=" * 60)
    finally:
        # Clean up: Free GPU memory from translation model and Ollama
        print("Cleaning up GPU memory...")
        if translator is not None:
            if hasattr(translator, 'use_ollama') and translator.use_ollama:
                JapaneseToChinese.unload_ollama()
            del translator
        print()


if __name__ == "__main__":
    main()
