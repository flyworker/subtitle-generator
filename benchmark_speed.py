#!/usr/bin/env python3
"""
Speed comparison benchmark between Ollama and HuggingFace models.
Tests actual translation speed for subtitle segments.
"""
import time
import requests
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from translate import SubtitleTranslator

def benchmark_model(model_name: str, test_texts: list, source_lang: str = "ja", target_lang: str = "zh"):
    """Benchmark a single model's translation speed."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*70}")
    
    # Initialize translator
    try:
        translator = SubtitleTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_name
        )
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Warm up - first request is usually slower
    print("Warming up (first translation may be slower)...")
    try:
        warmup_text = test_texts[0] if test_texts else "こんにちは"
        _ = translator.translate_text(warmup_text)
        print("✓ Warmup complete\n")
    except Exception as e:
        print(f"⚠ Warmup failed: {e}\n")
    
    # Benchmark translations
    times = []
    total_chars = 0
    successful = 0
    
    print(f"Translating {len(test_texts)} segments...")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        start = time.time()
        try:
            translated = translator.translate_text(text)
            elapsed = time.time() - start
            times.append(elapsed)
            total_chars += len(text)
            successful += 1
            status = "✓"
        except Exception as e:
            elapsed = time.time() - start
            times.append(float('inf'))
            status = "✗"
            print(f"  [{i:3d}/{len(test_texts)}] {status} {elapsed:.3f}s - Error: {e}")
            continue
        
        # Show progress for first few and every 10th
        if i <= 5 or i % 10 == 0 or i == len(test_texts):
            print(f"  [{i:3d}/{len(test_texts)}] {status} {elapsed:.3f}s - {len(text)} chars → {len(translated)} chars")
    
    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]
    
    if not valid_times:
        print("\n❌ All translations failed!")
        return None
    
    avg_time = sum(valid_times) / len(valid_times)
    total_time = sum(valid_times)
    min_time = min(valid_times)
    max_time = max(valid_times)
    chars_per_sec = total_chars / total_time if total_time > 0 else 0
    
    # Calculate percentiles
    sorted_times = sorted(valid_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    
    print("-" * 70)
    print(f"\nResults for {model_name}:")
    print(f"  Successful translations: {successful}/{len(test_texts)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per segment: {avg_time:.3f}s")
    print(f"  Median (p50): {p50:.3f}s")
    print(f"  95th percentile (p95): {p95:.3f}s")
    print(f"  Min time: {min_time:.3f}s")
    print(f"  Max time: {max_time:.3f}s")
    print(f"  Throughput: {chars_per_sec:.1f} chars/sec")
    print(f"  Segments per minute: {len(valid_times) / total_time * 60:.1f}")
    
    return {
        "model": model_name,
        "successful": successful,
        "total": len(test_texts),
        "total_time": total_time,
        "avg_time": avg_time,
        "median_time": p50,
        "p95_time": p95,
        "min_time": min_time,
        "max_time": max_time,
        "chars_per_sec": chars_per_sec,
        "segments_per_min": len(valid_times) / total_time * 60 if total_time > 0 else 0
    }

def main():
    # Test texts - typical Japanese subtitle segments
    test_texts = [
        "こんにちは、元気ですか？",
        "今日は良い天気ですね。",
        "映画を見に行きましょう。",
        "このレストランはとても美味しいです。",
        "明日の会議について話しましょう。",
        "ありがとうございます。",
        "すみません、もう一度言ってください。",
        "駅はどこですか？",
        "お願いします。",
        "さようなら。",
        "彼はとても親切な人です。",
        "この本は面白いですね。",
        "コーヒーを飲みましょう。",
        "時間がありません。",
        "後で連絡します。",
        "問題ありません。",
        "了解しました。",
        "お疲れ様でした。",
        "また明日。",
        "気をつけて。",
        # Add more realistic subtitle segments
        "彼女は昨日、新しい車を買いました。",
        "会議は午後3時から始まります。",
        "このプロジェクトは来月完成予定です。",
        "彼は医者になるために勉強しています。",
        "私たちは来週、旅行に行く予定です。",
        "この映画はとても感動的でした。",
        "彼は日本語を話すことができます。",
        "今日は忙しい一日でした。",
        "明日は早く起きる必要があります。",
        "この問題を解決するのは難しいです。"
    ]
    
    models = [
        "huihui_ai/qwen3-abliterated:8b-v2",  # Ollama
        "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"  # HuggingFace
    ]
    
    print("=" * 70)
    print("SPEED COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Testing {len(test_texts)} subtitle segments")
    print(f"Source: Japanese → Target: Chinese")
    print("=" * 70)
    
    results = []
    for model in models:
        result = benchmark_model(model, test_texts)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        # Identify which is which
        ollama_result = next((r for r in results if "qwen3" in r["model"] or "ollama" in r["model"].lower()), None)
        hf_result = next((r for r in results if "Qwen2.5" in r["model"] or "huggingface" in r["model"].lower()), None)
        
        if not ollama_result:
            ollama_result = results[0] if "qwen3" in results[0]["model"] else results[1]
        if not hf_result:
            hf_result = results[1] if ollama_result == results[0] else results[0]
        
        print(f"\n{'Metric':<30} {'Ollama (qwen3)':<20} {'HuggingFace (qwen2.5)':<20} {'Winner':<15}")
        print("-" * 85)
        
        # Average time
        avg_diff = ((ollama_result["avg_time"] / hf_result["avg_time"]) - 1) * 100
        winner = "Ollama" if ollama_result["avg_time"] < hf_result["avg_time"] else "HuggingFace"
        print(f"{'Avg time/segment':<30} {ollama_result['avg_time']:.3f}s{'':<12} {hf_result['avg_time']:.3f}s{'':<12} {winner}")
        
        # Median time
        median_diff = ((ollama_result["median_time"] / hf_result["median_time"]) - 1) * 100
        winner = "Ollama" if ollama_result["median_time"] < hf_result["median_time"] else "HuggingFace"
        print(f"{'Median time (p50)':<30} {ollama_result['median_time']:.3f}s{'':<12} {hf_result['median_time']:.3f}s{'':<12} {winner}")
        
        # Throughput
        throughput_diff = ((ollama_result["chars_per_sec"] / hf_result["chars_per_sec"]) - 1) * 100
        winner = "Ollama" if ollama_result["chars_per_sec"] > hf_result["chars_per_sec"] else "HuggingFace"
        print(f"{'Throughput (chars/sec)':<30} {ollama_result['chars_per_sec']:.1f}{'':<12} {hf_result['chars_per_sec']:.1f}{'':<12} {winner}")
        
        # Segments per minute
        spm_diff = ((ollama_result["segments_per_min"] / hf_result["segments_per_min"]) - 1) * 100
        winner = "Ollama" if ollama_result["segments_per_min"] > hf_result["segments_per_min"] else "HuggingFace"
        print(f"{'Segments per minute':<30} {ollama_result['segments_per_min']:.1f}{'':<12} {hf_result['segments_per_min']:.1f}{'':<12} {winner}")
        
        # Total time
        total_diff = ((ollama_result["total_time"] / hf_result["total_time"]) - 1) * 100
        winner = "Ollama" if ollama_result["total_time"] < hf_result["total_time"] else "HuggingFace"
        print(f"{'Total time ({len(test_texts)} segs)':<30} {ollama_result['total_time']:.2f}s{'':<12} {hf_result['total_time']:.2f}s{'':<12} {winner}")
        
        print("\n" + "=" * 70)
        print("SPEED DIFFERENCE")
        print("=" * 70)
        
        if ollama_result["avg_time"] < hf_result["avg_time"]:
            speedup = (hf_result["avg_time"] / ollama_result["avg_time"] - 1) * 100
            print(f"✓ Ollama model is {speedup:.1f}% FASTER on average")
        else:
            slowdown = (ollama_result["avg_time"] / hf_result["avg_time"] - 1) * 100
            print(f"⚠ Ollama model is {slowdown:.1f}% SLOWER on average")
        
        print(f"\nFor {len(test_texts)} segments:")
        time_saved = abs(hf_result["total_time"] - ollama_result["total_time"])
        if ollama_result["total_time"] < hf_result["total_time"]:
            print(f"  Ollama saves: {time_saved:.1f}s ({time_saved/60:.1f} minutes)")
        else:
            print(f"  HuggingFace saves: {time_saved:.1f}s ({time_saved/60:.1f} minutes)")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        if ollama_result["avg_time"] < hf_result["avg_time"]:
            print("✓ Use Ollama model for faster translation speed")
            print(f"  Speed advantage: {((hf_result['avg_time'] / ollama_result['avg_time']) - 1) * 100:.1f}%")
        else:
            print("✓ Use HuggingFace model for faster translation speed")
            print(f"  Speed advantage: {((ollama_result['avg_time'] / hf_result['avg_time']) - 1) * 100:.1f}%")
        print("\nNote: Consider VRAM usage (Ollama: ~5GB, HuggingFace: ~14GB)")
        print("      and quality requirements when choosing a model.")
    else:
        print(f"\n⚠ Could not complete comparison (only {len(results)} model(s) succeeded)")

if __name__ == "__main__":
    main()

