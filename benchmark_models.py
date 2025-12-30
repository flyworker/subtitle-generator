#!/usr/bin/env python3
"""
Benchmark script to compare translation speed between different Ollama models.
"""
import time
import requests
from typing import List

def benchmark_model(model_name: str, test_texts: List[str], ollama_url: str = "http://localhost:11434"):
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Warm up - first request is usually slower
    print("Warming up model...")
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Translate to Chinese: Hello",
                "options": {"num_predict": 10, "temperature": 0.1},
                "keep_alive": "5m"
            },
            timeout=30
        )
    except:
        pass
    
    times = []
    total_chars = 0
    
    for i, text in enumerate(test_texts, 1):
        prompt = f"Translate the following Japanese text to Chinese:\n{text}"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": "You are a professional translator. Translate Japanese text to Chinese accurately and naturally.",
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.05 if "qwen3" in model_name.lower() else 0.1,
                "num_predict": min(max(len(text.split()) * 2 + 50, 64), 512),
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        start = time.time()
        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            elapsed = time.time() - start
            times.append(elapsed)
            total_chars += len(text)
            print(f"  [{i}/{len(test_texts)}] {elapsed:.2f}s - {len(text)} chars")
        except Exception as e:
            print(f"  [{i}/{len(test_texts)}] Error: {e}")
            times.append(float('inf'))
    
    if times and all(t != float('inf') for t in times):
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        chars_per_sec = total_chars / total_time if total_time > 0 else 0
        print(f"\nResults for {model_name}:")
        print(f"  Average time per segment: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {chars_per_sec:.1f} chars/sec")
        return {
            "model": model_name,
            "avg_time": avg_time,
            "total_time": total_time,
            "chars_per_sec": chars_per_sec
        }
    return None

def main():
    # Test texts (typical subtitle segments)
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
        "さようなら。"
    ]
    
    models = [
        "huihui_ai/qwen2.5-vl-abliterated:7b",
        "huihui_ai/qwen3-abliterated:8b-v2"
    ]
    
    results = []
    for model in models:
        result = benchmark_model(model, test_texts)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        qwen25 = results[0] if "qwen2.5" in results[0]["model"] else results[1]
        qwen3 = results[1] if "qwen3" in results[1]["model"] else results[0]
        
        speed_diff = (qwen3["avg_time"] / qwen25["avg_time"] - 1) * 100
        print(f"\nQwen2.5: {qwen25['avg_time']:.2f}s avg, {qwen25['chars_per_sec']:.1f} chars/sec")
        print(f"Qwen3:   {qwen3['avg_time']:.2f}s avg, {qwen3['chars_per_sec']:.1f} chars/sec")
        print(f"\nQwen3 is {abs(speed_diff):.1f}% {'slower' if speed_diff > 0 else 'faster'} than Qwen2.5")

if __name__ == "__main__":
    main()



