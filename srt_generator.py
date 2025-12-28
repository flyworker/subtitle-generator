"""
Generate SRT subtitle files from transcribed and translated segments.
"""
from typing import List, Dict
import os


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: List[Dict], output_path: str):
    """
    Generate an SRT subtitle file from segments.

    Args:
        segments: List of subtitle segments with 'start', 'end', 'text'
        output_path: Path to output .srt file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            # Subtitle index
            f.write(f"{i}\n")

            # Timestamp
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Subtitle text
            f.write(f"{segment['text']}\n")

            # Blank line between subtitles
            f.write("\n")

    print(f"SRT file saved to: {output_path}")


def generate_dual_srt(source_segments: List[Dict],
                      target_segments: List[Dict],
                      output_path: str):
    """
    Generate an SRT file with both source and target language subtitles.

    Args:
        source_segments: Source language subtitle segments
        target_segments: Target language subtitle segments
        output_path: Path to output .srt file
    """
    if len(source_segments) != len(target_segments):
        raise ValueError("Source and target segments must have the same length")

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (src_seg, tgt_seg) in enumerate(zip(source_segments, target_segments), start=1):
            # Subtitle index
            f.write(f"{i}\n")

            # Timestamp
            start_time = format_timestamp(src_seg['start'])
            end_time = format_timestamp(src_seg['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Subtitle text (source language on first line, target language on second)
            f.write(f"{src_seg['text']}\n")
            f.write(f"{tgt_seg['text']}\n")

            # Blank line between subtitles
            f.write("\n")

    print(f"Dual-language SRT file saved to: {output_path}")


# Backward compatibility alias
def generate_dual_srt_legacy(japanese_segments: List[Dict],
                              chinese_segments: List[Dict],
                              output_path: str):
    """Legacy function name for backward compatibility."""
    return generate_dual_srt(japanese_segments, chinese_segments, output_path)


if __name__ == "__main__":
    # Example usage
    test_segments = [
        {"start": 0.0, "end": 2.5, "text": "你好，世界！"},
        {"start": 2.5, "end": 5.0, "text": "这是一个测试字幕。"},
        {"start": 5.0, "end": 8.0, "text": "欢迎使用自动字幕生成器。"}
    ]

    generate_srt(test_segments, "test_output.srt")
    print("Test SRT file generated!")
