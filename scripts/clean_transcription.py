"""
Script to clean an existing transcription JSON file by filtering hallucinations.

This is useful for re-processing files that were transcribed without hallucination filtering.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _is_hallucination(segment: Dict[str, Any], compression_threshold: float = 2.4) -> bool:
    """Detect if a segment is likely a hallucination."""
    if segment.get('compression_ratio', 0) > compression_threshold:
        return True
    if segment.get('no_speech_prob', 0) > 0.8:
        return True

    text = segment.get('text', '').strip()
    if len(text) > 50:
        words = text.split()
        if len(words) > 10:
            phrases = set()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrases.add(phrase)
            if len(phrases) / (len(words) - 2) < 0.2:
                return True

    return False


def clean_transcription_file(input_path: str, output_path: str = None, compression_threshold: float = 2.4):
    """
    Clean a transcription JSON file by removing hallucinated segments.

    Args:
        input_path: Path to the input JSON file
        output_path: Path for the cleaned output (default: input_path with .cleaned.json)
        compression_threshold: Maximum acceptable compression ratio
    """
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"❌ Error: File not found: {input_path}")
        return

    # Load the transcription
    print(f"📂 Loading: {input_file.name}")
    with open(input_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    segments = result.get('segments', [])
    original_count = len(segments)

    print(
        f"📊 Original: {original_count} segments, {len(result.get('text', ''))} characters")

    # Filter hallucinations
    filtered_segments = []
    removed_segments = []

    for segment in segments:
        if _is_hallucination(segment, compression_threshold):
            removed_segments.append(segment)
            print(f"  🗑️  Removing segment {segment.get('id', '?')} at {segment.get('start', 0):.2f}s "
                  f"(compression: {segment.get('compression_ratio', 0):.2f})")
        else:
            filtered_segments.append(segment)

    # Rebuild text
    filtered_text = ' '.join(seg['text'].strip() for seg in filtered_segments)

    # Update result
    result['segments'] = filtered_segments
    result['text'] = filtered_text
    result['hallucination_filter'] = {
        'enabled': True,
        'original_segment_count': original_count,
        'filtered_segment_count': len(filtered_segments),
        'removed_segment_count': len(removed_segments),
        'compression_threshold': compression_threshold
    }

    # Determine output path
    if output_path is None:
        output_file = input_file.with_suffix('.cleaned.json')
    else:
        output_file = Path(output_path)

    # Save cleaned version
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"\n✅ Cleaned: {len(filtered_segments)} segments, {len(filtered_text)} characters")
    print(
        f"🗑️  Removed: {len(removed_segments)} segments ({len(removed_segments)/original_count*100:.1f}%)")
    print(f"💾 Saved to: {output_file.name}")

    # Also save a clean text file
    text_output = output_file.with_suffix('.txt')
    with open(text_output, 'w', encoding='utf-8') as f:
        f.write(filtered_text)
    print(f"💾 Text saved to: {text_output.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python clean_transcription.py <input_file.json> [output_file.json]")
        print("\nExample:")
        print("  python scripts/clean_transcription.py _data/_output/transcript.json")
        print("  python scripts/clean_transcription.py _data/_output/transcript.json _data/_output/cleaned.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print("\n" + "=" * 60)
    print("  VECINA Transcription Cleaner")
    print("=" * 60 + "\n")

    clean_transcription_file(input_file, output_file)

    print("\n" + "=" * 60)
