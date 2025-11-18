"""
Test script to demonstrate hallucination filtering functionality.

This script shows how the new filtering removes problematic segments.
NOTE: This is a standalone test that doesn't require the full environment.
"""
import json
from pathlib import Path
from typing import Any, Dict


def _is_hallucination(segment: Dict[str, Any], compression_threshold: float = 2.4) -> bool:
    """
    Detect if a segment is likely a hallucination based on Whisper quality metrics.
    (Copied from transcriber.py for standalone testing)
    """
    # Check compression ratio (high ratio = repetitive text)
    if segment.get('compression_ratio', 0) > compression_threshold:
        return True
    
    # Check for very high no_speech probability (likely silence or noise)
    if segment.get('no_speech_prob', 0) > 0.8:
        return True
    
    # Check for extremely repetitive patterns in text
    text = segment.get('text', '').strip()
    if len(text) > 50:
        # Check if text has very repetitive 3-word phrases
        words = text.split()
        if len(words) > 10:
            # Count unique 3-word phrases
            phrases = set()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrases.add(phrase)
            # If less than 20% unique phrases, likely hallucination
            if len(phrases) / (len(words) - 2) < 0.2:
                return True
    
    return False


def _filter_hallucinations(result: Dict[str, Any], compression_threshold: float = 2.4) -> Dict[str, Any]:
    """
    Filter out hallucinated segments from transcription result.
    (Copied from transcriber.py for standalone testing)
    """
    if 'segments' not in result:
        return result
    
    original_count = len(result['segments'])
    filtered_segments = []
    removed_segments = []
    
    for segment in result['segments']:
        if _is_hallucination(segment, compression_threshold):
            removed_segments.append(segment)
        else:
            filtered_segments.append(segment)
    
    # Rebuild full text from filtered segments
    filtered_text = ' '.join(seg['text'].strip() for seg in filtered_segments)
    
    result['segments'] = filtered_segments
    result['text'] = filtered_text
    result['hallucination_filter'] = {
        'enabled': True,
        'original_segment_count': original_count,
        'filtered_segment_count': len(filtered_segments),
        'removed_segment_count': len(removed_segments),
        'compression_threshold': compression_threshold
    }
    
    return result


def test_hallucination_detection():
    """Test the hallucination detection logic."""
    print("🔍 Testing Hallucination Detection\n")
    print("=" * 60)
    
    # Test case 1: High compression ratio (like the bug we saw)
    segment1 = {
        "text": " a school that is a school that is a school that is..." * 10,
        "compression_ratio": 27.5,
        "no_speech_prob": 0.65,
        "start": 3623.72,
        "end": 3653.72
    }
    
    print("\n✅ Test 1: High compression ratio segment")
    print(f"  Compression ratio: {segment1['compression_ratio']}")
    print(f"  No speech prob: {segment1['no_speech_prob']}")
    print(f"  Is hallucination: {_is_hallucination(segment1)}")
    assert _is_hallucination(segment1), "Should detect high compression ratio"
    
    # Test case 2: Normal segment
    segment2 = {
        "text": "This is a normal transcription with varied content.",
        "compression_ratio": 1.8,
        "no_speech_prob": 0.15,
        "start": 10.0,
        "end": 15.0
    }
    
    print("\n✅ Test 2: Normal segment")
    print(f"  Compression ratio: {segment2['compression_ratio']}")
    print(f"  No speech prob: {segment2['no_speech_prob']}")
    print(f"  Is hallucination: {_is_hallucination(segment2)}")
    assert not _is_hallucination(segment2), "Should not flag normal segment"
    
    # Test case 3: High no_speech_prob
    segment3 = {
        "text": "...",
        "compression_ratio": 1.5,
        "no_speech_prob": 0.85,
        "start": 100.0,
        "end": 105.0
    }
    
    print("\n✅ Test 3: High no_speech_prob segment")
    print(f"  Compression ratio: {segment3['compression_ratio']}")
    print(f"  No speech prob: {segment3['no_speech_prob']}")
    print(f"  Is hallucination: {_is_hallucination(segment3)}")
    assert _is_hallucination(segment3), "Should detect high no_speech_prob"
    
    print("\n" + "=" * 60)
    print("✅ All detection tests passed!\n")


def test_filter_results():
    """Test the filtering function on a mock result."""
    print("🔧 Testing Result Filtering\n")
    print("=" * 60)
    
    # Create a mock transcription result
    result = {
        "text": "Good segment. " + ("Bad segment. " * 50),
        "segments": [
            {
                "id": 0,
                "text": "Good segment.",
                "compression_ratio": 1.5,
                "no_speech_prob": 0.2,
                "start": 0.0,
                "end": 2.0
            },
            {
                "id": 1,
                "text": "Bad segment. " * 50,
                "compression_ratio": 28.0,
                "no_speech_prob": 0.7,
                "start": 2.0,
                "end": 5.0
            },
            {
                "id": 2,
                "text": "Another good segment.",
                "compression_ratio": 1.6,
                "no_speech_prob": 0.18,
                "start": 5.0,
                "end": 8.0
            }
        ],
        "language": "en"
    }
    
    print("\n📊 Original result:")
    print(f"  Total segments: {len(result['segments'])}")
    print(f"  Text length: {len(result['text'])} characters")
    
    # Filter the result
    filtered = _filter_hallucinations(result)
    
    print("\n📊 Filtered result:")
    print(f"  Total segments: {len(filtered['segments'])}")
    print(f"  Text length: {len(filtered['text'])} characters")
    print(f"  Removed segments: {filtered['hallucination_filter']['removed_segment_count']}")
    
    assert len(filtered['segments']) == 2, "Should have 2 segments after filtering"
    assert filtered['hallucination_filter']['removed_segment_count'] == 1
    assert "Bad segment" not in filtered['text'] or filtered['text'].count("Bad segment") < 5
    
    print("\n" + "=" * 60)
    print("✅ All filtering tests passed!\n")


def analyze_existing_file(filepath: str):
    """Analyze an existing JSON transcription file for hallucinations."""
    print(f"\n🔍 Analyzing file: {filepath}\n")
    print("=" * 60)
    
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    segments = result.get('segments', [])
    hallucinations = []
    
    for segment in segments:
        if _is_hallucination(segment):
            hallucinations.append(segment)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Hallucinated segments: {len(hallucinations)}")
    print(f"  Percentage: {len(hallucinations)/len(segments)*100:.1f}%")
    
    if hallucinations:
        print(f"\n⚠️  Found {len(hallucinations)} problematic segments:")
        for i, seg in enumerate(hallucinations[:5], 1):  # Show first 5
            print(f"\n  {i}. Segment {seg.get('id', '?')} at {seg.get('start', 0):.2f}s")
            print(f"     Compression ratio: {seg.get('compression_ratio', 0):.2f}")
            print(f"     No speech prob: {seg.get('no_speech_prob', 0):.2f}")
            print(f"     Text preview: {seg.get('text', '')[:100]}...")
        
        if len(hallucinations) > 5:
            print(f"\n  ... and {len(hallucinations) - 5} more")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VECINA Hallucination Filter Test Suite")
    print("=" * 60)
    
    # Run detection tests
    test_hallucination_detection()
    
    # Run filtering tests
    test_filter_results()
    
    # Analyze the problematic file if it exists
    problem_file = Path(__file__).parent.parent / "_data" / "_output" / "18c0b1432fe8829abf257468c79ea431.json"
    if problem_file.exists():
        analyze_existing_file(str(problem_file))
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60 + "\n")
