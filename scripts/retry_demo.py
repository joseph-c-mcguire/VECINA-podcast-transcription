"""
Demo script to show the retry functionality for hallucinated segments.

This script demonstrates how to use the retry feature.
"""
from pathlib import Path

# Simple demo showing how to use the retry feature
example_usage = """
# Example Usage of Retry Functionality

## CLI Usage

### Single File with Retry
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Transcribe with retry enabled
vecina-transcriber transcribe audio.mp3 --retry-hallucinations

# With chunking for better accuracy
vecina-transcriber transcribe audio.mp3 \\
  --chunk-duration 600 \\
  --retry-hallucinations
```

### Batch Processing with Retry
```bash
# Process directory with retry
vecina-transcriber batch _data/podcasts \\
  --retry-hallucinations \\
  --chunk-duration 600

# Full example
vecina-transcriber batch _data/podcasts \\
  --output-dir _data/_output \\
  --model large-v3 \\
  --chunk-duration 600 \\
  --retry-hallucinations \\
  --pattern "*.mp3"
```

## Python API Usage

```python
from vecina_transcriber.transcriber import EnglishTranscriber

transcriber = EnglishTranscriber(model_name="base")

# Standard: Remove hallucinations
result = transcriber.transcribe(
    "audio.mp3",
    filter_hallucinations=True,
    retry_hallucinations=False  # Default
)

# With Retry: Try to fix hallucinations first
result = transcriber.transcribe(
    "audio.mp3",
    filter_hallucinations=True,
    retry_hallucinations=True  # Enable retry
)

# Check results
if 'hallucination_filter' in result:
    stats = result['hallucination_filter']
    print(f"Retried: {stats.get('retried_segment_count', 0)}")
    print(f"Removed: {stats.get('removed_segment_count', 0)}")
```

## Modal Deployment

```python
from vecina_transcriber.modal_entrypoint import transcribe_with_modal

result = transcribe_with_modal(
    "audio.mp3",
    model_name="base",
    retry_hallucinations=True,  # Enable retry
    chunk_duration_seconds=600
)
```

## Configuration (config.json)

```json
{
  "model_name": "base",
  "language": "en",
  "filter_hallucinations": true,
  "retry_hallucinations": true,
  "chunk_duration_seconds": 600,
  "overrides": {
    "problematic_file.mp3": {
      "retry_hallucinations": true,
      "model_name": "large-v3"
    }
  }
}
```

## How It Works

When a hallucinated segment is detected (high compression ratio, repetitive text):

1. **Extract segment audio** - Pull out just that time range
2. **Retry with different parameters**:
   - Temperature: 1.0 (higher variation)
   - No context from previous segments
   - Stricter thresholds
3. **Check if better** - Compare compression ratios
4. **Accept or remove**:
   - If retry is better (< 70% of original compression): Use it ✅
   - If retry still bad or failed: Remove segment ❌

## Output Example

```json
{
  "text": "Cleaned and retried transcription...",
  "segments": [
    {
      "text": "Successfully retried segment",
      "retried": true,
      "compression_ratio": 1.8
    }
  ],
  "hallucination_filter": {
    "enabled": true,
    "retry_enabled": true,
    "retried_segment_count": 5,
    "removed_segment_count": 2,
    "original_segment_count": 400
  }
}
```

## CLI Output

```
[1/3] Transcribing: podcast.mp3
  💾 Saved: podcast.txt
  💾 Saved: podcast.json
  🔄 Retried 5 segments
  🗑️  Removed 2 segments
  ✅ Complete (12,450 chars, 287 segments)
```

## Performance Notes

- **Retry adds processing time** (~1-2 seconds per hallucinated segment)
- **GPU required** - Needs access to Whisper model for re-transcription
- **Best for**: Files with known hallucination issues
- **Not needed if**: Using high-quality audio with few issues

## When to Use Retry

✅ **Use retry when:**
- Processing important audio where every word matters
- Audio has quality issues (background noise, silence)
- Previous transcriptions had many hallucinations
- Willing to trade speed for accuracy

❌ **Don't use retry when:**
- Processing large batches quickly
- Audio is high quality with few issues
- Running on CPU only (very slow)
- Simple filtering is sufficient

## Tips

1. **Start without retry** - See if simple filtering is enough
2. **Enable for problem files** - Use overrides in config.json
3. **Monitor stats** - Check `retried_segment_count` in output
4. **Combine with chunking** - Better results with `--chunk-duration 600`
5. **Use better models** - `large-v3` with retry gives best results
"""

if __name__ == "__main__":
    print("=" * 70)
    print("  Hallucination Retry Feature - Usage Guide")
    print("=" * 70)
    print(example_usage)
    print("=" * 70)
    print("\n📚 For more details, see HALLUCINATION_FIX.md\n")
