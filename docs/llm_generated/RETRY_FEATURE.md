# Hallucination Retry Feature

## Overview
Instead of just removing hallucinated segments, the system can now **retry transcribing them** with different parameters to try to recover the actual audio content.

## How It Works

### Detection Phase
When a segment is flagged as a hallucination (compression ratio > 2.4, repetitive text, high no_speech_prob):

### Retry Phase (if enabled)
1. **Extract segment audio** - Pull out the specific time range (e.g., 30 seconds)
2. **Retry with optimized parameters**:
   - Temperature: `1.0` (higher = more variation, less repetition)
   - Context: `condition_on_previous_text=False` (fresh start, no influence from bad segments)
   - Thresholds: Stricter compression ratio (2.0) and confidence checks
3. **Quality check** - Compare new vs old:
   - Must have compression ratio < 2.4 (not hallucinating)
   - Must be at least 30% better than original
4. **Decision**:
   - ✅ **Better result**: Replace with new transcription, mark as `retried: true`
   - ❌ **Still bad/failed**: Remove segment entirely

## Usage

### CLI

```bash
# Single file with retry
vecina-transcriber transcribe audio.mp3 --retry-hallucinations

# Batch processing with retry
vecina-transcriber batch _data/podcasts \
  --retry-hallucinations \
  --chunk-duration 600

# Combined options
vecina-transcriber transcribe audio.mp3 \
  --model large-v3 \
  --chunk-duration 600 \
  --retry-hallucinations
```

### Python API

```python
from vecina_transcriber.transcriber import EnglishTranscriber

transcriber = EnglishTranscriber(model_name="base")

# With retry
result = transcriber.transcribe(
    "audio.mp3",
    filter_hallucinations=True,
    retry_hallucinations=True  # Enable retry
)

# Chunked with retry
result = transcriber.transcribe_chunked(
    "long_audio.mp3",
    chunk_duration_seconds=600,
    retry_hallucinations=True
)
```

### Modal Deployment

```python
from vecina_transcriber.modal_entrypoint import transcribe_with_modal

result = transcribe_with_modal(
    "audio.mp3",
    model_name="base",
    retry_hallucinations=True,
    chunk_duration_seconds=600
)
```

### Configuration File

```json
{
  "model_name": "base",
  "retry_hallucinations": true,
  "filter_hallucinations": true,
  "chunk_duration_seconds": 600,
  "overrides": {
    "important_podcast.mp3": {
      "retry_hallucinations": true,
      "model_name": "large-v3"
    }
  }
}
```

## Output

### Result Metadata
```json
{
  "text": "Transcription with recovered segments...",
  "segments": [
    {
      "id": 42,
      "text": "Successfully recovered text",
      "start": 120.5,
      "end": 125.0,
      "compression_ratio": 1.8,
      "retried": true  // Marks segments that were retried
    }
  ],
  "hallucination_filter": {
    "enabled": true,
    "retry_enabled": true,
    "original_segment_count": 400,
    "retried_segment_count": 8,    // Successfully recovered
    "removed_segment_count": 3,     // Still had to remove
    "filtered_segment_count": 397   // Final count
  }
}
```

### CLI Output
```
[1/5] Transcribing: podcast.mp3
  💾 Saved: podcast.txt
  💾 Saved: podcast.json
  🔄 Retried 8 segments
  🗑️  Removed 3 segments
  ✅ Complete (15,234 chars, 397 segments)
```

## Performance Considerations

### Speed Impact
- **Per segment**: ~1-2 seconds additional processing
- **Example**: File with 10 hallucinations = +10-20 seconds
- **GPU recommended**: CPU retry is very slow (~10-30 seconds per segment)

### Resource Usage
- **Memory**: Same as regular transcription (model stays loaded)
- **Disk**: Temporary files for extracted segments (auto-cleaned)
- **GPU**: Requires GPU access for re-transcription

### Optimization Tips
1. **Use chunking**: `--chunk-duration 600` reduces hallucinations overall
2. **Better models**: `large-v3` produces fewer hallucinations to begin with
3. **Selective retry**: Enable only for problem files via config overrides

## When to Use

### ✅ Use Retry When:
- **Critical content**: Every word matters (interviews, legal, medical)
- **Quality issues**: Background noise, silence, poor recording
- **Known problems**: Files that previously had many hallucinations
- **Accuracy > Speed**: Willing to wait longer for better results

### ❌ Skip Retry When:
- **Batch processing**: Large volumes where speed matters
- **High quality audio**: Clean recordings with few issues
- **CPU only**: Too slow without GPU
- **Filtering sufficient**: Simple removal good enough

## Examples

### Example 1: Recovered Segment
**Original (hallucinated)**:
```
"a school that is a school that is a school..." (compression: 27.5)
```

**After retry**:
```
"The Rhode Island School of Design is a prestigious institution..." (compression: 1.8)
```
✅ **Success**: 30% improvement, actual content recovered

### Example 2: Still Bad
**Original**:
```
"thank you thank you thank you..." (compression: 15.2)
```

**After retry**:
```
"thanks thanks thanks..." (compression: 12.8)
```
❌ **Removed**: Still repetitive, no meaningful improvement

### Example 3: Actual Silence
**Original**:
```
"..." (no_speech_prob: 0.95)
```

**After retry**:
```
No improvement possible (actually silence)
```
❌ **Removed**: Nothing to transcribe

## Statistics

Based on testing with the problematic file (18c0b1432fe8829abf257468c79ea431.json):

| Metric | Value |
|--------|-------|
| Total hallucinations detected | 66 |
| Typical retry success rate | ~10-30% |
| Expected recovery | 7-20 segments |
| Processing time increase | +60-120 seconds |
| Quality improvement | Moderate to High |

## Best Practices

### 1. Progressive Approach
```bash
# Step 1: Try without retry first
vecina-transcriber transcribe audio.mp3

# Step 2: If many hallucinations, enable retry
vecina-transcriber transcribe audio.mp3 --retry-hallucinations
```

### 2. Use Configuration for Selective Retry
```json
{
  "filter_hallucinations": true,
  "retry_hallucinations": false,  // Default off
  "overrides": {
    "important/*.mp3": {
      "retry_hallucinations": true  // Enable only for important files
    }
  }
}
```

### 3. Monitor Results
```python
result = transcriber.transcribe("audio.mp3", retry_hallucinations=True)

if 'hallucination_filter' in result:
    stats = result['hallucination_filter']
    retried = stats['retried_segment_count']
    removed = stats['removed_segment_count']
    
    print(f"Recovery rate: {retried}/{retried+removed} segments")
```

### 4. Combine with Chunking
```bash
# Best results: chunk + retry + good model
vecina-transcriber transcribe long_audio.mp3 \
  --model large-v3 \
  --chunk-duration 600 \
  --retry-hallucinations
```

## Technical Details

### Retry Parameters
- **Temperature**: `1.0` (vs default `0.0`) - Increases randomness
- **Context**: `condition_on_previous_text=False` - No contamination
- **Compression threshold**: `2.0` (vs `2.4`) - Stricter on retry
- **No speech threshold**: `0.5` (vs `0.6`) - More sensitive
- **Logprob threshold**: `-0.5` (vs `-1.0`) - Higher confidence required

### Success Criteria
```python
new_compression < old_compression * 0.7 and new_compression < 2.4
```
- Must be at least 30% better
- Must be below normal threshold

### Failure Handling
- Extraction errors: Log and skip to removal
- Transcription errors: Log and skip to removal
- Poor quality retry: Compare and remove if not better

## Limitations

1. **Not magic**: Can't recover audio that isn't there (actual silence)
2. **GPU dependent**: Slow/impractical on CPU
3. **No guarantee**: Some hallucinations can't be fixed
4. **Time cost**: Adds processing time per attempt
5. **Same model**: Uses same Whisper model (not a different AI)

## Future Improvements

Potential enhancements (not yet implemented):
- Multiple retry strategies
- Different models for retry
- Confidence-based retry decisions
- Parallel retry processing
- Custom retry parameters per file

## See Also
- **HALLUCINATION_FIX.md** - Overview of filtering system
- **QUICKSTART.md** - Quick usage guide
- **scripts/retry_demo.py** - Usage examples
