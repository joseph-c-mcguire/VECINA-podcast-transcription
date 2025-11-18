# Quick Start: Hallucination Filtering

## ✅ What Was Fixed
The transcription was producing repetitive gibberish like `"a school that is a school that is..."` 66 times in one file. This is now automatically filtered out.

## 🚀 Quick Usage

### Option 1: CLI - Single File
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Transcribe with auto-filtering (default)
vecina-transcriber transcribe audio.mp3 --chunk-duration 600

# Or use the batch command for multiple files
vecina-transcriber batch _data/podcasts --output-dir _data/_output
```

### Option 2: Clean Existing Files
```bash
# Clean a transcription that already has hallucinations
python scripts/clean_transcription.py _data/_output/transcript.json
```

### Option 3: Modal (Cloud Processing)
The modal pipeline automatically uses filtering now:
```bash
scripts\run_transcription_pipeline.sh
```

## 📊 What Changed

### Before
```json
{
  "segments": [
    {...},
    {
      "text": "a school that is a school that is..." (repeated 50+ times),
      "compression_ratio": 27.5,  // Way too high!
      "no_speech_prob": 0.65
    }
  ]
}
```

### After
```json
{
  "segments": [
    {...}  // Hallucinated segment removed
  ],
  "hallucination_filter": {
    "removed_segment_count": 66,
    "filtered_segment_count": 407
  }
}
```

## 🎯 Key Features
- ✅ **Auto-enabled**: Filtering works by default
- ✅ **No setup needed**: Just update your code and run
- ✅ **Transparent**: Reports what was filtered
- ✅ **Configurable**: Can disable with `--no-filter-hallucinations`
- ✅ **Batch-friendly**: Works with all processing modes

## 🛠️ New Commands

### Batch Processing
```bash
# Process entire directory
vecina-transcriber batch _data/podcasts

# With custom settings
vecina-transcriber batch _data/podcasts \
  --model large-v3 \
  --chunk-duration 600 \
  --pattern "*.mp3"
```

### Clean Existing Files
```bash
# Clean a single file
python scripts/clean_transcription.py _data/_output/bad_transcript.json

# Test the filtering
python scripts/test_hallucination_filtering.py
```

## 📚 Documentation
- **Full details**: See `HALLUCINATION_FIX.md`
- **API usage**: Check updated docstrings in `transcriber.py`
- **Configuration**: Edit `_data/config.json` for custom settings

## 🔧 Files Modified
1. **vecina_transcriber/transcriber.py** - Core filtering logic
2. **vecina_transcriber/cli.py** - Added `batch` command + filtering flags
3. **vecina_transcriber/modal_entrypoint.py** - Modal integration with filtering

## ✨ Example Output
```
[1/3] Transcribing: podcast.mp3
  💾 Saved: podcast.txt
  💾 Saved: podcast.json
  🔍 Filtered 15 hallucinated segments
  ✅ Complete (12,450 chars, 287 segments)
```

## 💡 Pro Tips
1. Use `--chunk-duration 600` for long files (better accuracy)
2. Check `hallucination_filter` in JSON output to see what was removed
3. Use the batch command for processing multiple files efficiently
4. Run `clean_transcription.py` on old files with hallucinations

## ⚠️ Important
- Old transcription files are NOT automatically updated
- Re-run transcription or use `clean_transcription.py` to fix them
- The filtering is conservative - it only removes obvious hallucinations
