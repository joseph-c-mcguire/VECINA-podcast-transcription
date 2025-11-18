"""
Upload audio files from local _data directory to Modal volume for transcription.

Usage:
    modal run scripts/upload_to_modal.py --audio-dir _data
"""

import modal
from pathlib import Path

app = modal.App("vecina-upload")

# Reference the same volume used by the transcriber
data_volume = modal.Volume.from_name("vecina-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume}, timeout=600)
def upload_files(files_data: list[tuple[str, bytes]]):
    """Upload audio files to the Modal volume."""
    from pathlib import Path

    audio_dir = Path("/data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for filename, data in files_data:
        target = audio_dir / filename
        target.write_bytes(data)
        uploaded.append(filename)
        print(f"✓ Uploaded {filename} ({len(data)} bytes)")

    # Commit changes to volume
    data_volume.commit()

    return {"uploaded": len(uploaded), "files": uploaded}


@app.local_entrypoint()
def main(audio_dir: str = "_data"):
    """Upload local audio files to Modal volume."""
    local_dir = Path(audio_dir)

    if not local_dir.exists():
        print(f"❌ Directory not found: {local_dir}")
        return

    # Find audio files
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    audio_files = [p for p in local_dir.rglob(
        "*") if p.is_file() and p.suffix.lower() in audio_exts]

    if not audio_files:
        print(f"❌ No audio files found in {local_dir}")
        return

    print(f"📂 Found {len(audio_files)} audio file(s) in {local_dir}")

    # Read files into memory
    files_data = []
    for path in audio_files:
        print(f"📖 Reading {path.name}...")
        files_data.append((path.name, path.read_bytes()))

    # Upload to Modal
    print(f"\n☁️  Uploading to Modal volume...")
    result = upload_files.remote(files_data)

    print(f"\n✅ Upload complete!")
    print(f"📊 Uploaded {result['uploaded']} file(s)")
    print(f"\n🎯 Next step: Run batch transcription")
    print(f"   modal run vecina_transcriber/modal_entrypoint.py::transcribe_all_modal")
