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
    """Upload audio files to the Modal volume, preserving directory structure."""
    from pathlib import Path

    audio_dir = Path("/data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for relative_path, data in files_data:
        target = audio_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        uploaded.append(relative_path)
        print(f"✓ Uploaded {relative_path} ({len(data)} bytes)")

    # Commit changes to volume
    data_volume.commit()

    return {"uploaded": len(uploaded), "files": uploaded}


@app.local_entrypoint()
def main(audio_dir: str = "_data"):
    """Upload local audio files to Modal volume, preserving directory structure."""
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

    # Read files into memory with relative paths preserved
    files_data = []
    for path in audio_files:
        # Get relative path from base directory to preserve folder structure
        try:
            rel_path = path.relative_to(local_dir)
        except ValueError:
            rel_path = Path(path.name)

        print(f"📖 Reading {rel_path}...")
        files_data.append(
            (str(rel_path).replace("\\", "/"), path.read_bytes()))

    # Upload to Modal
    print(f"\n☁️  Uploading to Modal volume...")
    result = upload_files.remote(files_data)

    print(f"\n✅ Upload complete!")
    print(f"📊 Uploaded {result['uploaded']} file(s)")
    print(f"\n🎯 Next step: Run batch transcription")
    print(f"   modal run vecina_transcriber/modal_entrypoint.py::transcribe_all_modal")
