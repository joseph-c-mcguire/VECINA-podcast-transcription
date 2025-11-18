"""
Download transcripts from Modal volume to local _data/_output directory.

Usage:
    modal run scripts/download_from_modal.py --output-dir _data/_output
"""

import modal
from pathlib import Path

app = modal.App("vecina-download")

# Reference the same volume
data_volume = modal.Volume.from_name("vecina-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume}, timeout=600)
def list_and_download():
    """List and return all transcript files from Modal volume, preserving directory structure."""
    from pathlib import Path

    transcripts_dir = Path("/data/transcripts")

    if not transcripts_dir.exists():
        return {"files": [], "count": 0}

    files_data = []
    for p in transcripts_dir.rglob("*.*"):
        if p.is_file():
            # Get relative path from transcripts_dir to preserve folder structure
            rel_path = p.relative_to(transcripts_dir)
            files_data.append({
                "name": str(rel_path).replace("\\", "/"),
                "data": p.read_bytes()
            })

    return {"files": files_data, "count": len(files_data)}


@app.local_entrypoint()
def main(output_dir: str = "_data/_output"):
    """Download transcripts from Modal volume."""
    local_dir = Path(output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"☁️  Downloading transcripts from Modal volume...")
    result = list_and_download.remote()

    if result["count"] == 0:
        print("❌ No transcripts found in Modal volume")
        return

    print(f"📥 Downloading {result['count']} file(s)...")

    for file_info in result["files"]:
        target = local_dir / file_info["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file_info["data"])
        print(
            f"✓ Downloaded {file_info['name']} ({len(file_info['data'])} bytes)")

    print(f"\n✅ Download complete!")
    print(f"📂 Files saved to: {local_dir.absolute()}")
