# VECINA Podcast Transcription Pipeline
# This script uploads audio files to Modal, runs transcription, and downloads results

param(
    [string]$AudioDir = "_data/podcasts",
    [string]$OutputDir = "_data/_output"
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Blue }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "╔════════════════════════════════════════════════════════╗"
Write-Info "║   VECINA Podcast Transcription Pipeline              ║"
Write-Info "║   RIOS Institute - Community Involvement              ║"
Write-Info "╚════════════════════════════════════════════════════════╝"
Write-Host ""

# Step 1: Upload audio files
Write-Warning "📤 Step 1/3: Uploading audio files to Modal..."
Write-Info "   Source: $AudioDir"
Write-Host ""

try {
    modal run scripts/upload_to_modal.py --audio-dir $AudioDir
    if ($LASTEXITCODE -ne 0) { throw "Upload failed" }
}
catch {
    Write-Error "❌ Upload failed!"
    exit 1
}

Write-Host ""
Write-Success "✅ Upload complete!"
Write-Host ""

# Step 2: Run transcription
Write-Warning "🎙️  Step 2/3: Running batch transcription..."
Write-Info "   This may take a while depending on file sizes and chunk settings"
Write-Host ""

try {
    modal run vecina_transcriber/modal_entrypoint.py::transcribe_all_modal
    if ($LASTEXITCODE -ne 0) { throw "Transcription failed" }
}
catch {
    Write-Error "❌ Transcription failed!"
    exit 1
}

Write-Host ""
Write-Success "✅ Transcription complete!"
Write-Host ""

# Step 3: Download results
Write-Warning "📥 Step 3/3: Downloading transcripts..."
Write-Info "   Destination: $OutputDir"
Write-Host ""

try {
    modal run scripts/download_from_modal.py --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "Download failed" }
}
catch {
    Write-Error "❌ Download failed!"
    exit 1
}

Write-Host ""
Write-Success "╔════════════════════════════════════════════════════════╗"
Write-Success "║   ✅ PIPELINE COMPLETE!                               ║"
Write-Success "╚════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Info "📊 Summary:"
Write-Host "   Audio source: $AudioDir"
Write-Host "   Transcripts saved to: $OutputDir"
Write-Host ""
Write-Info "💡 Tip: Check your transcripts in $OutputDir"
