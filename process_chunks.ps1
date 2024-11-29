# process_chunks.ps1

# Set the output directory
$chunksDir = "chunks"

# Create the output directory if it doesn't exist
if (-Not (Test-Path $chunksDir)) {
    New-Item -Path $chunksDir -ItemType Directory
}

# List all chunk files in the current directory (if needed)
Get-ChildItem -Filter "chunk_*.csv" | ForEach-Object {
    $chunk = $_.FullName
    Write-Host "Processing $chunk..."

    # Process the chunk and pass it to your Python script
    # The output should be saved in the 'chunks/' directory if necessary
    python notebooks/complexity_pred.py $chunk

    # Example: If you want to save a processed chunk as new CSV inside the 'chunks' directory
    # This part depends on how your Python script works and what output is generated
    # If the Python script generates new chunks, move or save them in the chunks directory:
    $processedChunk = "processed_$($_.Name)"
    Move-Item -Path $_.FullName -Destination "$chunksDir\$processedChunk"
    Write-Host "Processed chunk saved to $chunksDir\$processedChunk"
}

Write-Host "Chunk processing complete."
