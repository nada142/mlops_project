
#!/bin/bash

# List all chunk files
for chunk in chunk_*.csv; do
    # Run your model for each chunk and pass the chunk filename
    echo "Processing $chunk..."
    python notebooks/complexity_pred.py $chunk
done
