#!/bin/bash

# echo "Downloading Visual Genome annotations..."


# # curl -L -o images.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# # curl -L -o images2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# curl -L -o annotations.zip https://cs.stanford.edu/people/rak248/VG_100K_2/annotations.zip
# curl -L -o relationships.zip https://cs.stanford.edu/people/rak248/VG_100K_2/relationships.zip
# curl -L -o objects.zip https://cs.stanford.edu/people/rak248/VG_100K_2/objects.zip

# echo "Unzipping..."
# unzip '*.zip'
# rm *.zip

#!/bin/bash

# echo "Downloading Visual Genome annotations..."

# urls=(
#   "https://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
#   "https://visualgenome.org/static/data/dataset/relationships.json.zip"
#   "https://cs.stanford.edu/people/rak248/VG_100K_2/objects.json.zip"
# )

# for url in "${urls[@]}"; do
#   filename=$(basename "$url")
#   echo "→ $filename"
#   curl -L -o "$filename" "$url" || exit 1
# done

# echo "Unzipping..."
# unzip -q '*.zip' && rm -f *.zip

# echo "Done."
# ls -lh *.json 2>/dev/null || echo "No .json files found"

echo "Fetching Visual Genome v1.4 annotations from Washington mirror..."

# These links are confirmed to be active on the CS Washington mirror
urls=(
  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip"
  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"
)

for url in "${urls[@]}"; do
  filename=$(basename "$url")
  
  # Crucial: Delete any partial/failed files first
  rm -f "$filename"

  echo "→ Downloading $filename..."
  # -L follows redirects, -f fails if 404/500, no -C to avoid resume errors
  curl -L -f -o "$filename" "$url" || { echo "Error: Failed to download $filename from $url"; exit 1; }
done

echo "Unzipping (this may take a few minutes)..."
unzip -o '*.zip' && rm -f *.zip

echo "---"
echo "Success! Annotation files ready in $DATA_DIR:"
ls -lh *.json