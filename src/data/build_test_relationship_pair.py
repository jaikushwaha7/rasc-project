import json
from pathlib import Path
from sklearn.model_selection import train_test_split

input_path = Path("data/processed/relationships/relationship_pairs.json")
output_dir = Path("data/processed/relationships/test")
output_dir.mkdir(parents=True, exist_ok=True)

# Load original relationships
with open(input_path) as f:
    data = json.load(f)

# Split
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save
with open(output_dir / "train.json", "w") as f:
    json.dump(train_data, f, indent=2)
with open(output_dir / "val.json", "w") as f:
    json.dump(val_data, f, indent=2)
with open(output_dir / "test.json", "w") as f:
    json.dump(test_data, f, indent=2)
