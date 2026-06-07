from synthesis import generate_dataset
from pathlib import Path
import yaml

# Get the directory where this script is located
script_dir = Path(__file__).parent
config_path = script_dir / "configs" / "base_config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

generator = generate_dataset.SyntheticDatasetGenerator(config)
generator.generate("/home/atiyehghm/Desktop/hamseda/diarization_data_synthesis/shemo_test", num_samples=5000)
