from synthesis import generate_dataset
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent
config_path = script_dir / "configs" / "base_config.yaml"

generator = generate_dataset.SyntheticDatasetGenerator(str(config_path))
generator.generate("diarization_data_synthesis/output_path", num_samples=10)
