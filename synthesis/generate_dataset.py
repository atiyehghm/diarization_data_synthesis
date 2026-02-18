import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .dialog_composer import DialogComposer
from .speaker_database import SpeakerDatabase
#from .data_scripts.data_utils import NoiseAugmenter, RirAugmenter
import soundfile as sf
from .rir_augmenter import RIRAugmenter
from .noise_augmenter import NoiseAugmenter
import json


class SyntheticDatasetGenerator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.speaker_db = SpeakerDatabase(self.config)
        self.composer = DialogComposer(self.config, self.speaker_db)
        self.noise_aug = NoiseAugmenter(self.config)
        self.rir_aug = RIRAugmenter(self.config)

    def _generate_track(self):
        dialog_type = np.random.choice(
            ['monologue', 'dialog', 'overlap'],
            p=self.config['dialog_type_probs']
        )
        
        if dialog_type == 'monologue':
            audio, metadata = self.composer._generate_monologue()
        elif dialog_type == 'dialog':
            audio, metadata = self.composer._generate_dialog()
        else:
            audio, metadata = self.composer._generate_overlap()

        if np.random.rand() < self.config['noise_prob']:
            audio = self.noise_aug.apply_noise(audio,)
        
        if np.random.rand() < self.config['rir_prob']:
            audio = self.rir_aug.apply_rir(audio)
        
        return {
            'audio': audio,
            'metadata': metadata,
            'dialog_type': dialog_type,
            'sr': self.config['sr']
        }

    def generate(self, output_dir, num_samples):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for i in tqdm(range(num_samples)):
            track = self._generate_track()
            filename = f"track_{i:06d}_{track['dialog_type']}.wav"

            sf.write(output_path / filename, track['audio'], track['sr'])
            self._save_metadata(output_path / f"{filename}.rttm", track['metadata'])

    def _save_metadata(self, path, metadata):
        """Сохранение RTTM-разметки"""
        with open(path, 'w') as f:
            for seg in metadata:
                if seg['type'] == 'speech':
                    f.write(
                        f"SPEAKER {path.stem} 1 {seg['start']:.3f} {seg['end']-seg['start']:.3f} "
                        f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                    )


if __name__ == "__main__":
    with open('../librispeech.json') as f:
        meta = json.load(f)
    generator = SyntheticDatasetGenerator("../configs/base_config.yaml")
    generator.generate("./output", num_samples=10)
