import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from synthesis.white_noise_augmenter import WhiteNoiseAugmenter
from .dialog_composer import DialogComposer
from .speaker_database import SpeakerDatabase
import soundfile as sf
from .rir_augmenter import RIRAugmenter
from .noise_augmenter import NoiseAugmenter
import json
import os
import pandas as pd

class SyntheticDatasetGenerator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.speaker_db = SpeakerDatabase(self.config)
        self.composer = DialogComposer(self.config, self.speaker_db)
        self.noise_aug = NoiseAugmenter(self.config)
        self.rir_aug = RIRAugmenter(self.config)
        self.white_noise_aug = WhiteNoiseAugmenter(self.config)

    def _generate_track(self):
        dialog_type = np.random.choice(
            ['monologue', 'dialog', 'overlap'],
            p=self.config['dialog_type_probs']
        )
        
        if dialog_type == 'monologue':
            audio, metadata, stats = self.composer._generate_monologue()
        elif dialog_type == 'dialog':
            audio, metadata, stats = self.composer._generate_dialog()
        else:
            audio, metadata, stats = self.composer._generate_overlap()

        stats['noise_applied'] = False
        if np.random.rand() < self.config['noise_prob']:
            audio = self.noise_aug.apply_noise(audio)
            stats['noise_applied'] = True
        
        stats['rir_applied'] = False
        if np.random.rand() < self.config['rir_prob']:
            audio = self.rir_aug.apply_rir(audio)
            stats['rir_applied'] = True

        stats['white_noise_applied'] = False
        if np.random.rand() < self.config['white_noise_prob']:
            audio = self.white_noise_aug.apply_noise(audio)
            stats['white_noise_applied'] = True

        return {
            'audio': audio,
            'metadata': metadata,
            'dialog_type': dialog_type,
            'sr': self.config['sr'],
            'stats': stats
        }

    def generate(self, output_dir, num_samples):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        data_path = Path(os.path.join(output_path, 'data'))
        data_path.mkdir(exist_ok=True, parents=True)


        spec_data = []
        meta_data = []

        for i in tqdm(range(num_samples)):
            track = self._generate_track()
            filename = f"track_{i:06d}_{track['dialog_type']}.wav"

            sf.write(os.path.join(data_path, filename), track['audio'], track['sr'])
            self._save_annotations(data_path / f"{filename}.rttm", track['metadata'])

            track_spec = {
                'file_path': "/".join(os.path.join(data_path, filename).split("/")[-3:]),
                'annotation_path': "/".join(os.path.join(data_path, f"{filename}.rttm").split("/")[-3:]),
                'type': track['dialog_type'],
                'audio_duration': round(len(track['audio']) / track['sr'], 3),
                'num_speakers': track['stats']['num_speakers'],
                'overlap_duration': track['stats']['overlap_duration'],
                'silence_duration': track['stats']['silence_duration'],
                'rir_applied': track['stats']['rir_applied'],
                'noise_applied': track['stats']['noise_applied'],
                'white_noise_applied' : track['stats']['white_noise_applied']
            }

            starts = []
            ends = []
            speakers = []

            for seg in track['metadata']:
                if seg['type'] != "speech":
                    continue
                starts.append(seg['start'])
                ends.append(seg['end'])
                speakers.append(seg['speaker'])

            track_meta = {
                'file_path': "/".join(os.path.join(data_path, filename).split("/")[-3:]),
                'timestamps_start': starts,
                'timestamps_end': ends,
                'speakers': speakers,
            }
            spec_data.append(track_spec)
            meta_data.append(track_meta)
        
        pd.DataFrame(meta_data).to_csv(os.path.join(output_path, "metadata.csv"), index=False)
        pd.DataFrame(spec_data).to_csv(os.path.join(output_path, "spec.csv"), index=False)

    def _save_annotations(self, path, metadata):
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
