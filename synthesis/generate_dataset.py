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
    def __init__(self, config):

        self.config = config
        
        self.speaker_db = SpeakerDatabase(self.config)
        self.composer = DialogComposer(self.config, self.speaker_db)
        self.noise_aug = NoiseAugmenter(self.config)
        self.rir_aug = RIRAugmenter(self.config)
        self.white_noise_aug = WhiteNoiseAugmenter(self.config)

    def _generate_track(self):
        dialog_type = np.random.choice(
            ['monologue', 'dialog', 'overlap', 'backchannel'],
            p=self.config['dialog_type_probs']
        )
        
        if dialog_type == 'monologue':
            audio, metadata, stats = self.composer._generate_monologue()
        elif dialog_type == 'dialog':
            audio, metadata, stats = self.composer._generate_dialog()
        elif  dialog_type == 'overlap':
            audio, metadata, stats = self.composer._generate_overlap()
        else:
            audio, metadata, stats = self.composer._generate_backchannel()

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

    def _calculate_overlap_seconds(self, segments):
        """
        Returns total overlap duration in seconds.
        """
        events = []

        for seg in segments:
            if seg['type'] != "speech":
                    continue
            events.append((seg['start'], 1))
            events.append((seg['end'], -1))

        events.sort()

        active = 0
        previous_time = None
        overlap = 0.0

        for time, change in events:

            if previous_time is not None and active >= 2:
                overlap += time - previous_time

            active += change
            previous_time = time

        return overlap

    def _calculate_backchannel_seconds(self, segments):
        """
        Returns total duration of segments that are fully
        contained inside another segment.
        """
        backchannel_duration = 0.0

        for i, seg_i in enumerate(segments):
            if seg_i['type'] != 'speech':
                continue

            start_i = seg_i['start']
            end_i = seg_i['end']

            is_backchannel = False

            for j, seg_j in enumerate(segments):
                if seg_j['type'] != "speech":
                    continue

                start_j = seg_j['start']
                end_j = seg_j['end']

                if i == j:
                    continue

                if (
                    start_i >= start_j
                    and end_i <= end_j
                ):
                    is_backchannel = True
                    break

            if is_backchannel:
                backchannel_duration += end_i - start_i

        return backchannel_duration

    def _calculate_silence_seconds(self, segments):
        total_silence = 0.0

        for seg in segments:
            if seg['type'] == 'silence':
                total_silence +=  seg['end'] - seg['start']

        return total_silence

    def generate(self, output_dir, num_samples):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        data_path = Path(os.path.join(output_path, 'data'))
        data_path.mkdir(exist_ok=True, parents=True)


        spec_data = []
        meta_data = []
        transcripts_data = []

        for i in tqdm(range(num_samples)):
            track = self._generate_track()
            filename = f"track_{i:06d}_{track['dialog_type']}"

            sf.write(os.path.join(data_path, f"{filename}.wav"), track['audio'], track['sr'])
            self._save_annotations(data_path / f"{filename}.rttm", track['metadata'])

            overlap_duration = self._calculate_overlap_seconds(track['metadata'])
            backchannel_duration = self._calculate_backchannel_seconds(track['metadata'])
            silence_duration = self._calculate_silence_seconds(track['metadata'])

            if self.config.get("include_transcripts", False) and track['dialog_type'] in ['overlap', 'backchannel']:
                transcripts_data.extend(self._save_stt_transcripts(track['audio'], track['sr'], track['metadata'], data_path.parent ,filename))

            track_spec = {
                'file_path': "/".join(os.path.join(data_path, filename).split("/")[-3:]),
                'annotation_path': "/".join(os.path.join(data_path, f"{filename}.rttm").split("/")[-3:]),
                'type': track['dialog_type'],
                'audio_duration': round(len(track['audio']) / track['sr'], 3),
                'num_speakers': track['stats']['num_speakers'],
                'audio_paths': track['stats']['segments_path'], 
                'overlap_duration': overlap_duration,
                'silence_duration': silence_duration,
                'backchannel_duration': backchannel_duration,
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
        if self.config.get("include_transcripts", False):
            pd.DataFrame(transcripts_data).to_csv(os.path.join(os.path.join(data_path.parent, "stt"), "metadata.csv"), index=False)

    def _save_annotations(self, path, metadata):
        with open(path, 'w') as f:
            for seg in metadata:
                if seg['type'] == 'speech':
                    f.write(
                        f"SPEAKER {path.stem} 1 {seg['start']:.3f} {seg['end']-seg['start']:.3f} "
                        f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                    )


    def _save_stt_transcripts(self, audio, sampling_rate, metadata, data_path, filename):
        transcripts_meta = []
        os.makedirs(os.path.join(data_path, "stt"), exist_ok=True)
        for i, seg in enumerate(metadata):
            if seg['type'] == 'speech':
                cropped_audio = audio[int(seg['start'] * sampling_rate): int(seg['end'] * sampling_rate) ]
                sf.write(os.path.join(data_path, "stt", f"{filename}_seg_{i}.wav"), cropped_audio, samplerate=sampling_rate)
                transcripts_meta.append({
                    'name': f"{filename}_seg_{i}.wav",
                    'transcript': seg['transcript'],
                    'length': seg['end'] - seg['start'],
                })

        return transcripts_meta
        


if __name__ == "__main__":
    with open('../librispeech.json') as f:
        meta = json.load(f)
    generator = SyntheticDatasetGenerator("../configs/base_config.yaml")
    generator.generate("./output", num_samples=10)
