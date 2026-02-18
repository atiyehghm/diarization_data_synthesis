import os
import random
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import soundfile as sf


class SpeakerDatabase:
    def __init__(self, config):
        """
        :param data_root: the root path of dataset.
        :param min_duration: minimum duration of samples.
        """
        self.data_root = Path(config['speaker_data_path'])
        self.min_duration = config.get('min_sample_duration', 1.0)
        self.min_samples = config.get('min_speaker_samples', 1)

        self.meta = {}
        self.audio_index = defaultdict(list)

        for i, dir in enumerate(os.listdir(self.data_root)):
            self.meta[i] = dir
            for data in os.listdir(os.path.join(self.data_root, dir)):
                if os.path.splitext(data)[1] == '.wav':
                    audio_path = os.path.join(self.data_root, dir, data)
                    try:
                        duration = self._get_audio_duration(audio_path)
                        if duration >= self.min_duration:
                            self.audio_index[i].append(
                                {
                                    'audio_path': audio_path,
                                    'duration': duration,
                                }
                            )
                    except Exception as e:
                        logging.warning(f"Error processing {audio_path}: {str(e)}")

        self._validate_speakers()

    def _get_audio_duration(self, path: Path) -> float:
        with sf.SoundFile(path) as f:
            return f.frames / f.samplerate

    def _validate_speakers(self):
        initial_count = len(self.audio_index)
        self.audio_index = {k: v for k, v in self.audio_index.items() if len(v) >= self.min_samples}
        
        logging.info(f"Database initialized. Speakers: {len(self.audio_index)}/"
                    f"({initial_count} before validation)")

    def get_random_speakers(self, num_speakers: int) -> List[str]:
        return random.sample(list(self.audio_index.keys()), min(num_speakers, len(self.audio_index)))

    def get_random_utterance(self, speaker_id: str) -> Dict:
        if speaker_id not in self.audio_index:
            raise ValueError(f"Speaker {speaker_id} not found in database")

        audio_info = random.choice(self.audio_index[speaker_id])
        return {
            'speaker_id': speaker_id,
            'audio_path': os.path.join(self.data_root, audio_info['audio_path']),
            'duration': audio_info['duration'],
            'volume': 1.0
        }

    def get_speaker_metadata(self, speaker_id: str) -> Dict:
        return self.audio_index.get(speaker_id, {})

    @property
    def total_speakers(self) -> int:
        return len(self.audio_index)

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self.audio_index.values())
