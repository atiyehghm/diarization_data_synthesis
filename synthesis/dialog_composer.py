import numpy as np
from typing import List, Dict, Tuple
from data_scripts.data_utils import AudioNormalizer, AudioMixer, FadeGenerator
import random


class DialogComposer:
    def __init__(self, config: Dict, speaker_db):
        self.config = config
        self.speaker_db = speaker_db

        self.normalizer = AudioNormalizer()
        self.fade = FadeGenerator()

    def _generate_monologue(self) -> Tuple[np.ndarray, List[Dict]]:
        speaker = self.speaker_db.get_random_speakers(1)[0]
        turns_per_speaker = random.choice(range(self.config['turns_per_speaker'][0], self.config['turns_per_speaker'][1]+1))

        segments = [self.speaker_db.get_random_utterance(speaker) for _ in range(turns_per_speaker)]

        return self._compose_track(segments, pause_range=(0, self.config['max_pause']))

    def _generate_dialog(self) -> Tuple[np.ndarray, List[Dict]]:
        num_speakers = random.choice(range(self.config['num_speakers'][0], self.config['num_speakers'][1] + 1))
        speakers = self.speaker_db.get_random_speakers(num_speakers)
        volumes = self._get_speaker_volume(num_speakers)

        all_segments = []
        for i, spk in enumerate(speakers):
            turns_per_speaker = random.choice(range(self.config['turns_per_speaker'][0], self.config['turns_per_speaker'][1]+1))
            for _ in range(turns_per_speaker):
                seg = self.speaker_db.get_random_utterance(spk)
                seg['volume'] = volumes[i]
                all_segments.append(seg)
        random.shuffle(all_segments)
        return self._compose_track(all_segments, pause_range=(0, self.config['max_pause']))

    def _generate_overlap(self) -> Tuple[np.ndarray, List[Dict]]:
        num_speakers = random.choice(range(self.config['num_speakers'][0], self.config['num_speakers'][1] + 1))
        speakers = self.speaker_db.get_random_speakers(num_speakers)
        volumes = self._get_speaker_volume(num_speakers)

        segments = []
        for i, spk in enumerate(speakers):
            turns_per_speaker = random.choice(range(self.config['turns_per_speaker'][0], self.config['turns_per_speaker'][1]+1))
            for _ in range(turns_per_speaker):
                seg = self.speaker_db.get_random_utterance(spk)
                seg['volume'] = volumes[i]
                # overlap = np.random.uniform(*self.config['overlap_range'])
                # pause = np.random.uniform(-overlap, self.config['max_pause'])
                pause = self._sample_pause(
                        max_overlap=self.config['max_overlap'],
                        max_pause=self.config['max_pause']
                    )
                segments.append((seg, pause))

        random.shuffle(segments)
        return self._compose_overlap_track(segments)

    
    def _get_speaker_volume(self, num_speakers):
        """
        Set the volume for each speaker (either equal volume or variable speaker volume).
        """
        volume_type = np.random.choice(
            ['equal', 'varied'],
            p=self.config['volume_type_probs']
        )
        if volume_type == 'equal':
            volumes = np.ones(num_speakers)
        elif volume_type == 'varied':
            volumes = np.random.normal(
                loc=1.0,
                scale=self.config['normalization_var'],
                size=num_speakers,
            )
            volumes = np.clip(
                np.array(volumes),
                a_min=self.config['min_volume'],
                a_max=self.config['max_volume'],
            ).tolist()

        return volumes

    def _compose_track(self, segments: List, pause_range: Tuple) -> Tuple[np.ndarray, List[Dict]]:
        track = []
        metadata = []
        current_pos = 0.0
        
        for seg in segments:
            audio = self.normalizer.normalize(seg['audio_path'])
            audio = self.fade.apply_fade(audio, fade_out=0.02)
            audio *= seg['volume']
            track.append(audio)
            metadata.append({
                'speaker': seg['speaker_id'],
                'start': current_pos,
                'end': current_pos + len(audio)/self.config['sr'],
                'type': 'speech'
            })

            pause = np.random.uniform(*pause_range)
            if pause > 0:
                silence = np.zeros(int(pause * self.config['sr']))
                track.append(silence)
                metadata.append({
                    'start': current_pos + len(audio)/self.config['sr'],
                    'end': current_pos + len(audio)/self.config['sr'] + pause,
                    'type': 'silence'
                })
                current_pos += pause
            
            current_pos += len(audio)/self.config['sr']
            
        return np.hstack(track), metadata


    def _sample_pause(self, max_overlap, max_pause):
        r = np.random.rand()
        #todo: Do not hardcode the ranges
        if r < 0.4:
            return np.random.uniform(0, max_pause)# No overlap(just  pause)
        elif r < 0.75:
            return -1 * np.random.uniform(0.05, 0.4) # small overlap
        else:
            return -1 * np.random.uniform(0.3, max_overlap)# heavy overlap


    def _compose_overlap_track(self, segments: List) -> Tuple[np.ndarray, List[Dict]]:
        """Сборка трека с перекрытием"""
        tracks = []
        metadata = []
        max_duration = 0
        
        current_pos = 0
        result = np.zeros((0,))
        
        for seg, pause in segments:
            if len(result) == 0:
                pause = max(0, pause)

            audio = self.normalizer.normalize(seg['audio_path'])
            audio = self.fade.apply_fade(audio, fade_out=0.02)
            audio *= seg['volume']

            start_offset = int(self.config['sr'] * pause)

            current_pos = len(result)
            segment_start = current_pos + start_offset
            segment_end = segment_start + len(audio)

            # If segment starts before 0 → left pad
            if segment_start < 0:
                left_pad = abs(segment_start)
                result = np.pad(result, (left_pad, 0))
                segment_start = 0
                segment_end = segment_start + len(audio)

            # If segment extends beyond result → right pad
            if segment_end > len(result):
                right_pad = segment_end - len(result)
                result = np.pad(result, (0, right_pad))

            # Mix audio
            result[segment_start:segment_end] += audio

            metadata.append({
                'speaker': seg['speaker_id'],
                'start': segment_start / self.config['sr'],
                'end': segment_end / self.config['sr'],
                'type': 'speech'
            })


        return result, metadata