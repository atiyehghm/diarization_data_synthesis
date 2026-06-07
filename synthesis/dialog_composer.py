import numpy as np
from typing import Any, List, Dict, Tuple
from data_scripts.data_utils import AudioNormalizer, FadeGenerator
import random
from loguru import logger


class DialogComposer:
    def __init__(self, config: Dict, speaker_db):
        self.config = config
        self.speaker_db = speaker_db

        self.normalizer = AudioNormalizer()
        self.fade = FadeGenerator(self.config)
        self.include_transcript = self.config.get('include_transcripts', False)

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
        for i, spk in enumerate[Any](speakers):
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

                pause = self._sample_pause(
                        max_overlap=self.config['max_overlap'],
                        max_pause=self.config['max_pause'] * 0.1
                    )
                segments.append((seg, pause))
        random.shuffle(segments)
        return self._compose_overlap_track(segments)


    def _generate_backchannel(self) -> Tuple[np.ndarray, List[Dict]]:
        max_bc_duration = 8.0
        min_primary_duration = self.config.get('min_backchannel_primary_duration', 3.0)
        min_offset = self.config.get('backchannel_min_offset', 1.0)
        bc_volume_scale = self.config.get('backchannel_volume_scale', 0.8)
        segments_path = []
        num_speakers = max(
            2,
            random.choice(range(self.config['num_speakers'][0], self.config['num_speakers'][1] + 1)),
        )
        speakers = self.speaker_db.get_random_speakers(num_speakers)
        # volumes = self._get_speaker_volume(num_speakers)
        volume_map = {spk: 1.0 for i, spk in enumerate(speakers)}

        primary_speaker = random.choice(speakers)
        primary_seg = self._sample_utterance_with_min_duration(primary_speaker, min_primary_duration)
        primary_audio = self._prepare_segment_audio(primary_seg, volume_map[primary_speaker])
        segments_path.append(primary_seg['audio_path'])

        result = primary_audio.copy()
        sr = self.config['sr']
        metadata = [self._build_speech_metadata(primary_seg, 0.0, len(result) / sr)]
        used_speakers = {primary_speaker}

        n_backchannels = random.choice(range(1, 3))
        other_speakers = [s for s in speakers if s != primary_speaker]
        for _ in range(n_backchannels):
            bc_seg = self._sample_short_utterance(other_speakers, max_bc_duration)
            if bc_seg is None:
                continue
            segments_path.append(bc_seg['audio_path'])
            bc_audio = self._prepare_segment_audio(
                bc_seg, volume_map[bc_seg['speaker_id']] * bc_volume_scale
            )
            result, start_idx, end_idx = self._insert_backchannel(
                result, bc_audio, min_offset=min_offset, max_duration=max_bc_duration
            )
            metadata.append(self._build_speech_metadata(
                bc_seg, start_idx / sr, end_idx / sr
            ))
            used_speakers.add(bc_seg['speaker_id'])

        stats = {
            "num_speakers": len(used_speakers),
            "segments_path": segments_path,
        }
        return result, metadata, stats

    def _sample_utterance_with_min_duration(self, speaker_id, min_duration, max_attempts=20):
        best_seg = None
        for _ in range(max_attempts):
            seg = self.speaker_db.get_random_utterance(speaker_id)
            if seg['duration'] >= min_duration:
                return seg
            if best_seg is None or seg['duration'] > best_seg['duration']:
                best_seg = seg

        # Warn but return the longest found rather than silently using a bad sample
        logger.warning(
            f"Speaker {speaker_id}: no utterance >= {min_duration}s found "
            f"in {max_attempts} attempts. Using best found: {best_seg['duration']:.2f}s"
        )
        return best_seg


    def _sample_short_utterance(self, speaker_ids, max_duration, max_attempts=30):
        for _ in range(max_attempts):
            spk = random.choice(speaker_ids)
            seg = self.speaker_db.get_random_utterance(spk)
            if seg['duration'] < max_duration:
                return seg
        return None

    def _prepare_segment_audio(self, seg, volume):
        audio = self.normalizer.normalize(seg['audio_path'])
        audio = self.fade.apply_fade(
            audio, fade_out=self.config['fade_out'], fade_in=self.config['fade_in']
        )
        return audio * volume

    def _build_speech_metadata(self, seg, start, end):
        meta = {
            'speaker': seg['speaker_id'],
            'start': start,
            'end': end,
            'type': 'speech',
        }
        if self.include_transcript and 'transcript' in seg:
            meta['transcript'] = seg['transcript']
        return meta

    def _insert_backchannel(self, primary_audio, backchannel_audio, min_offset=1.0, max_duration=1.5):
        sr = self.config['sr']
        duration_p = len(primary_audio)
        duration_b = len(backchannel_audio)
        offset_samples = int(min_offset * sr)
        max_samples = int(max_duration * sr)

        if duration_b > max_samples:
            backchannel_audio = backchannel_audio[:max_samples]
            duration_b = len(backchannel_audio)

        # Guard: backchannel must fit between both offsets
        available = duration_p - 2 * offset_samples
        if duration_b >= available:
            # Truncate to fit, or if even that's impossible, center it
            duration_b = min(duration_b, max(1, available))
            backchannel_audio = backchannel_audio[:duration_b]

        min_start = offset_samples
        max_start = duration_p - duration_b - offset_samples  # guarantees end offset too

        if max_start <= min_start:
            # No valid range: place as centered as possible within bounds
            start_idx = max(offset_samples, (duration_p - duration_b) // 2)
        else:
            start_idx = random.randint(min_start, max_start)

        end_idx = start_idx + duration_b
        output_length = max(duration_p, end_idx)

        mixed = np.zeros(output_length, dtype=primary_audio.dtype)
        mixed[:duration_p] += primary_audio
        mixed[start_idx:end_idx] += backchannel_audio
        return mixed, start_idx, end_idx

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
        speakers = []
        segments_path = []
        for seg in segments:
            audio = self.normalizer.normalize(seg['audio_path'])
            audio = self.fade.apply_fade(audio, fade_out=self.config['fade_out'], fade_in=self.config['fade_in'])
            audio *= seg['volume']
            track.append(audio)
            speakers.append(seg['speaker_id'])
            audio_meta = {
                'speaker': seg['speaker_id'],
                'start': current_pos,
                'end': current_pos + len(audio)/self.config['sr'],
                'type': 'speech'
            }
            segments_path.append(seg['audio_path'])
            if self.include_transcript:
                audio_meta['transcript'] = seg['transcript']
            metadata.append(audio_meta)

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
        stats = {
                 "num_speakers": len(set(speakers)),
                 "segments_path": segments_path
                 }

        return np.hstack(track), metadata, stats


    def _sample_pause(self, max_overlap, max_pause):
        r = np.random.rand()
        if r < self.config['overlap_probs'][0]:
            return np.random.uniform(0, max_pause)# No overlap(just  pause)
        elif r < self.config['overlap_probs'][1]:
            return -1 * np.random.uniform(0.0, self.config['small_overlap']) # small overlap
        else:
            return -1 * np.random.uniform(self.config['small_overlap'], max_overlap)# heavy overlap


    def _compose_overlap_track(self, segments: List) -> Tuple[np.ndarray, List[Dict]]:
        metadata = []
        
        current_pos = 0
        result = np.zeros((0,))
        speakers = []
        segments_path = []

        for seg, pause in segments:
            if len(result) == 0:
                pause = max(0, pause)
            
            audio = self.normalizer.normalize(seg['audio_path'])
            audio = self.fade.apply_fade(audio, fade_out=self.config['fade_out'], fade_in=self.config['fade_in'])
            audio *= seg['volume']
            segments_path.append(seg['audio_path'])

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
            
            speakers.append(seg['speaker_id'])
            audio_meta = {
                'speaker': seg['speaker_id'],
                'start': segment_start / self.config['sr'],
                'end': segment_end / self.config['sr'],
                'type': 'speech',
            }
            if self.include_transcript:
                audio_meta['transcript'] = seg['transcript']
            metadata.append(audio_meta)
        

        stats = {
                 "num_speakers": len(set(speakers)),
                 "segments_path": segments_path
                 }
                 
        return result, metadata, stats