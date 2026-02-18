import argparse
import numpy as np
from pydub import AudioSegment
from .utils.audio_io import load_audio, save_audio


class AudioMixer:
    def __init__(self, sample_rate=16000, format='wav'):
        self.sample_rate = sample_rate
        self.format = format

    def mix_audios(self, audio_paths, output_path, volumes=None):
        tracks = [load_audio(path) for path in audio_paths]
        max_len = max(len(t) for t in tracks)

        if volumes is None:
            volumes = [1.0] * len(tracks)

        mixed = np.zeros(max_len)
        for track, vol in zip(tracks, volumes):
            padded = np.pad(track, (0, max_len - len(track)), 'constant')
            mixed += padded * vol

        save_audio(mixed, output_path, self.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", nargs='+', required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--volumes", nargs='+', type=float)
    args = parser.parse_args()

    mixer = AudioMixer()
    mixer.mix_audios(args.inputs, args.output, args.volumes)
