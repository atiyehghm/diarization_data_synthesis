import argparse
import soundfile as sf
import numpy as np
from .utils.audio_io import load_audio, save_audio


class AudioNormalizer:
    def __init__(self, target_level=-23.0, peak_level=-1.0, mode='rms'):
        self.target_level = target_level
        self.peak_level = peak_level
        self.mode = mode

    def normalize(self, input_path, output_path=None):
        audio = load_audio(input_path)
        
        if self.mode == 'peak':
            self._peak_normalization(audio)
        elif self.mode == 'rms':
            self._rms_normalization(audio)
        elif self.mode == 'LUFS':
            self._lufs_normalization(audio)
        
        #self._apply_peak_limiter(audio)
        if output_path is None:
            return audio
        save_audio(audio, output_path)

    def _peak_normalization(self, audio):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio *= 10**(self.peak_level / 20) / peak

    def _rms_normalization(self, audio):
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio *= 10**(self.target_level / 20) / rms

    def _lufs_normalization(self, audio):
        import pyloudnorm as pyln
        meter = pyln.Meter(audio.frame_rate)
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, self.target_level)
        return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--mode", choices=['peak', 'rms', 'LUFS'], default='LUFS')
    parser.add_argument("--target", type=float, default=-23.0)
    args = parser.parse_args()

    normalizer = AudioNormalizer(target_level=args.target, mode=args.mode)
    normalizer.normalize(args.input, args.output)
