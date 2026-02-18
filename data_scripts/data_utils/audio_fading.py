import argparse
import numpy as np
from .utils.audio_io import load_audio, save_audio


class FadeGenerator:
    def apply_fade(self, audio, fade_in=0.0, fade_out=0.0, curve='linear'):
        length = len(audio)
        
        if fade_in > 0:
            audio = self._apply_fade_in(audio, fade_in, curve)
        if fade_out > 0:
            audio = self._apply_fade_out(audio, fade_out, curve)
        
        return audio

    def _apply_fade_in(self, audio, duration, curve):
        fade_samples = int(duration * 1000)
        fade_curve = self._generate_curve(fade_samples, curve)
        audio[:fade_samples] *= fade_curve
        return audio

    def _apply_fade_out(self, audio, duration, curve):
        fade_samples = int(duration * 1000)
        fade_curve = self._generate_curve(fade_samples, curve)[::-1]
        audio[-fade_samples:] *= fade_curve
        return audio

    def _generate_curve(self, n_samples, curve_type):
        if curve_type == 'linear':
            return np.linspace(0, 1, n_samples)
        elif curve_type == 'log':
            return np.logspace(-10, 0, n_samples, base=10.0)
        elif curve_type == 'exp':
            return np.exp(np.linspace(-5, 0, n_samples))
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--fade-in", type=float, default=0.0)
    parser.add_argument("--fade-out", type=float, default=0.0)
    parser.add_argument("--curve", choices=['linear', 'log', 'exp'], default='linear')
    args = parser.parse_args()

    audio = load_audio(args.input)
    processor = FadeGenerator()
    processed = processor.apply_fade(audio, args.fade_in, args.fade_out, args.curve)
    save_audio(processed, args.output)
