import numpy as np
import random
from data_scripts.data_utils.utils.audio_io import load_audio

class WhiteNoiseAugmenter:
    def __init__(self, config):
        self.snr_range = config.get('white_noise_snr_range', [10, 30])
        self.config = config

    def apply_noise(self, audio_input):
        # 1. Handle Input
        if isinstance(audio_input, str):
            # Assuming load_audio is available as in your previous snippet
            audio = load_audio(audio_input, sr=self.config.get('sr', 16000))
        else:
            audio = audio_input

        noise = np.random.normal(0, 1, len(audio)).astype(np.float32)

        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)

        rms_signal = np.sqrt(np.mean(audio**2))
        rms_noise = np.sqrt(np.mean(noise**2)) + 1e-9

        scale_factor = rms_signal / (snr_linear * rms_noise)

        noisy_audio = audio + (noise * scale_factor)

        return noisy_audio