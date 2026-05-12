import random
import numpy as np
from scipy.io import wavfile
import os
import glob
from data_scripts.data_utils.utils.audio_io import load_audio


class NoiseAugmenter:
    def __init__(self, config):
        self.noise_samples = self._load_noise_samples(config['noise_data_path'])
        self.snr_range = config.get('snr_range', [10, 30])

    def _load_noise_samples(self, noise_dataset_path):
        """
        :param noise_dataset_path: Путь до директории с шумовыми файлами
        :return: Список массивов шумов
        """
        noise_files = glob.glob(os.path.join(noise_dataset_path, "**", '*.wav'), recursive=True)
        noise_samples = []
        for noise_file in noise_files:
            _, noise_sample = wavfile.read(noise_file)
            noise_samples.append(noise_sample)
        return noise_samples

    def apply_noise(self, audio_input):
        if isinstance(audio_input, str):
            audio_samples = load_audio(audio_input)
        else:
            audio_samples = audio_input

        # Convert to float
        audio_samples = audio_samples.astype(np.float32)
        audio_samples /= np.max(np.abs(audio_samples)) + 1e-9

        noise_sample = random.choice(self.noise_samples).astype(np.float32)
        noise_sample -= np.mean(noise_sample)

        if len(noise_sample) < len(audio_samples):
            repeats = len(audio_samples) // len(noise_sample) + 1
            noise_sample = np.tile(noise_sample, repeats)
        noise_sample = noise_sample[:len(audio_samples)]

        # RMS-based SNR
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)

        signal_rms = np.sqrt(np.mean(audio_samples ** 2))
        noise_rms = np.sqrt(np.mean(noise_sample ** 2)) + 1e-9

        scale_factor = signal_rms / (snr_linear * noise_rms)

        noisy_audio = audio_samples + noise_sample * scale_factor

        return noisy_audio


if __name__ == "__main__":
    import soundfile as sf
    import librosa

    def load_audio(path, sr=None):
        audio, sample_rate = sf.read(path)
        if sr and sample_rate != sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
        return audio

    path_to_noises = "../data/noises"

    augmenter = NoiseAugmenter({'noise_dataset_path': path_to_noises, 'snr_range': (10, 30)})

    audio_array_example = load_audio("../data/speakers/2.wav")
    noisy_audio = augmenter.apply_noise(audio_array_example)

    def save_audio(audio, path, sr=16000):
        sf.write(path, audio, sr)
    save_audio(noisy_audio, "../data/speakers/aug2.wav")
