import random
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import os
from data_scripts.data_utils.utils.audio_io import load_audio
import glob


class RIRAugmenter:
    def __init__(self, config):
        self.rir_samples = self._load_rir_samples(config['rir_data_path'])
        self.reverb_level_range = config['reverb_level_range']
        self.config = config

    def _load_rir_samples(self, rir_dataset_path):
        """
        :param rir_dataset_path: Путь до директории с RIR файлами
        :return: Список массивов импульсных откликов
        """
        rir_files = glob.glob(os.path.join(rir_dataset_path, "**", '*.wav'), recursive=True)
        rir_samples = []
        for rir_file in rir_files:
            _, rir_sample = wavfile.read(rir_file)
            rir_samples.append(rir_sample)
        return rir_samples

    def apply_rir(self, audio_input, sr=None):
        if sr is None:
            sr = self.config['sr']

        if isinstance(audio_input, str):
            audio = load_audio(audio_input)
        else:
            audio = audio_input

        # Convert to float for convolution
        audio = audio.astype(np.float32)
        rir = random.choice(self.rir_samples).astype(np.float32)

        rir = rir / np.linalg.norm(rir)

        # Convolution
        reverbed = fftconvolve(audio, rir, mode='full')[:len(audio)]

        if np.max(np.abs(reverbed)) > 0:
            reverbed = reverbed / np.max(np.abs(reverbed))

        return reverbed


if __name__ == "__main__":
    import soundfile as sf
    import librosa

    def load_audio(path, sr=None):
        audio, sample_rate = sf.read(path)
        if sr and sample_rate != sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
        return audio

    path_to_openrir = "../data/rirs"

    augmenter = RIRAugmenter({'rir_dataset_path': path_to_openrir, 'reverb_level_range': (0.3, 0.7)})

    audio_array_example = load_audio("../data/speakers/2.wav")
    print(audio_array_example.shape)
    noisy_audio = augmenter.apply_rir(audio_array_example)

    def save_audio(audio, path, sr=16000):
        sf.write(path, audio, sr)
    save_audio(noisy_audio, "../data/speakers/rir2.wav")
