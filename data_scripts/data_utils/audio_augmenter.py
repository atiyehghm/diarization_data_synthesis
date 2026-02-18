import argparse
import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from .utils.audio_io import load_audio, save_audio
from typing import Optional, Dict, Any


class AudioAugmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        time_aug_params: Optional[Dict] = None,
        spec_aug_params: Optional[Dict] = None
    ):
        """
        :param sample_rate: частота дискретизации аудио
        :param time_aug_params: параметры временных аугментаций
        :param spec_aug_params: параметры SpecAugment
        """
        self.sr = sample_rate
        self.time_augmentations = self._init_time_augmentations(time_aug_params)
        self.spec_augmentations = self._init_spec_augmentations(spec_aug_params)

    def _init_time_augmentations(self, params: Optional[Dict]) -> Compose:
        default_params = {
            'AddGaussianNoise': {
                'min_amplitude': 0.001,
                'max_amplitude': 0.015,
                'p': 0.5
            },
            'PitchShift': {
                'min_semitones': -4,
                'max_semitones': 4,
                'p': 0.5
            },
            'TimeStretch': {
                'min_rate': 0.8,
                'max_rate': 1.2,
                'p': 0.5
            }
        }

        if params:
            default_params.update(params)

        return Compose([
            AddGaussianNoise(**default_params['AddGaussianNoise']),
            PitchShift(**default_params['PitchShift']),
            TimeStretch(**default_params['TimeStretch'])
        ])

    def _init_spec_augmentations(self, params: Optional[Dict]) -> Dict:
        default_params = {
            'time_masks': 2,
            'freq_masks': 2,
            'max_time_mask_size': 50,
            'max_freq_mask_size': 15,
            'apply_time_warp': True,
            'max_time_warp': 5,
            'p': 0.5
        }

        if params:
            default_params.update(params)

        return default_params

    def _apply_spec_augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """Применение SpecAugment к спектрограмме"""
        n_mels, n_steps = spectrogram.shape
        
        # Time warping
        if self.spec_augmentations['apply_time_warp'] and np.random.rand() < self.spec_augmentations['p']:
            w = np.random.randint(-self.spec_augmentations['max_time_warp'], self.spec_augmentations['max_time_warp'])
            spectrogram = librosa.phase_vocoder(spectrogram, rate=1.0, delta=w)

        # Frequency masking
        for _ in range(self.spec_augmentations['freq_masks']):
            f = np.random.randint(0, self.spec_augmentations['max_freq_mask_size'])
            f0 = np.random.randint(0, n_mels - f)
            spectrogram[f0:f0+f, :] = 0

        # Time masking
        for _ in range(self.spec_augmentations['time_masks']):
            t = np.random.randint(0, self.spec_augmentations['max_time_mask_size'])
            t0 = np.random.randint(0, n_steps - t)
            spectrogram[:, t0:t0+t] = 0

        return spectrogram

    def apply_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """
        Применение цепочки аугментаций к аудио
        
        :param audio: исходный аудиосигнал
        :return: аугментированный аудиосигнал
        """
        if self.time_augmentations:
            audio = self.time_augmentations(samples=audio, sample_rate=self.sr)

        if np.random.rand() < self.spec_augmentations['p']:
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_mels=128,
                fmax=8000
            )
            log_S = librosa.power_to_db(S, ref=np.max)

            aug_S = self._apply_spec_augment(log_S)

            S_aug = librosa.db_to_power(aug_S)
            audio = librosa.feature.inverse.mel_to_audio(
                S_aug,
                sr=self.sr,
                n_fft=2048,
                hop_length=512
            )

        return audio

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Audio Augmentations with SpecAugment')
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")

    # Time augmentation parameters
    parser.add_argument("--noise-p", type=float, default=0.5, help="Probability of adding noise")
    parser.add_argument("--pitch-p", type=float, default=0.5, help="Probability of pitch shift")
    parser.add_argument("--stretch-p", type=float, default=0.5, help="Probability of time stretch")

    # SpecAugment parameters
    parser.add_argument("--spec-p", type=float, default=0.5, help="Probability of SpecAugment")
    parser.add_argument("--time-masks", type=int, default=2, help="Number of time masks")
    parser.add_argument("--freq-masks", type=int, default=2, help="Number of frequency masks")
    parser.add_argument("--max-time-mask", type=int, default=50, help="Max time mask size")
    parser.add_argument("--max-freq-mask", type=int, default=15, help="Max frequency mask size")
    
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    audio, sr = load_audio(args['input'], sr=args['sr'])

    augmenter = AudioAugmenter(
        sample_rate=sr,
        time_aug_params={
            'AddGaussianNoise': {'p': args['noise_p']},
            'PitchShift': {'p': args['pitch_p']},
            'TimeStretch': {'p': args['stretch_p']}
        },
        spec_aug_params={
            'p': args['spec_p'],
            'time_masks': args['time_masks'],
            'freq_masks': args['freq_masks'],
            'max_time_mask_size': args['max_time_mask'],
            'max_freq_mask_size': args['max_freq_mask']
        }
    )

    augmented_audio = augmenter.apply_augmentations(audio)

    save_audio(augmented_audio, args['output'], sr)
