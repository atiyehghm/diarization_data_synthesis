import librosa
import soundfile as sf
import numpy as np


def load_audio(path, sr=None):
    audio, sample_rate = sf.read(path)
    if sr and sample_rate != sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
    return audio


def save_audio(audio, path, sr=16000):
    sf.write(path, audio, sr)


def trim_silence(audio, threshold=0.01, window_size=200):
    energy = np.convolve(np.abs(audio), np.ones(window_size)/window_size, mode='same')
    mask = energy > threshold
    indices = np.where(mask)[0]
    return audio[indices[0]:indices[-1]] if len(indices) > 0 else audio
