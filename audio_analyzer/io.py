# io.py

from __future__ import  annotations
import numpy as np
from dataclasses import dataclass
import warnings
import librosa
import soundfile as sf

class AudioLoadError(Exception):
    """
    Raise when no available decoder could load the audio file
    """

    def __init__(self,path: str, librosa_error: Exception | None, soundfile_error: Exception | None):
        self.path = path
        self.librosa_error = librosa_error
        self.soundfile_error = soundfile_error
        parts = [f"could not load audio file {path}"]
        if librosa_error is not None:
            parts.append(f"librosa error: {librosa_error!r}")
        if soundfile_error is not None:
            parts.append(f"soundfile error: {soundfile_error!r}")
        super().__init__("|".join(parts))

@dataclass
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int
    decoder_used: str

def load_audio(path: str) -> LoadedAudio:
    librosa_error: Exception | None = None
    soundfile_error: Exception | None = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples, sample_rate = librosa.load(path, sr=None, mono=True)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            raise AudioLoadError(path, librosa_error=RuntimeError("decoded zero samples"), soundfile_error=None)
        return LoadedAudio(samples=samples, sample_rate=int(sample_rate), decoder_used="librosa")
    except Exception as e:
        librosa_error = e
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        if data.size == 0:
            raise AudioLoadError(path, librosa_error=None, soundfile_error=RuntimeError("decoded zero samples"))
        return LoadedAudio(samples=data, sample_rate=int(sample_rate), decoder_used="soundfile")
    except Exception as e:
        soundfile_error = e
    raise AudioLoadError(path, librosa_error, soundfile_error)

