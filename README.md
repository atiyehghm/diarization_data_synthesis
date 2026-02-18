# Speech Diarization Synthetic Dataset Generator

A framework for generating realistic synthetic data for training speech diarization models with controlled parameters.

## Overview

This framework synthesizes multi-speaker audio conversations by combining individual speaker utterances, controlling various acoustic parameters such as overlap, pauses, noise, and reverberation. The generated datasets include RTTM annotation files for training and evaluation of diarization models.

## Algorithm

The generation process follows these steps:

### 1. **Speaker Database Initialization**
   - Loads speaker audio files from the specified directory structure
   - Each speaker should have their own subdirectory containing multiple WAV files
   - Validates speakers based on minimum duration and sample count requirements
   - Creates an index mapping speakers to their available utterances

### 2. **Dialog Type Selection**
   The system randomly selects one of three dialog types based on configured probabilities:
   - **Monologue**: Single speaker delivering multiple turns
   - **Dialog**: Multiple speakers taking turns with pauses between utterances
   - **Overlap**: Multiple speakers with overlapping speech segments

### 3. **Audio Composition**
   For each selected dialog type:
   - **Speaker Selection**: Randomly selects N speakers (configurable range)
   - **Utterance Selection**: For each speaker, randomly selects multiple utterances
   - **Volume Control**: Applies either equal or varied volume levels per speaker
   - **Normalization**: Normalizes each utterance to consistent RMS levels
   - **Fade Application**: Applies fade-out to prevent clicks/pops
   - **Temporal Arrangement**:
     - **Monologue/Dialog**: Sequential arrangement with configurable pauses
     - **Overlap**: Segments can start before previous ones end, creating overlapping speech

### 4. **Audio Augmentation** (Optional, probabilistic)
   - **Noise Addition**: Adds background noise at configurable SNR levels (5-25 dB)
   - **Reverberation**: Applies Room Impulse Response (RIR) convolution for realistic acoustic simulation

### 5. **Output Generation**
   - Saves synthesized audio as WAV files (16 kHz sample rate)
   - Generates RTTM annotation files with speaker segment timestamps
   - Metadata includes dialog type, speaker IDs, and temporal boundaries

## Configuration Files

The framework uses YAML configuration files to control all generation parameters. The main configuration file is located at `configs/base_config.yaml`.

### Configuration Parameters

#### Audio Settings
- `sr`: Sample rate (default: 16000 Hz)

#### Augmentation Parameters
- `reverb_level_range`: [min, max] reverb level range (default: [0.3, 0.7])
- `rir_prob`: Probability of applying RIR augmentation (default: 0.1)
- `snr_range`: [min, max] Signal-to-Noise Ratio in dB (default: [5, 25])
- `noise_prob`: Probability of adding noise (default: 0.1)

#### Generation Parameters
- `dialog_type_probs`: Probabilities for [monologue, dialog, overlap] (default: [0.1, 0.5, 0.4])
- `volume_type_probs`: Probabilities for [equal, varied] volume distribution (default: [0.5, 0.5])
- `normalization_var`: Standard deviation for volume variation (default: 0.5)
- `min_volume`: Minimum volume multiplier (default: 0.1)
- `max_volume`: Maximum volume multiplier (default: 2.0)
- `num_speakers`: [min, max] number of speakers per track (default: [2, 6])
- `turns_per_speaker`: [min, max] number of turns per speaker (default: [2, 4])
- `max_pause`: Maximum pause duration in seconds (default: 1.5)
- `overlap_range`: [min, max] overlap ratio for overlapping segments (default: [0.2, 0.8])
- `max_overlap`: Maximum overlap duration in seconds (default: 1.5)

#### Data Paths
- `speaker_data_path`: Path to directory containing speaker subdirectories
- `noise_data_path`: Path to directory containing noise WAV files
- `rir_data_path`: Path to directory containing RIR WAV files

## Required Input File Structure

### 1. Speaker Data Directory

The speaker data should be organized as follows:

```
speaker_data_path/
├── speaker_001/
│   ├── utterance_001.wav
│   ├── utterance_002.wav
│   └── ...
├── speaker_002/
│   ├── utterance_001.wav
│   ├── utterance_002.wav
│   └── ...
└── ...
```

**Requirements:**
- Each speaker must have their own subdirectory
- Each subdirectory should contain one or more WAV audio files
- Audio files should be at least 1.0 seconds in duration (configurable via `min_sample_duration`)
- Each speaker should have at least 1 valid sample (configurable via `min_speaker_samples`)

### 2. Noise Data Directory (Optional)

For noise augmentation:

```
noise_data_path/
├── noise_file_001.wav
├── noise_file_002.wav
└── ...
```

**Requirements:**
- WAV files containing background noise samples
- Files are searched recursively in subdirectories
- Any WAV file found will be used as a potential noise source

### 3. RIR Data Directory (Optional)

For reverberation augmentation:

```
rir_data_path/
├── rir_file_001.wav
├── rir_file_002.wav
└── ...
```

**Requirements:**
- WAV files containing Room Impulse Response recordings
- Files are searched recursively in subdirectories
- RIR files are convolved with the audio signal to simulate room acoustics

## Usage

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from synthesis import generate_dataset

# Initialize generator with configuration
generator = generate_dataset.SyntheticDatasetGenerator("configs/base_config.yaml")

# Generate dataset
generator.generate("./output", num_samples=1000)
```

### Output Structure

The generator creates:

```
output/
├── track_000000_dialog.wav
├── track_000000_dialog.wav.rttm
├── track_000001_overlap.wav
├── track_000001_overlap.wav.rttm
├── track_000002_monologue.wav
├── track_000002_monologue.wav.rttm
└── ...
```

**RTTM Format:**
```
SPEAKER <file_id> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

### Creating Metadata CSV

After generation, create a metadata CSV file:

```python
python metadata_creation.py
```

This generates a CSV with columns:
- `file_path`: Path to audio file
- `annotation_path`: Path to RTTM file
- `type`: Dialog type (monologue/dialog/overlap)
- `audio_duration`: Duration in seconds
- `source_dataset`: Source dataset identifier
- `num_speakers`: Number of speakers in the track

### Pushing to Hugging Face Hub

To upload the dataset to Hugging Face:

```python
python hf_creation_push.py
```

**Note:** Ensure you have configured Hugging Face authentication and updated the `repo_id` in the script.

## Project Structure

```
speech_benchmark/
├── configs/                    # Configuration files
│   └── base_config.yaml
├── data_scripts/              # Data processing utilities
│   ├── data_utils/           # Audio processing utilities
│   │   ├── audio_normalizer.py
│   │   ├── audio_fading.py
│   │   ├── audio_mixer.py
│   │   └── audio_augmenter.py
│   └── download_scripts/    # Data download utilities
├── synthesis/                 # Core generation code
│   ├── generate_dataset.py   # Main generator class
│   ├── dialog_composer.py    # Dialog composition logic
│   ├── speaker_database.py   # Speaker management
│   ├── noise_augmenter.py    # Noise augmentation
│   └── rir_augmenter.py      # RIR augmentation
├── generation.py             # Generation script
├── metadata_creation.py      # Metadata CSV generator
├── hf_creation_push.py        # Hugging Face upload script
└── requirements.txt          # Python dependencies
```

## Key Features

- **Controlled Parameters**: Fine-grained control over dialog characteristics
- **Realistic Scenarios**: Supports monologues, dialogs, and overlapping speech
- **Acoustic Simulation**: Optional noise and reverberation for realistic conditions
- **Automatic Annotation**: Generates RTTM files compatible with standard diarization tools
- **Scalable**: Can generate datasets of any size
- **Flexible**: Easy to configure via YAML files

## Notes

- All audio is processed at 16 kHz sample rate
- Audio normalization uses RMS-based leveling
- Overlap generation uses negative pause values to create temporal overlaps
- Volume variation follows a normal distribution with configurable variance
- RTTM files follow the standard NIST format for speaker diarization evaluation
