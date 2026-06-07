import os
import pandas as pd
import librosa
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio

# ==============================
# Config
# ==============================
HF_TOKEN = "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"  
INPUT_CSV = "/home/atiyehghm/Desktop/hamseda/diarization_data_synthesis/ganjoor_results_20k/stt/metadata.csv"
AUDIO_BASE_PATH = "/home/atiyehghm/Desktop/hamseda/diarization_data_synthesis/ganjoor_results_20k/stt/"
HF_DATASET_NAME = "atiyehghm/ganjoor_stt_overlap"


def process_row(path, text):
    full_path = os.path.join(AUDIO_BASE_PATH, path)

    audio, sr = librosa.load(full_path, sr=None)
    duration = len(audio) / sr

    return {
        "audio": full_path,
        "transcript": text,
        "word_count": len(text.split()),
        "character_count": len(text),
        "duration": duration,
        "sample_rate": sr,
    }


def main():
    df = pd.read_csv(INPUT_CSV)
    assert "filename" in df.columns and "transcript" in df.columns

    processed_data = []
    total_steps = len(df) + 3

    with tqdm(total=total_steps, desc="Full Pipeline Progress") as pbar:
        # Step 1: Process audio
        for row in tqdm(df.itertuples(index=False),
                        total=len(df),
                        desc="Processing audio",
                        leave=False):
            try:
                processed_data.append(process_row(row.filename, row.transcript))
            except Exception as e:
                print(f"Skipping {row.filename}: {e}")
            pbar.update(1)

        print("\nTotal samples:", len(processed_data))

        # Step 2: Create dataset
        dataset = Dataset.from_list(processed_data)
        dataset = dataset.cast_column("audio", Audio())
        pbar.update(1)

        # Step 3: Split dataset
        split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
        dataset_dict = DatasetDict({
            "train": split_dataset["train"],
            "test": split_dataset["test"],
        })
        print(dataset_dict)
        pbar.update(1)

        # Step 4: Upload to Hugging Face (token passed directly)
        dataset_dict.push_to_hub(HF_DATASET_NAME, token=HF_TOKEN)
        pbar.update(1)


if __name__ == "__main__":
    main()