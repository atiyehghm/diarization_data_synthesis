from datasets import Dataset, Audio
import pandas as pd
import os

df = pd.read_csv('/home/atiyehghm/Desktop/hamseda/hamseda-test/speech_benchmark/customer_service_persian_diarization_dataset/metadata.csv')

data_dict = df.to_dict(orient="list")
dataset = Dataset.from_dict(data_dict)

dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

# Push to Hub (replace with your repo name)
dataset.push_to_hub("atiyehghm/customer_service_persian_diarization_dataset")