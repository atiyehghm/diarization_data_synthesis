import os
import csv
import soundfile as sf

# Path to your folder
audio_folder = "/home/atiyehghm/Desktop/hamseda/hamseda-test/speech_benchmark/customer_service_diarization_dataset/data"
source_dataset = "customer_service"

# Prepare CSV output
csv_file = "metadata.csv"
csv_columns = ["file_path", "annotation_path", "type", "audio_duration", "source_dataset", "num_speakers"]

def get_num_speakers(rttm_path):
    speakers = set()
    with open(rttm_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                speaker_id = parts[7]  # RTTM format: speaker_id is 8th column
                speakers.add(speaker_id)
    return len(speakers)

def get_audio_duration(audio_path):
    with sf.SoundFile(audio_path) as f:
        return len(f) / f.samplerate

rows = []

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file)
        rttm_path = audio_path + ".rttm"
        if not os.path.exists(rttm_path):
            print(f"Warning: RTTM file missing for {audio_path}")
            continue
        
        num_speakers = get_num_speakers(rttm_path)
                # Extract type from filename
        filename_lower = file.lower()
        if "monologue" in filename_lower or "monologue" in filename_lower:
            audio_type = "monologue"
        elif "dialog" in filename_lower:
            audio_type = "dialog"
        elif "overlap" in filename_lower:
            audio_type = "overlap"
        else:
            audio_type = "unknown"
        
        duration = get_audio_duration(audio_path)
        
        rows.append({
            "file_path": audio_path,
            "annotation_path": rttm_path,
            "type": audio_type,
            "audio_duration": round(duration, 3),
            "source_dataset": source_dataset,
            "num_speakers": num_speakers
        })

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Metadata CSV saved as {csv_file} with {len(rows)} entries.")
