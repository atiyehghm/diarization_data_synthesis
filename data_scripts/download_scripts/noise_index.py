import soundfile as sf
import os
import json

index = []
for root, _, files in os.walk(os.environ["OUTPUT_DIR"]):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            info = sf.info(path)
            index.append({
                "path": path,
                "duration": info.duration,
                "type": "noise"
            })

with open(f"{os.environ['OUTPUT_DIR']}/noise_index.json", "w") as f:
    json.dump(index, f)
