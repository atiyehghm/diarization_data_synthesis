import os
import argparse
import subprocess


LIBRISPEECH_URLS = {
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz"
}

def download_librispeech(output_dir="data/librispeech", parts=None):
    os.makedirs(output_dir, exist_ok=True)

    parts = parts or LIBRISPEECH_URLS.keys()

    for part in parts:
        url = LIBRISPEECH_URLS.get(part)
        if not url:
            raise ValueError(f"Invalid part name: {part}")

        print(f"Downloading {part}...")
        tar_path = os.path.join(output_dir, f"{part}.tar.gz")

        subprocess.run([
            "wget", "-q", "--show-progress", "--continue",
            "-O", tar_path, url
        ], check=True)

        print(f"Extracting {part}...")
        subprocess.run([
            "tar", "xzf", tar_path,
            "-C", output_dir,
            "--strip-components=1"
        ], check=True)

        os.remove(tar_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data/librispeech")
    parser.add_argument("--parts", nargs="+", choices=LIBRISPEECH_URLS.keys())
    args = parser.parse_args()

    download_librispeech(args.output_dir, args.parts)
