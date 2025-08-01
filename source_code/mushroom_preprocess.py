# mushroom_preprocess.py
import os
import requests

def download_mushroom_data(data_dir="data/"):
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/"
    files = ["agaricus-lepiota.data", "agaricus-lepiota.names"]
    file_paths = {}

    for file in files:
        url = base_url + file
        file_path = os.path.join(data_dir, file)
        file_paths[file] = file_path

        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved: {file_path}")
            else:
                raise Exception(f"❌ Failed to download {file} (status code {response.status_code})")
        else:
            print(f"🟢 {file} already exists at: {file_path}")

    return file_paths["agaricus-lepiota.data"]  # Only return .data path for cleaning step
