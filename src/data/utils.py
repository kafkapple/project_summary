import os
import json
import requests
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import DialogueDataset

def download_file(url: str, save_path: str):
    """파일 다운로드 함수"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_dialogsum(data_dir: str):
    """DialogSum 데이터셋 다운로드"""
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data"
    files = {
        'train.json': f"{base_url}/dialogsum.train.jsonl",
        'val.json': f"{base_url}/dialogsum.dev.jsonl"
    }
    
    for filename, url in files.items():
        save_path = os.path.join(data_dir, filename)
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            try:
                download_file(url, save_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists.")

def load_jsonl(file_path: str) -> list:
    """JSONL 파일을 로드합니다."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data 

def create_dataloader(data, tokenizer, batch_size, max_length=1024, shuffle=True):
    dataset = DialogueDataset(data, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 