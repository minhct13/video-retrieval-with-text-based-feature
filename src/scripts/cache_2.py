import torch
from marlin.src.marlin_pytorch import Marlin
import glob
import tqdm
from video import EncoderService
import numpy as np
import os
import json
import argparse


model = Marlin.from_file("marlin_vit_base_ytf", "marlin_vit_base_ytf.encoder.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
es = EncoderService()
es.init_app("pretrain_clipvip_base_16.pt")

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_text_embeddings(cache_file):
    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as data:
            return {key: data[key] for key in data}
    return None

def cache_embeddings(cache_file, cache):
    """
    Save dynamic key text embeddings to a compressed NPZ file.
    
    Args:
        cache_file (str): The path to the file where embeddings will be cached.
        **kwargs: Key-value pairs where the key is a string identifier and the value is the embedding.
    """
    np.savez_compressed(cache_file, **cache)

def load_json_files(json_dir):
    data = {}
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data.update(json.load(f))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_dir", type=str, help="Path to the cache file.")
    parser.add_argument("--num_samples", default=1000, type=int, required=False, help="num_samples")
    
    args = parser.parse_args()
    
    ground_truth = load_ground_truth(args.ground_truth_file)
    ground_truth = [{k: v} for k, v in ground_truth.items()]

    types = ['description', 'frame']
    video_text_data = load_json_files("../data/kaggle/working/output") 
    for data_type in types: 
        cache_file = os.path.join(args.cache_dir, data_type + f"_{args.num_samples}")
        cache = load_text_embeddings(cache_file) or {}
        if not cache:
            print(f"Caching data type: {data_type}")
            embeddings = cache.get(data_type)
            if embeddings is None:
                embeddings = []
                for sample in tqdm.tqdm(ground_truth):
                    video_name, value = [(k,v) for k, v in sample.items()][0]
                    if video_name.replace(".mp4", "") not in video_text_data.keys():
                        continue
                    if data_type == 'frame':
                        frame = value[0].get('frame')
                        embedding = model.extract_image(frame)
                        for _ in range(len(value)):
                            embeddings.append({video_name: embedding.cpu().flatten().numpy()})
                    elif data_type == 'description':
                        descriptions = [v.get('description') for v in value]
                        for description in descriptions:
                            embedding = es.encode_text(description)
                            embeddings.append({video_name: embedding})


                # embeddings = np.array(embeddings)
                # print(embeddings.shape)
                cache[data_type] = embeddings
    
            cache_embeddings(cache_file, cache)
        else:
            print(f"SKIPPED {strategy}")


# python scripts/cache_2.py --ground_truth_file ../data/groundtruth_face_crops.json --cache_dir cache/ --num_samples 5