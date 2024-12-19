from marlin.src.marlin_pytorch import Marlin
import torch
import argparse
import tqdm
import os
import numpy as np
from deepface import DeepFace
import glob
import json 


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

model = Marlin.from_file("marlin_vit_base_ytf", "marlin_vit_base_ytf.encoder.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_dir", type=str, help="Path to the cache file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--num_samples", default=666,type=int, required=False, help="num_samples")
    
    args = parser.parse_args()
    
    # Load or compute keyframe vectors
    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}.")
        exit()
    ground_truth = load_ground_truth(args.ground_truth_file)
    ground_truth = [{k: v} for k, v in ground_truth.items()][:args.num_samples]

    strategies = ['keyframe_marlin', 'keyframe_df_arcface', 'keyframe_df_deepface', 'keyframe_df_facenet', 'keyframe_df_facenet512', 'keyframe_df_vgg']
    model_name = {
        "keyframe_marlin": "marlin",
        "keyframe_df_arcface": "ArcFace",
        'keyframe_df_deepface': 'DeepFace',
        'keyframe_df_facenet': 'Facenet',
        'keyframe_df_facenet512': 'Facenet512',
        'keyframe_df_vgg': 'VGG-Face'
    }

    for strategy in strategies: 
        cache_file = os.path.join(args.cache_dir, strategy + f"_{args.num_samples}")
        cache = load_text_embeddings(cache_file) or {}
        if not cache:
            print(f"Caching strategy: {strategy}")
            embeddings = cache.get(strategy)
            if embeddings is None:
                embeddings = []
                for sample in tqdm.tqdm(ground_truth):
                    video_name, value = [(k,v) for k, v in sample.items()][0]
                    frames = value.get("frames")

                    for frame in frames:
                        if strategy == "keyframe_marlin":
                            embedding = model.extract_image(frame, crop_face=True)
                            embeddings.append(embedding.cpu().flatten().numpy())
                        else:
                            embedding = DeepFace.represent(
                                img_path=frame,
                                model_name=model_name[strategy],
                                detector_backend='skip'
                            )[0].get("embedding")
                            embeddings.append(np.array(embedding))

                embeddings = np.array(embeddings)
                cache[strategy] = embeddings
    
            cache_embeddings(cache_file, cache)
        else:
            print(f"SKIPPED {strategy}")

# python scripts/cache_1.py --video_dir ../data/faces  --ground_truth_file ../data/groundtruth_face.json