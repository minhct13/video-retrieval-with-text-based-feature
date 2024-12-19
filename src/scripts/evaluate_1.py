import numpy as np
# from video import *
import os
import json
import argparse
# import sqlalchemy
# from sqlalchemy.orm import sessionmaker
# from video import *
import glob
from tqdm import tqdm 
# from marlin.src.marlin_pytorch import Marlin
from compute_funcs import *


def load_text_embeddings(cache_file):

    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as data:
            return {key: data[key] for key in data}
    return None

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_keyframe_vectors(video_name, cache_dir):
    # Define the cache path and check if vectors have been cached already
    cache_path = os.path.join(cache_dir, f"{video_name}_keyframes.npz")
    if os.path.exists(cache_path):
        # print(f"Loading keyframe vectors from cache for {video_name}...")
        return np.load(cache_path)["keyframe_vectors"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_dir", default="", type=str, help="Path to the cache file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--keyframe_dir", type=str, required=True, help="Directory to save extracted keyframe vectors.")
    parser.add_argument("--num_samples", default=666,type=int, required=False, help="num_samples")
    args = parser.parse_args()

    # Load cached text embeddings
    ground_truth = load_ground_truth(args.ground_truth_file)
    print(f"Loaded ground truth for {len(ground_truth)} videos.")
    # Load or compute keyframe vectors
    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}.")
        exit()
    video_names = [k for k in ground_truth.keys()]
    ground_truth_indices = [video_names.index(k) for k in ground_truth.keys() for _ in ground_truth[k]['frames']]

    # Define strategies
    strategies = ['keyframe_marlin', 'keyframe_df_arcface', 'keyframe_df_deepface', 'keyframe_df_facenet', 'keyframe_df_facenet512', 'keyframe_df_vgg']

    for strategy in strategies:
        print(f"Evaluating strategy: {strategy}")
        keyframe_dir = os.path.join(args.keyframe_dir, strategy)
        keyframe_vectors = {}
        for i, video_path in tqdm(enumerate(video_files)):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if video_name in ground_truth.keys():
                keyframe_vector = get_keyframe_vectors(video_name, keyframe_dir)
                keyframe_vectors[video_name] = keyframe_vector
        cache_file = os.path.join(args.cache_dir, strategy + f"_{args.num_samples}.npz")
        cache = load_text_embeddings(cache_file) or {}
        if not cache:
            print(f"Cache {cache_file} not found!")
            exit()
        embeddings = cache.get(strategy)
        recall_metrics = compute_keyframe(embeddings, keyframe_vectors, ground_truth_indices, video_names)
        print(f"Recall Metrics for {strategy}: {recall_metrics}")


#     

# python scripts/face_compute_metrics.py --video_dir ../data/faces  --ground_truth_file ../data/processed_videos_log.json --checkpoint ./pretrain_clipvip_base_16.pt --cache_file cache_face0.npz --keyframe_dir ../data/keyframes/
