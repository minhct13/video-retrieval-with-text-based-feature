import numpy as np
# from video import *
import os
import json
import argparse
from video import *
import glob
from tqdm import tqdm 
# from marlin.src.marlin_pytorch import Marlin
from compute_funcs import *
import sqlalchemy
from sqlalchemy.orm import sessionmaker


DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_dir", default="", type=str, help="Path to the cache file.")
    parser.add_argument("--num_samples", default=1000, type=int, required=False, help="num_samples")
    args = parser.parse_args()

    video_vectors, text_vectors, text_probs, video_face_vectors, video_names = fetch_video_data(session, limit=args.num_samples)

    # Load cached text embeddings
    ground_truth = load_ground_truth(args.ground_truth_file)
    ground_truth = {k: v for k, v in ground_truth.items() if k in video_names}
    ground_truth_indices = [video_names.index(k) for k in ground_truth.keys() for _ in ground_truth[k]]
    # video_names = list(ground_truth.keys())

    
    for data_type in ['description', 'frame']:
        cache_file = os.path.join(args.cache_dir, data_type + f"_{args.num_samples}.npz")
        cache = load_text_embeddings(cache_file) or {}
        if not cache:
            print(f"Cache {cache_file} not found!")
            exit()
        if data_type == 'description':
            text_embeddings = cache.get("description").tolist()
        else:
            image_embeddings = cache.get("frame").tolist()
        print(f"Loaded cache for {data_type} as {cache_file}!")
    
    image_embeddings = [value for item in image_embeddings for value in item.values()]
    text_embeddings = [value for item in text_embeddings for value in item.values()]
    # print(len(image_embeddings))
    # Define strategies
    strategies = ['text', 'marlin', 'marlin_text_1', 'marlin_text_3', 'marlin_text_5']
    for strategy in strategies:
        if strategy == 'text':
            coefficients = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            recall_metrics = compute_text_video(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_vectors=video_face_vectors,
                text_vectors=text_vectors,
                ground_truth=ground_truth_indices,
                coefficients=coefficients
            )
        elif strategy == 'marlin':
            coefficients = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            recall_metrics = compute_text_video(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_vectors=video_face_vectors,
                text_vectors=text_vectors,
                ground_truth=ground_truth_indices,
                coefficients=coefficients
            )
        elif strategy == 'marlin_text_1':
            coefficients = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
            recall_metrics = compute_text_video(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_vectors=video_face_vectors,
                text_vectors=text_vectors,
                ground_truth=ground_truth_indices,
                coefficients=coefficients
            )
        elif strategy == 'marlin_text_3':
            coefficients = [0.4, 0.3, 0.3, 0.3, 0.0, 0.0]
            recall_metrics = compute_text_video(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_vectors=video_face_vectors,
                text_vectors=text_vectors,
                ground_truth=ground_truth_indices,
                coefficients=coefficients
            )
        elif strategy == 'marlin_text_5':
            coefficients = [0.4, 0.2, 0.2, 0.2, 0.2, 0.2]
            recall_metrics = compute_text_video(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_vectors=video_face_vectors,
                text_vectors=text_vectors,
                ground_truth=ground_truth_indices,
                coefficients=coefficients
            )
        print(f"Recall Metrics for {strategy}: {recall_metrics}")