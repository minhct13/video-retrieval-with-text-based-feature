import numpy as np
from video import *
import numpy as np
import os
import json
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from video import EncoderService, Video
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from video import *
import glob
from tqdm import tqdm 
from compute_metrics import compute_recall_metrics
DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()



def compute_recall_metrics_jsp(query_vectors, video_vectors, text_vectors, ground_truth, coefficients, k_values=[1, 5, 10]):
    num_queries = len(query_vectors)
    recall_metrics = {k: 0 for k in k_values}
    ranks = []

    # Compute cosine similarities
    query_video_similarities = cosine_similarity(query_vectors, video_vectors)  # [q x v]
    query_text_similarities = np.array([cosine_similarity(query_vectors, text_vectors[:, j]) for j in range(text_vectors.shape[1])])  # [5 x q x v]

    for i, query_vector in tqdm(enumerate(query_vectors), total=num_queries):
        sim_video = query_video_similarities[i]  # [v]
        sim_generated = [query_text_similarities[j, i] for j in range(query_text_similarities.shape[0])]  # [5 x v]

        # Compute the total similarity
        total_similarity = coefficients[0] * sim_video + sum(c * sg for c, sg in zip(coefficients[1:], sim_generated))  # [v]
        top_k_indices = np.argsort(-total_similarity)[:max(k_values)]

        correct_video = ground_truth[i]
        rank = np.where(top_k_indices == correct_video)[0][0] + 1 if correct_video in top_k_indices else max(k_values) + 1
        ranks.append(rank)

        for k in k_values:
            if correct_video in top_k_indices[:k]:
                recall_metrics[k] += 1

    for k in k_values:
        recall_metrics[k] /= num_queries

    recall_metrics['MeanR'] = np.mean(ranks)
    recall_metrics['MedR'] = np.median(ranks)

    return recall_metrics

def compute_recall_metrics_jvb(query_vectors, video_vectors, text_vectors, text_probs, ground_truth, k_values=[1, 5, 10], video_weight=70, text_weight=30):
    num_queries = len(query_vectors)
    recall_metrics = {k: 0 for k in k_values}
    ranks = []

    # Stack video vectors and text vectors for vectorized similarity calculation
    video_vectors_matrix = np.vstack(video_vectors)
    text_vectors_matrix = [np.vstack(tv) for tv in text_vectors]

    # Aggregate vectors before computing cosine similarity
    combined_vectors = []
    for video_vector, prob in zip(video_vectors_matrix, text_probs):
        combined_vector = aggregate_vectors(video_vector, text_vectors_matrix, prob, video_weight, text_weight)
        combined_vectors.append(combined_vector)
    
    combined_vectors_matrix = np.vstack(combined_vectors)

    # Compute cosine similarities between all query vectors and combined vectors
    similarities_matrix = cosine_similarity(query_vectors, combined_vectors_matrix)

    for i, similarities in tqdm(enumerate(similarities_matrix), total=num_queries):
        # Get the indices that would sort the similarities array in descending order
        top_k_indices = np.argsort(-similarities)[:max(k_values)]

        correct_video = ground_truth[i]
        rank = np.where(top_k_indices == correct_video)[0][0] + 1 if correct_video in top_k_indices else max(k_values) + 1
        ranks.append(rank)

        for k in k_values:
            if correct_video in top_k_indices[:k]:
                recall_metrics[k] += 1

    for k in k_values:
        recall_metrics[k] /= num_queries

    recall_metrics['MeanR'] = np.mean(ranks)
    recall_metrics['MedR'] = np.median(ranks)

    return recall_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_file", default="", type=str, help="Path to the cache file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    args = parser.parse_args()


    es = EncoderService()  # EncoderService for CLIP-VIP
    es.init_app(args.checkpoint)

    # Fetch video data from the database
    video_vectors, text_vectors, text_probs, video_names = fetch_video_data(session, limit=1000)

    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth_file)

    # Filter ground truth to only include videos that are in video_names
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in video_names}

    # Load cached text embeddings
    cache = load_text_embeddings(args.cache_file) or {}
    clip_vip_text_embeddings = cache.get('clip_vip_text_embeddings')
    
    ground_truth_indices = [video_names.index(k) for k in filtered_ground_truth.keys() for _ in filtered_ground_truth[k]]

    # Prepare text embeddings for CLIP-VIP strategy
    if clip_vip_text_embeddings is None:
        clip_vip_text_embeddings = []
        for video_name, descriptions in filtered_ground_truth.items():
            for description in descriptions:
                text_embedding = es.encode_text(description)
                clip_vip_text_embeddings.append(text_embedding)
        clip_vip_text_embeddings = np.array(clip_vip_text_embeddings)
        # cache['clip_vip_text_embeddings'] = clip_vip_text_embeddings


    # Save all embeddings to a single cache file
    # cache_text_embeddings(cache, args.cache_file)

    # Define strategies
    strategies = ['CLIP-VIP', 'JVB', 'JSB']

    # Coefficients for JSP
    coefficients = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]

    for strategy in strategies:
        print(f"Evaluating strategy: {strategy}")
        if strategy == 'CLIP-VIP':
            video_embeddings = np.array(video_vectors)
            recall_metrics = compute_recall_metrics(clip_vip_text_embeddings, video_embeddings, ground_truth_indices, is_keyframe=False)
        if strategy == 'JVB':
            recall_metrics = compute_recall_metrics_jvb(clip_vip_text_embeddings, video_vectors, text_vectors, text_probs, ground_truth_indices)
        elif strategy == 'JSB':
            recall_metrics = compute_recall_metrics_jsp(clip_vip_text_embeddings, video_vectors, text_vectors, ground_truth_indices, coefficients)
        
        # Print recall metrics
        print(f"Recall Metrics for {strategy}: {recall_metrics}")



# python scripts/jsp_jvp.py --video_dir ../data  --ground_truth_file ../data/video_retrieval/msrvtt/train_7k.json --checkpoint ./pretrain_clipvip_base_16.pt --cache_file cache.npz
