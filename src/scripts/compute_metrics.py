import numpy as np
import os
import json
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from app import EncoderService
from app.models import Video
import sqlalchemy
from sqlalchemy.orm import sessionmaker


def aggregate_vectors(video_vector, text_vectors, text_probs, video_weight=60, text_weight=40):
    """
    Aggregate the video vector with text vectors based on given weights.

    Args:
        video_vector (np.ndarray): The video vector.
        text_vectors (list of np.ndarray): List of text vectors.
        text_probs (list of float): List of text probabilities.
        video_weight (float): Weight for the video vector.
        text_weight (float): Weight for the text vectors.

    Returns:
        np.ndarray: Aggregated vector.
    """
    # Convert text_vectors and text_probs to numpy arrays
    text_vectors = np.array(text_vectors)
    text_probs = np.array(text_probs)
    # Normalize text_probs to sum up to 1
    text_probs = text_probs / np.sum(text_probs)
    # Compute weighted text vector
    weighted_text_vector = np.sum(text_vectors * text_probs[:, np.newaxis], axis=0)
    # Normalize the weights
    video_weight = video_weight / 100
    text_weight = text_weight / 100
    # Compute final aggregated vector
    aggregated_vector = video_weight * video_vector + text_weight * weighted_text_vector
    return aggregated_vector

# Initialize the database connection
DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

# Function to load ground truth from JSON file
def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def fetch_video_data(session):
    videos = session.query(Video).all()
    video_vectors = []
    text_vectors = []
    text_probs = []
    video_names = []

    for video in videos:
        video_vectors.append(video.video_vector)
        text_vectors.append([
            video.text_vector_1,
            video.text_vector_2,
            video.text_vector_3,
            video.text_vector_4,
            video.text_vector_5,
        ])
        text_probs.append([
            video.text_prob_1,
            video.text_prob_2,
            video.text_prob_3,
            video.text_prob_4,
            video.text_prob_5,
        ])
        video_names.append(video.name)
    
    return np.array(video_vectors), np.array(text_vectors), np.array(text_probs), video_names

def get_top_k_retrieved(text_embedding, video_embeddings, k=10):
    similarities = cosine_similarity([text_embedding], video_embeddings)[0]
    top_k_indices = np.argsort(-similarities)[:k]
    return top_k_indices

def compute_recall_metrics(text_embeddings, video_embeddings, ground_truth, k_values=[1, 5, 10]):
    num_queries = len(text_embeddings)
    recall_metrics = {k: 0 for k in k_values}

    for i, text_embedding in enumerate(text_embeddings):
        top_k_retrieved = get_top_k_retrieved(text_embedding, video_embeddings, max(k_values))
        correct_video = ground_truth[i]

        for k in k_values:
            if correct_video in top_k_retrieved[:k]:
                recall_metrics[k] += 1

    for k in k_values:
        recall_metrics[k] /= num_queries

    return recall_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_file", type=str, required=True, help="Path to the cache file.")
    args = parser.parse_args()

    es = EncoderService()
    es.init_app(args.checkpoint)

    # Check if cache exists
    if os.path.exists(args.cache_file):
        print("Loading embeddings from cache...")
        with np.load(args.cache_file, allow_pickle=True) as data:
            video_vectors = data['video_vectors']
            text_vectors = data['text_vectors']
            text_probs = data['text_probs']
            video_names = data['video_names'].tolist()
            text_embeddings = data['text_embeddings']
            ground_truth_video_indices = data['ground_truth_video_indices']
            ground_truth_text_indices = data['ground_truth_text_indices']
    else:
        # Fetch video and text vectors from the database
        video_vectors, text_vectors, text_probs, video_names = fetch_video_data(session)

        # Load ground truth and filter by existing records in the database
        ground_truth = load_ground_truth(args.ground_truth_file)
        filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in video_names}

        if not filtered_ground_truth:
            print("No matching records found between ground truth and database.")
            exit()

        # video_embeddings = np.array(video_vectors)

        # Encode ground truth text descriptions using EncoderService
        text_embeddings = []
        ground_truth_pairs = []

        for video_name, descriptions in filtered_ground_truth.items():
            for description in descriptions:
                encoded_text = es.encode_text(description)
                text_embeddings.append(encoded_text)
                ground_truth_pairs.append((video_names.index(video_name), len(text_embeddings) - 1))

        text_embeddings = np.array(text_embeddings)

        # Extract indices for ground truth pairs
        ground_truth_video_indices = np.array([pair[0] for pair in ground_truth_pairs])
        ground_truth_text_indices = np.array([pair[1] for pair in ground_truth_pairs])

        # Save embeddings and ground truth indices to cache
        print("Saving embeddings to cache...")
        np.savez(args.cache_file, video_vectors=video_vectors, text_vectors=text_vectors,
                 text_probs=text_probs, video_names=video_names, text_embeddings=text_embeddings,
                 ground_truth_video_indices=ground_truth_video_indices, ground_truth_text_indices=ground_truth_text_indices)

    # Compute aggregated vectors
    aggregated_vectors = [aggregate_vectors(v, t, p) for v, t, p in zip(video_vectors, text_vectors, text_probs)]
    # Convert aggregated vectors to numpy array
    # video_embeddings = np.array(aggregated_vectors)
    video_embeddings = np.array(video_vectors)

    # Compute recall metrics
    recall_metrics = compute_recall_metrics(text_embeddings, video_embeddings, ground_truth_video_indices)
    print(f"Recall Metrics: {recall_metrics}")
    
# python scripts/compute_metrics.py --ground_truth_file ../data/video_retrieval/msrvtt/train_7k.json --checkpoint ../pretrain_clipvip_base_16.pt
  