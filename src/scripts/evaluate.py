import numpy as np
import os
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from src.scripts import process_video
from concurrent.futures import ThreadPoolExecutor
from app import EncoderService
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import torch
from torch.utils.data import DataLoader, Dataset

class TextVideoDataset(Dataset):
    def __init__(self, video_dir, text_dir, encoder_service):
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.text_dir = text_dir
        self.encoder_service = encoder_service
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        text_path = os.path.join(self.text_dir, os.path.basename(video_path).replace(".mp4", ".txt"))
        
        if not os.path.exists(text_path):
            return None
        
        texts, probs = self.parse_text_file(text_path)
        text_vectors = [self.encoder_service.encode_text(text) for text in texts]
        video_vector = self.encoder_service.encode_video(video_path)
        
        return video_path, video_vector, text_vectors, probs
    
    def parse_text_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            texts = [line.rsplit('|', 1)[0].strip() for line in lines]
            probs = [float(line.rsplit('|', 1)[1].strip()) for line in lines]
        return texts, probs

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

def load_ground_truth(file_path):
    df = pd.read_csv(file_path)
    video_ids = df['video_id'].tolist()
    return video_ids

def compute_metrics(similarity_matrix, ground_truth_indices):
    # Get sorted indices based on similarity scores (highest scores first)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    # Find the rank of the correct match for each query
    matches = np.array([np.where(sorted_indices[i] == ground_truth_indices[i])[0][0] for i in range(len(ground_truth_indices))])
    # Calculate recall metrics
    r1 = float(np.sum(matches == 0)) / len(matches)
    r5 = float(np.sum(matches < 5)) / len(matches)
    r10 = float(np.sum(matches < 10)) / len(matches)
    medr = np.median(matches) + 1
    meanr = np.mean(matches) + 1
    
    return r1, r5, r10, medr, meanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing text files.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth CSV file.")

    args = parser.parse_args()

    es = EncoderService(args.checkpoint)
    
    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]

    futures = []
    with ThreadPoolExecutor() as executor:
        for video_path in video_files:
            text_path = os.path.join(args.text_dir, os.path.basename(video_path).replace(".mp4", ".txt"))
            if not os.path.exists(text_path):
                continue

            texts, probs = TextVideoDataset(args.video_dir, args.text_dir, es).parse_text_file(text_path)
            text_vectors = [es.encode_text(text) for text in texts]
            text_probs = probs

            futures.append(
                executor.submit(
                    process_video,
                    es,
                    video_path,
                    text_vectors,
                    text_probs
                )
            )

    # Collect results
    video_vectors = []
    text_embeddings = []
    text_embeddings_list = []
    
    for future in futures:
        video_vector, text_vector, text_probs = future.result()
        aggregated_vector = aggregate_vectors(video_vector, text_vector, text_probs)
        video_vectors.append(aggregated_vector)
        text_embeddings_list.extend(text_vector)

    # Convert lists to numpy arrays
    video_embeddings = np.array(video_vectors)
    text_embeddings = np.array(text_embeddings_list)

    # Load ground truth and compute similarity
    ground_truth = load_ground_truth(args.ground_truth_file)
    sim_matrix = cosine_similarity(text_embeddings, video_embeddings)

    # Assuming each video has 5 corresponding text queries in the ground truth
    ground_truth_indices = np.array([i // 5 for i in range(5 * len(video_embeddings))])

    v2tr1, v2tr5, v2tr10, v2tmedr, v2tmeanr = compute_metrics(sim_matrix.T, ground_truth_indices)
    t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = compute_metrics(sim_matrix, ground_truth_indices)

    print(f"Video-to-Text R@1: {v2tr1}, R@5: {v2tr5}, R@10: {v2tr10}, MedR: {v2tmedr}, MeanR: {v2tmeanr}")
    print(f"Text-to-Video R@1: {t2vr1}, R@5: {t2vr5}, R@10: {t2vr10}, MedR: {t2vmedr}, MeanR: {t2vmeanr}")

# python compute_metrics.py --ground_truth_file ground_truth.csv --text_embeddings_file text_embeddings.npy --video_embeddings_file video_embeddings.npy
