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

# Initialize the database connection
DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()



# Function to compute recall metrics
# Compute recall metrics
def compute_recall_metrics(query_vectors, video_embeddings, ground_truth, k_values=[1, 5, 10], is_keyframe=False):
    num_queries = len(query_vectors)
    recall_metrics = {k: 0 for k in k_values}
    ranks = []
    stats = {}

    if is_keyframe:
        video_names = list(video_embeddings.keys())
        print(len(video_names))
        stats = {v_n: 0 for v_n in video_names}
        # Convert video_embeddings from dict to matrix
        embeddings_list = []
        for vectors in video_embeddings.values():
            embeddings_list.append(vectors)
        video_embeddings_matrix = np.vstack(embeddings_list)
    else:
        video_embeddings_matrix = video_embeddings
    # print("video_embedding: ", video_embeddings_matrix.shape)
    
    # Compute similarities for all queries against all video embeddings
    similarities_matrix = cosine_similarity(query_vectors, video_embeddings_matrix)
    # print("cosine matrix", similarities_matrix.shape)

    for i, similarities in tqdm(enumerate(similarities_matrix), total=num_queries):
        # Get the indices that would sort the similarities array in descending order
        top_k_indices = np.argsort(-similarities)[:max(k_values)]
        if is_keyframe:
            for j in top_k_indices:
                video_index = j // len(video_embeddings[video_names[0]])  # Map back to video index
                stats[video_names[video_index]] += 1

        correct_video = ground_truth[i]

        # Check if the correct video is in the top-k results
        if correct_video in top_k_indices:
            rank = np.where(top_k_indices == correct_video)[0][0] + 1
            ranks.append(rank)
            for k in k_values:
                if rank <= k:
                    recall_metrics[k] += 1
        else:
            ranks.append(max(k_values) + 1)  # If not found, rank is set beyond the number of videos

    # Calculate recall for each k and average rank metrics
    for k in k_values:
        recall_metrics[k] /= num_queries

    meanR = np.mean(ranks)
    medR = np.median(ranks)
    recall_metrics['MeanR'] = meanR
    recall_metrics['MedR'] = medR
    
    print(stats)
    return recall_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_file", default="", type=str, help="Path to the cache file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--keyframe_dir", type=str, required=True, help="Directory to save extracted keyframe vectors.")
    args = parser.parse_args()

    # # Load models
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # clip_model, clip_processor = load_clip_model()  # Huggingface CLIP model for Keyframe CLIP-VIP
    # clip_model.to(device)
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # clip_model.load_state_dict(
    #     {key.replace("clipmodel.", ""): value for key, value in checkpoint.items()},  strict=False
    # )

    es = EncoderService()  # EncoderService for CLIP-VIP
    es.init_app(args.checkpoint)

    # Fetch video data from the database
    video_vectors, text_vectors, text_probs, video_names = fetch_video_data(session, limit=30)

    # Load or compute keyframe vectors
    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4'))[:30]
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}.")
        exit()

    keyframe_vectors = {}
    print(f"Loading keyframes...")
    for i, video_path in tqdm(enumerate(video_files)):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframe_vectors[video_name] = es.encode_keyframes(video_path, args.keyframe_dir)
        # keyframe_vectors[video_name] = get_keyframe_vectors(video_path, video_name, args.keyframe_dir, clip_model, clip_processor)

    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth_file)

    # Filter ground truth to only include videos that are in video_names
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in video_names}

    # Load cached text embeddings
    cache = load_text_embeddings(args.cache_file) or {}
    clip_vip_text_embeddings = cache.get('clip_vip_text_embeddings')
    # keyframe_clip_text_embeddings = cache.get('keyframe_clip_text_embeddings')
    
    ground_truth_indices = [video_names.index(k) for k in filtered_ground_truth.keys() for _ in filtered_ground_truth[k]]

    # Prepare text embeddings for CLIP-VIP strategy
    if clip_vip_text_embeddings is None:
        clip_vip_text_embeddings = []
        for video_name, descriptions in filtered_ground_truth.items():
            for description in descriptions:
                text_embedding = es.encode_text(description)
                clip_vip_text_embeddings.append(text_embedding)
        clip_vip_text_embeddings = np.array(clip_vip_text_embeddings)
        cache['clip_vip_text_embeddings'] = clip_vip_text_embeddings

    # Prepare text embeddings for Keyframe CLIP-VIP strategy
    # if keyframe_clip_text_embeddings is None:
    #     keyframe_clip_text_embeddings = []
    #     for video_name, descriptions in filtered_ground_truth.items():
    #         for description in descriptions:
    #             inputs = clip_processor(text=[description], return_tensors="pt", padding=True).to(clip_model.device)
    #             with torch.no_grad():
    #                 text_embedding = clip_model.get_text_features(**inputs).cpu().numpy().flatten()
    #             keyframe_clip_text_embeddings.append(text_embedding)
    #     keyframe_clip_text_embeddings = np.array(keyframe_clip_text_embeddings)
    #     cache['keyframe_clip_text_embeddings'] = keyframe_clip_text_embeddings

    # Save all embeddings to a single cache file
    cache_text_embeddings(cache, args.cache_file)

    # Define strategies
    strategies = ['Keyframe CLIP']

    for strategy in strategies:
        print(f"Evaluating strategy: {strategy}")
        # if strategy == 'CLIP-VIP':
        #     video_embeddings = np.array(video_vectors)
        #     recall_metrics = compute_recall_metrics(clip_vip_text_embeddings, video_embeddings, ground_truth_indices, is_keyframe=False)
        if strategy == 'Keyframe CLIP':
            recall_metrics = compute_recall_metrics(clip_vip_text_embeddings, keyframe_vectors, ground_truth_indices, is_keyframe=True)

        
        # Print recall metrics
        print(f"Recall Metrics for {strategy}: {recall_metrics}")


        

    
# python scripts/compute_metrics.py --video_dir ../data --keyframe_dir ../data/keyframes --ground_truth_file ../data/video_retrieval/msrvtt/train_7k.json --checkpoint ./pretrain_clipvip_base_16.pt --cache_file cache.npz
  