import numpy as np
# from video import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 

def compute_keyframe(query_vectors, video_embeddings, ground_truth, video_names ,k_values=[1, 5, 10]):
    num_queries = len(query_vectors)
    recall_metrics = {k: 0 for k in k_values}
    ranks = []
    # Convert video_embeddings from dict to matrix and create a mapping from index to video name
    embeddings_list = []
    index_2_video_id = {}
    start_idx = 0
    for video_name, vectors in video_embeddings.items():
        embeddings_list.append(vectors)
        num_keyframes = vectors.shape[0]
        for idx in range(start_idx, start_idx + num_keyframes):
            index_2_video_id[idx] = video_name
        start_idx += num_keyframes
    video_embeddings_matrix = np.vstack(embeddings_list)
    
    # Compute similarities for all queries against all video embeddings
    similarities_matrix = cosine_similarity(query_vectors, video_embeddings_matrix)
    for i, similarities in tqdm(enumerate(similarities_matrix), total=num_queries):
        # Get the indices that would sort the similarities array in descending order
        top_k_indices = np.argsort(-similarities)[:max(k_values)].tolist()
        
        top_k_videos = [index_2_video_id[j] for j in top_k_indices]
        
        correct_video = video_names[ground_truth[i]].replace(".mp4","")
        # print(top_k_videos, correct_video, correct_video in top_k_videos)
        # Check if the correct video is in the top-k results
        if correct_video in top_k_videos:
            rank = top_k_videos.index(correct_video) + 1
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
    
    return recall_metrics

def compute_marlin(embeddings, video_vectors, ground_truth ,k_values=[1, 5, 10]):
    num_queries = len(embeddings)
    recall_metrics = {k: 0 for k in k_values}
    ranks = []
    # Compute similarities for all queries against all video embeddings
    similarities_matrix = cosine_similarity(embeddings, video_vectors)
    for i, similarities in tqdm(enumerate(similarities_matrix), total=num_queries):
        # Get the indices that would sort the similarities array in descending order
        top_k_indices = np.argsort(-similarities)[:max(k_values)].tolist()
        # top_k_videos = [index_2_video_id[j] for j in top_k_indices]
        correct_video = ground_truth[i]
        # Check if the correct video is in the top-k results
        if correct_video in top_k_indices:
            rank = top_k_indices.index(correct_video) + 1
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
    
    return recall_metrics

def compute_text_video(
        image_embeddings,
        text_embeddings,
        video_vectors,
        text_vectors,
        ground_truth,
        coefficients,
        k_values=[1, 5, 10]
    ):
    num_image_queries = len(image_embeddings)
    num_text_queries = len(text_embeddings)
    
    recall_metrics = {k: 0 for k in k_values}
    ranks = []

    # Compute cosine similarities separately
    image_video_similarities = cosine_similarity(image_embeddings, video_vectors)  # [num_images x num_videos]
    
    # Compute cosine similarity between text queries and text vectors
    text_video_similarities = []
    for j in range(text_vectors.shape[1]):
        similarity = cosine_similarity(text_embeddings, text_vectors[:, j])
        text_video_similarities.append(similarity)  # [text_vector_dim x num_texts x num_videos]
    text_video_similarities = np.array(text_video_similarities)  # Shape: [text_vector_dim x num_texts x num_videos]

    for i in tqdm(range(num_image_queries)):
        # Image query similarity with video vectors
        sim_video_from_images = image_video_similarities[i]  # [num_videos]

        # If the number of text queries is less than the number of image queries, repeat the last text query
        text_index = min(i, num_text_queries - 1)

        # Aggregate the text similarities across the dimensions
        sim_video_from_texts = np.sum([coefficients[j + 1] * text_video_similarities[j, text_index] for j in range(text_video_similarities.shape[0])], axis=0)

        # Combine the similarities from images and texts using coefficients
        total_similarity = (
            coefficients[0] * sim_video_from_images +
            sim_video_from_texts  # Already weighted in the sum
        )  # [num_videos]

        # Rank videos based on total similarity
        top_k_indices = np.argsort(-total_similarity)[:max(k_values)]

        # Evaluate the rank of the correct video
        correct_video = ground_truth[i]
        rank = np.where(top_k_indices == correct_video)[0][0] + 1 if correct_video in top_k_indices else max(k_values) + 1
        ranks.append(rank)

        for k in k_values:
            if correct_video in top_k_indices[:k]:
                recall_metrics[k] += 1

    # Normalize recall metrics by the number of queries
    for k in k_values:
        recall_metrics[k] /= num_image_queries

    recall_metrics['MeanR'] = np.mean(ranks)
    recall_metrics['MedR'] = np.median(ranks)

    return recall_metrics