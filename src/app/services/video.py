import requests
import json
import numpy as np
import os
from flask import current_app
from app.models import Video
from sklearn.metrics.pairwise import cosine_similarity


def fetch_video_data(session):
    video_vectors = []
    text_vectors = []
    text_probs = []
    video_names = []
    face_video_vectors = []
    videos = session.query(Video).filter(
        Video.video_vector.isnot(None),
        Video.face_video_vector.isnot(None)
    ).all()
    for video in videos:
        video_vectors.append(video.video_vector)
        face_video_vectors.append(video.face_video_vector)
        text_vectors.append([
            video.text_vector_1,
            video.text_vector_2,
            video.text_vector_3,
            video.text_vector_4,
            video.text_vector_5
        ])
        text_probs.append([
            video.text_prob_1,
            video.text_prob_2,
            video.text_prob_3,
            video.text_prob_4,
            video.text_prob_5
        ])
        video_names.append(video.name)

    return video_vectors, face_video_vectors, text_vectors, text_probs, video_names

class VideoService:
    def __init__(self):
        self.es = current_app.es
        self.session = current_app.db.session
        self.video_vectors, self.face_video_vectors, self.text_vectors, self.text_probs, self.video_names = fetch_video_data(self.session)
    #     self.aggregated_vectors = self.aggregate_vectors()

    # def aggregate_vectors(self):
    #     aggregated_vectors = []
    #     for video_vector, text_vectors, text_probs in zip(self.video_vectors, self.text_vectors, self.text_probs):
    #         aggregated_vector = np.mean([video_vector] + [t * p for t, p in zip(text_vectors, text_probs)], axis=0)
    #         aggregated_vectors.append(aggregated_vector)
    #     return np.array(aggregated_vectors)

    def query_by_text(self, query: str, top_n=20):
        """Retrieve video names and paths based on the query.

        Args:
            query (str): The input query text.
            top_n (int): The number of top results to return.

        Returns:
            str: JSON response with video names and paths.
        """
        res = []

        # Encode the input query
        query_embedding = self.es.encode_text(query)
        # Compute cosine similarity between the query and video embeddings
        # Compute cosine similarity between the query and video embeddings
        similarities = cosine_similarity([query_embedding], self.video_vectors)[0]

        # Get the top N indices
        top_n_indices = np.argsort(-similarities)[:top_n]

        # Construct the response
        res = {
            "data": [{
                "video_name": self.video_names[i],
                "video_path": self.get_video_path(self.video_names[i]),
                "similarity": float(similarities[i])  # Ensure similarity is a float
            } for i in top_n_indices]
        }

        return json.dumps(res), requests.codes.ok
    
    def query_by_image(self, image, top_n=20):
        """Retrieve video names and paths based on the input image.

        Args:
            image (str): The path to the input image.
            top_n (int): The number of top results to return.

        Returns:
            str: JSON response with video names and paths.
        """
        res = []

        # Step 1: Encode the input image
        image_embedding = self.es.encode_image(image)

        # Step 2: Compute cosine similarity between the image embedding and face video vectors
        similarities = cosine_similarity([image_embedding], self.face_video_vectors)[0]

        # Step 3: Get the top N indices based on similarity
        top_n_indices = np.argsort(-similarities)[:top_n]

        # Construct the response
        res = {
            "data": [{
                "video_name": self.video_names[i],
                "video_path": self.get_video_path(self.video_names[i]),
                "similarity": float(similarities[i])  # Ensure similarity is a float
            } for i in top_n_indices]
        }

        return json.dumps(res), requests.codes.ok

    
    def query_by_image_text(self, query: str, image, top_n=20, coefficients=[0.8, 0.2]):
        """Retrieve video names and paths based on the query and image.

        Args:
            query (str): The input query text.
            image (str): The path to the input image.
            top_n (int): The number of top results to return.
            coefficients (list): Coefficients for combining image and text similarities.

        Returns:
            str: JSON response with video names and paths.
        """
        res = []
        query_embedding = self.es.encode_text(query)
        image_embedding = self.es.encode_image(image)
        # Calculate similarity between the query embedding and video vectors
        sim_video_from_queries = cosine_similarity([query_embedding], self.video_vectors)[0]
        # Calculate similarity between the image embedding and video vectors
        sim_video_from_images = cosine_similarity([image_embedding], self.face_video_vectors)[0]
        # Step 4: Combine the similarities using coefficients
        total_similarity = (
            coefficients[0] * sim_video_from_images +
            coefficients[1] * sim_video_from_queries
        )

        # Step 5: Get the top N indices based on total similarity
        top_n_indices = np.argsort(-total_similarity)[:top_n]

        # Construct the response
        res = {
            "data": [{
                "video_name": self.video_names[i],
                "video_path": self.get_video_path(self.video_names[i]),
                "similarity": float(total_similarity[i])  # Ensure similarity is a float
            } for i in top_n_indices]
        }

        return json.dumps(res), requests.codes.ok


    def suggest(self):
        """
        Return suggestions
        """
        return {
            "data" : [
                "A man is singing and standing in the road",
                "Three woman doing a fashion show to music",
                "Cartoon birds are flying",
                "Two astronauts experiencing a tense situation before relaxing afterwards",
                "A guy reports on complex news",
            ]
        }, requests.codes.ok


    def get_video_path(self, video_name):
        # Dummy implementation, replace with actual logic to get video path
        return f"{current_app.config['VIDEO_URL']}/{video_name}"