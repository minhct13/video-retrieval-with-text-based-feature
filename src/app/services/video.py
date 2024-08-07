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

    videos = session.query(Video).all()
    for video in videos:
        video_vectors.append(video.video_vector)
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

    return video_vectors, text_vectors, text_probs, video_names

class VideoService:
    def __init__(self):
        self.es = current_app.es
        self.session = current_app.db.session
        self.video_vectors, self.text_vectors, self.text_probs, self.video_names = fetch_video_data(self.session)
        self.aggregated_vectors = self.aggregate_vectors()

    def aggregate_vectors(self):
        aggregated_vectors = []
        for video_vector, text_vectors, text_probs in zip(self.video_vectors, self.text_vectors, self.text_probs):
            aggregated_vector = np.mean([video_vector] + [t * p for t, p in zip(text_vectors, text_probs)], axis=0)
            aggregated_vectors.append(aggregated_vector)
        return np.array(aggregated_vectors)

    def query(self, queries: str, top_n=20):
        """Retrieve video names and paths based on the query.

        Args:
            queries (str): The input query text.
            top_n (int): The number of top results to return.

        Returns:
            str: JSON response with video names and paths.
        """
        res = []

        # Encode the input query
        query_embedding = self.es.encode_text(queries)
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

    def suggest(self):
        """
        Return suggestions
        """
        return {
            "data" : [
                "Playing tennis",
                "",
                "What is the setting or location where the video takes place?",
                "What objects or items are prominently featured in the video?",
                "What is the overall mood or atmosphere of the video?",
            ]
        }, requests.codes.ok


    def get_video_path(self, video_name):
        # Dummy implementation, replace with actual logic to get video path
        return f"{current_app.config['VIDEO_URL']}/{video_name}"