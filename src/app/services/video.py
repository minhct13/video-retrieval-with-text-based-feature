import requests
import json
import numpy as np
import os
from flask import current_app
from app.models import Video, VideoKeyframe
from sklearn.metrics.pairwise import cosine_similarity
from time import time

def fetch_video_data(session):
    video_ids = []
    frame_indexs = []
    clip_vip_vectors = []
    marlin_video_vectors = []
    # clip_description_vector_1 = []
    # clip_description_vector_3  = []
    # clip_description_vector_5 = []
    # datasets = []
    # video_names = []

    videos = session.query(VideoKeyframe).filter(
        VideoKeyframe.clip_vip_vector.isnot(None),
        VideoKeyframe.marlin_video_vector.isnot(None)
    ).all()
    
    for video in videos:
        clip_vip_vectors.append(video.clip_vip_vector)
        # clip_description_vector_1.append(video.clip_description_vector_1)
        # clip_description_vector_3.append(video.clip_description_vector_3)
        # clip_description_vector_5.append(video.clip_description_vector_5)
        marlin_video_vectors.append(video.marlin_video_vector)
        # datasets.append(video.dataset)
        # video_names.append(video.name)  # Add video name to the list
        video_ids.append(video.video_id)
        frame_indexs.append(video.frame_index)

    return clip_vip_vectors, video_ids, frame_indexs, marlin_video_vectors


class VideoService:
    def __init__(self):
        self.es = current_app.es
        self.session = current_app.db.session
        # self.clip_vip_vectors, self.clip_description_vector_1, self.clip_description_vector_3, self.clip_description_vector_5, self.marlin_video_vectors, self.datasets, self.video_names = fetch_video_data(self.session)
        self.clip_vip_vectors, self.video_ids, self.frame_indexs, self.marlin_video_vectors = fetch_video_data(self.session)

    def query_by_text(self, query: str, top_n=20):
        """Retrieve video names and paths based on the input query.

        Args:
            query (str): The input query text.
            top_n (int): The number of top results to return.

        Returns:
            str: JSON response with video names and paths.
        """
        # Step 1: Encode the input query
        query_embedding = self.es.encode_text(query)

        # Step 2: Compute cosine similarity with precomputed video embeddings
        similarities = cosine_similarity([query_embedding], self.clip_vip_vectors)[0]

        # Step 3: Get the top N indices based on similarity
        top_n_indices = np.argsort(-similarities)[:top_n]
        similarities_map = {self.video_ids[int(i)]: similarities[i] for i in top_n_indices}
        top_n_indices = list(similarities_map.keys())

        # Step 4: Fetch videos from the database
        videos = self.session.query(Video).filter(Video.id.in_(top_n_indices)).all()

        # Step 5: Prepare response data
        video_data = [
            {
                "video_name": video.name,
                "video_path": self.get_video_path(video.name),
                "similarity": float(similarities_map[video.id])
            }
            for video in videos
        ]

        # Construct and return the response
        return json.dumps({"data": video_data}), requests.codes.ok
    
    def query_by_image(self, image, top_n=5):
        """Retrieve video names and paths based on the input image.

        Args:
            image (str): The path to the input image.
            top_n (int): The number of top results to return.

        Returns:
            str: JSON response with video names and paths.
        """
        res = []

        # Step 1: Encode the input image
        s = time()
        image_embedding = self.es.encode_image(image)
        print("Inference Image cost:", time() - s,'s')
        # Step 2: Compute cosine similarity with precomputed video embeddings
        similarities = cosine_similarity([image_embedding], self.marlin_video_vectors)[0]

        # Step 3: Get the top N indices based on similarity
        top_n_indices = np.argsort(-similarities)[:top_n]
        similarities_map = {self.video_ids[int(i)]: similarities[i] for i in top_n_indices}
        top_n_indices = list(similarities_map.keys())

        # Step 4: Fetch videos from the database
        videos = self.session.query(Video).filter(Video.id.in_(top_n_indices)).all()

        # # Step 5: Load video data, extract frames, and calculate embeddings
        # video_embeddings = []
        # for video in videos:
        #     video_path = video.path
        #     video_embedding = self.get_video_embedding(video_path)
        #     video_embeddings.append(video_embedding)

        # # Step 6: Calculate pairwise cosine similarities for video embeddings
        # pairwise_similarities = cosine_similarity(video_embeddings)

        # # Log pairwise similarities
        # print("Pairwise cosine similarities for top N videos:")
        # for i, video_i in enumerate(videos):
        #     for j, video_j in enumerate(videos):
        #         print(f"Similarity between video '{video_i.name}' and video '{video_j.name}': {pairwise_similarities[i, j]}")
        # s = time()
        # Step 7: Prepare response data
        video_data = [
            {
                "video_name": video.name,
                "video_path": self.get_video_path(video.name),
                "similarity": float(similarities_map[video.id])
            }
            for video in videos
        ]
        print(f"Inference {len(videos)} video cost:", (time() - s)/len(videos),'s')
        # Construct and return the response
        return json.dumps({"data": video_data}), requests.codes.ok

    
    def query_by_image_text(self, query: str, image, top_n=20, coefficients=[0.5, 0.5]):
        """Retrieve video names and paths based on the query and image.

        Args:
            query (str): The input query text.
            image (str): The path to the input image.
            top_n (int): The number of top results to return.
            coefficients (list): Coefficients for combining image and text similarities.

        Returns:
            str: JSON response with video names and paths.
        """
        start_time = time.time()  # Start timing

        query_embedding = self.es.encode_text(query)
        image_embedding = self.es.encode_image(image)
        sim_video_from_queries = cosine_similarity([query_embedding], self.clip_vip_vectors)[0]
        sim_video_from_images = cosine_similarity([image_embedding], self.marlin_video_vectors)[0]

        total_similarity = (
            coefficients[0] * sim_video_from_images +
            coefficients[1] * sim_video_from_queries
        )
        top_n_indices = np.argsort(-total_similarity)[:top_n]
        similarities_map = {self.video_ids[int(i)]: total_similarity[i] for i in top_n_indices}
        top_n_indices = list(similarities_map.keys())

        videos = self.session.query(Video).filter(Video.id.in_(top_n_indices)).all()
        video_data = [
            {
                "video_name": video.name,
                "video_path": self.get_video_path(video.name),
                "similarity": float(similarities_map[video.id])
            }
            for video in videos
        ]

        # End timing
        inference_time = time.time() - start_time
        print(f"Inference time for combined query (image + text): {inference_time:.4f} seconds")

        return json.dumps({"data": video_data, "inference_time": inference_time}), requests.codes.ok


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
    
    def get_video_embedding(self, video_path):
        """Extract video embeddings using PyAV and DeepFace.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Averaged embedding for the video.
        """
        from deepface import DeepFace 
        import av
        try:
            # Open video file with PyAV
            container = av.open(video_path)

            frame_embeddings = []
            frame_count = 0
            for frame in container.decode(video=0):
                # Extract frames at intervals (e.g., every 10th frame)
                if frame_count % 10 == 0:
                    frame_image = frame.to_image()  # Convert AVFrame to PIL Image
                    
                    # Use DeepFace to compute embedding for the frame
                    try:
                        embedding = DeepFace.represent(np.array(frame_image), model_name="Facenet", enforce_detection=False)[0]["embedding"]
                        frame_embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error processing frame at {frame_count} in {video_path}: {e}")
                
                frame_count += 1

            # Average embeddings across all frames
            if frame_embeddings:
                video_embedding = np.mean(frame_embeddings, axis=0)
            else:
                raise ValueError(f"No frames could be processed from video {video_path}.")

            return video_embedding

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            raise e