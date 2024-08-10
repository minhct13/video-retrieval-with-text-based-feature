from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from pgvector.sqlalchemy import Vector
import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast, AutoProcessor
from PIL import Image
# from clipvip.CLIP_VIP import CLIPModel, clip_loss
# from transformers import AutoProcessor, CLIPModel
import torch
import json
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from clipvip.CLIP_VIP import CLIPModel as CLIP_VIP_Model, clip_loss
from transformers import AutoProcessor, CLIPModel as Huggingface_CLIPModel


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:1234@localhost/videodb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Video(db.Model):
    __tablename__ = 'videos'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String, unique=True, index=True, nullable=False)
    path = db.Column(db.String, unique=True, nullable=False)
    video_vector = db.Column(Vector(512), nullable=False)
    text_vector_1 = db.Column(Vector(512), nullable=False)
    text_vector_2 = db.Column(Vector(512), nullable=False)
    text_vector_3 = db.Column(Vector(512), nullable=False)
    text_vector_4 = db.Column(Vector(512), nullable=False)
    text_vector_5 = db.Column(Vector(512), nullable=False)
    text_prob_1 = db.Column(db.Float, nullable=False)
    text_prob_2 = db.Column(db.Float, nullable=False)
    text_prob_3 = db.Column(db.Float, nullable=False)
    text_prob_4 = db.Column(db.Float, nullable=False)
    text_prob_5 = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<Video {self.name}>"


class EncoderService:
    def init_app(self, checkpoint=""):
        self.extraCfg = edict({
            "type": "ViP",
            "temporal_size": 12,
            "if_use_temporal_embed": 1,
            "logit_scale_init_value": 4.60,
            "add_cls_num": 3
        })
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the checkpoint from the specified path
        self.checkpoint = torch.load(checkpoint, map_location=self.device)
        self.cleanDict = {key.replace("clipmodel.", ""): value for key, value in self.checkpoint.items()}
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
        # Load and configure the pre-trained CLIP model
        self.clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16")
        self.clipconfig.vision_additional_config = self.extraCfg
        self.model = CLIP_VIP_Model(config=self.clipconfig).to(self.device)
        self.model.load_state_dict(self.cleanDict)


    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])


    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len) if converted_len < seg_len else converted_len - 1
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
        
    def encode_text(self, text):
        # Tokenize the text input
        tokens = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            textOutput = self.model.get_text_features(**tokens)
        return textOutput.cpu().numpy().flatten()

    def encode_keyframes(self, video_path, cache_dir):
        # video_name = os.path.splitext(os.path.basename(video_path))[0]
        # cache_path = os.path.join(cache_dir, f"{video_name}_keyframes.npz")
        # if os.path.exists(cache_path):
        #     return np.load(cache_path)["keyframe_vectors"]

        container = av.open(video_path)
        stream = container.streams.video[0]
        keyframe_vectors = []

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for frame in container.decode(stream):
            if frame.key_frame:
                img = frame.to_image()  # Convert the frame to PIL image
                inputs = processor(images=img, return_tensors="pt")  # Process the image
                # Ensure that inputs are on the correct device and add an extra dimension for batch size
                inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    keyframe_vector = image_features.squeeze().cpu().numpy()  # Extract embeddings
                    keyframe_vectors.append(keyframe_vector)

        # Convert list of vectors to numpy array
        keyframe_vectors = np.array(keyframe_vectors)
        # Save the vectors to the cache
        # np.savez_compressed(cache_path, keyframe_vectors=keyframe_vectors)

        return keyframe_vectors

    def encode_video(self, video_path):
        # Process video file from the video_path
        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
        if video_path.endswith(".mp4"):
            container = av.open(video_path)
            clip_len = 12
            fcount = container.streams.video[0].frames
            indices = self.sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
            video = self.read_video_pyav(container, indices)
            pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values.to(self.device)
            inputs = {
                "if_norm": True,
                "pixel_values": pixel_values
            }

            with torch.no_grad():
                frame_features = self.model.get_image_features(**inputs)
            video_embedding = frame_features.mean(dim=0, keepdim=True).cpu().numpy().flatten()
        return video_embedding


def get_keyframe_vectors(video_path, video_name, cache_dir, model, processor):
    # Define the cache path and check if vectors have been cached already
    cache_path = os.path.join(cache_dir, f"{video_name}_keyframes.npz")
    if os.path.exists(cache_path):
        # print(f"Loading keyframe vectors from cache for {video_name}...")
        return np.load(cache_path)["keyframe_vectors"]

    # Open the video file
    container = av.open(video_path)
    stream = container.streams.video[0]
    keyframe_vectors = []

    for frame in container.decode(stream):
        if frame.key_frame:
            img = frame.to_image()  # Convert the frame to PIL image
            inputs = processor(images=img, return_tensors="pt")  # Process the image
            # Ensure that inputs are on the correct device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                keyframe_vector = image_features.squeeze().cpu().numpy()  # Extract embeddings
                keyframe_vectors.append(keyframe_vector)

    # Convert list of vectors to numpy array
    keyframe_vectors = np.array(keyframe_vectors)
    # Save the vectors to the cache
    np.savez_compressed(cache_path, keyframe_vectors=keyframe_vectors)

    return keyframe_vectors


# Function to load and initialize the CLIP model
# def load_clip_model():
#     model = Huggingface_CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = 
#     return model, processor

# Function to load pre-extracted keyframe vectors from cache
def load_keyframe_vectors(cache_dir):
    keyframe_vectors = {}
    for file in os.listdir(cache_dir):
        if file.endswith('_keyframes.npz'):
            video_name = file.replace('_keyframes.npz', '')
            data = np.load(os.path.join(cache_dir, file))
            keyframe_vectors[video_name] = data['keyframe_vectors']
    return keyframe_vectors

# Function to extract feature vector for a query image
def extract_query_vector(image_path, model, processor):
    img = Image.open(image_path)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.squeeze().cpu().numpy()

# Function to load ground truth from JSON file
def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def fetch_video_data(session, limit):
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
        if len(video_names) == limit:
            break
    
    return np.array(video_vectors), np.array(text_vectors), np.array(text_probs), video_names

def get_top_k_retrieved(text_embedding, video_embeddings, k=10):
    similarities = cosine_similarity([text_embedding], video_embeddings)[0]
    top_k_indices = np.argsort(-similarities)[:k]
    return top_k_indices

# Function to get top-k retrieved items for 'Keyframe CLIP' strategy
def get_top_k_retrieved_keyframes(query_vector, keyframe_vectors, k=10):
    all_similarities = []
    video_name_mapping = []

    for video_name, vectors in keyframe_vectors.items():
        similarities = cosine_similarity([query_vector], vectors)[0]
        all_similarities.extend(similarities)
        video_name_mapping.extend([video_name] * len(vectors))

    all_similarities = np.array(all_similarities)
    top_k_indices = np.argsort(-all_similarities)[:k]

    top_k_videos = [video_name_mapping[idx] for idx in top_k_indices]

    # Debug information
    print(f"Top-{k} similarities: {all_similarities[top_k_indices]}")
    print(f"Top-{k} videos: {top_k_videos}")

    return top_k_videos

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

import os
def cache_text_embeddings(text_embeddings, cache_file):
    np.savez_compressed(cache_file, text_embeddings=text_embeddings)

def load_text_embeddings(cache_file):
    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as data:
            return {key: data[key] for key in data}
    return None

# Function to extract text embedding using CLIP model
def extract_text_embedding(text, model, processor):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.squeeze().cpu().numpy()

def cache_keyframe_text_embeddings(text_embeddings, cache_file):
    np.savez_compressed(cache_file, text_embeddings=text_embeddings)

def load_keyframe_text_embeddings(cache_file):
    if os.path.exists(cache_file):
        with np.load(cache_file) as data:
            return data['text_embeddings']
    return None