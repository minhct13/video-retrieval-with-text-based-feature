import os
import numpy as np
import av
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

# Function to load and initialize the CLIP model
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_keyframe_vectors(video_path, video_name, cache_dir, model, processor):
    # Define the cache path and check if vectors have been cached already
    cache_path = os.path.join(cache_dir, f"{video_name}_keyframes.npz")
    if os.path.exists(cache_path):
        print(f"Loading keyframe vectors from cache for {video_name}...")
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

import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache keyframe vectors from video files.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--keyframe_dir", type=str, required=True, help="Directory to save extracted keyframe vectors.")
    
    args = parser.parse_args()
    model, processor = load_clip_model()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure the model is on the right device

    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}.")
        exit()

    for i, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframe_vectors = get_keyframe_vectors(video_path, video_name, args.keyframe_dir, model, processor)
        print(f"{i+1}/{len(video_files)}: Processed {video_name}, keyframe vectors cached in {args.keyframe_dir}.")

        
# python scripts/parse_keyframe.py --video_dir ../data --keyframe_dir ../data/keyframes --checkpoint ./pretrain_clipvip_base_16.pt
