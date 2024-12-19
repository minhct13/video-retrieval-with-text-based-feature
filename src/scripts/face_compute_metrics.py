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
from marlin.src.marlin_pytorch import Marlin
from compute_funcs import *


DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()
model = Marlin.from_file("marlin_vit_base_ytf", "marlin_vit_base_ytf.encoder.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--cache_file", default="", type=str, help="Path to the cache file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--keyframe_dir", type=str, required=True, help="Directory to save extracted keyframe vectors.")
    args = parser.parse_args()
    es = EncoderService()  # EncoderService for CLIP-VIP
    
    es.init_app(args.checkpoint)
    # Load cached text embeddings
    cache = load_text_embeddings(args.cache_file) or {}

    # Fetch video data from the database
    # video_vectors = cache.get('video_vectors')
    # text_vectors = cache.get('text_vectors')
    # text_probs = cache.get('text_probs')
    # video_names = cache.get('video_names')
    # if video_names is not None:
    #     video_names = video_names.tolist()
    
    # if video_vectors is None or text_vectors is None or text_probs is None or video_names is None:
    video_vectors, text_vectors, text_probs, video_names = fetch_video_data(session, limit=50)
    cache['video_vectors'] = video_vectors
    cache['text_vectors'] = text_vectors
    cache['text_probs'] = text_probs
    cache['video_names'] = video_names

    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth_file)
    print(f"Loaded ground truth for {len(ground_truth)} videos.")

    # Filter ground truth to only include videos that are in video_names
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k + ".mp4" in video_names}
    print(f"Filtered ground truth contains {len(filtered_ground_truth)} videos.")
    
    if not filtered_ground_truth:
        print("No ground truth entries match the provided video names.")
        exit()

    # print("Ground truth indices:", len(ground_truth_indices))

    print("Loading image_query_embeddings...")
    image_query_embeddings = cache.get('image_query_embeddings')
    if image_query_embeddings is None:
        image_query_embeddings = []
        for video_name, value in filtered_ground_truth.items():
            frames = value.get("frames")
            for frame in frames:
                image_embedding = model.extract_image(frame, crop_face=True)
                image_query_embeddings.append(image_embedding.cpu())
        image_query_embeddings = np.array(image_query_embeddings)
        cache['image_query_embeddings'] = image_query_embeddings

    # Load or compute keyframe vectors
    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}.")
        exit()

    ground_truth_indices = [video_names.index(k + ".mp4") for k in filtered_ground_truth.keys() for _ in filtered_ground_truth[k]['frames']]
    # Save all embeddings to a single cache file
    cache_embeddings(args.cache_file, cache)

    print("Loading text_query_embeddings...")
    text_query_embeddings = cache.get('text_query_embeddings')
    if text_query_embeddings is None:
        text_query_embeddings = []
        for video_name, value in filtered_ground_truth.items():
            descriptions = value.get("description")
            for description in descriptions:
                text_query_embedding = es.encode_text(description)
                text_query_embeddings.append(text_query_embedding)
        text_query_embeddings = np.array(text_query_embeddings)
        cache['text_query_embeddings'] = text_query_embeddings

    # Define strategies
    strategies = ['keyframe', 'marlin', 'text', 'marlin_text']

    # Coefficients for JSP
    for strategy in strategies:
        print(f"Evaluating strategy: {strategy}")
        if strategy == 'keyframe':
            keyframe_vectors = {}   
            print(f"Loading keyframes...")
            for i, video_path in tqdm(enumerate(video_files)):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                if video_name in filtered_ground_truth.keys():
                    keyframe_vector = get_keyframe_vectors(video_path, video_name, args.keyframe_dir, model)
                    keyframe_vectors[video_name] = keyframe_vector
            
            recall_metrics = compute_keyframe(image_query_embeddings, keyframe_vectors, ground_truth_indices, video_names)
        elif strategy == 'marlin':
            recall_metrics = compute_marlin(image_query_embeddings, video_vectors, ground_truth_indices, video_names)
        elif strategy == 'text':
            coefficients = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            recall_metrics = compute_text_video(image_query_embeddings, text_query_embeddings, video_vectors, text_vectors, ground_truth_indices, coefficients)
        elif strategy == 'marlin_text':
            coefficients = [0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
            recall_metrics = compute_text_video(image_query_embeddings, text_query_embeddings, video_vectors, text_vectors, ground_truth_indices, coefficients)
        
        # Print recall metrics
        print(f"Recall Metrics for {strategy}: {recall_metrics}")


 
# python scripts/jsp_jvp.py --video_dir ../data  --ground_truth_file ../data/video_retrieval/msrvtt/train_7k.json --checkpoint ./pretrain_clipvip_base_16.pt --cache_file cache.npz
# python scripts/face_compute_metrics.py --video_dir ../data/faces  --ground_truth_file ../data/processed_videos_log.json --checkpoint ./pretrain_clipvip_base_16.pt --cache_file cache_face0.npz --keyframe_dir ../data/keyframes/