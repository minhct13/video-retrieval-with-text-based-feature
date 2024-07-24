import os
import argparse
from app import EncoderService

if __name__ == "__main__":
    """Inference video embeddings and stored to database
    """
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Text to compute embeddings for.")
    
    args = parser.parse_args()
    es = EncoderService(args.checkpoint)
    for video_name in os.listdir(args.video_dir):
        video_path = os.path.join(args.video_dir, video_name)
        embedding_vector = es.encode(args.video_dir, args.text)