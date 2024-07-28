import os
import argparse
from app import EncoderService
from app.models import Video
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor

# Initialize the database connection
DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

def save_video_embedding(session, video_name, video_path, video_vector, text_vectors, text_probs):
    video = Video(
        name=video_name,
        path=video_path,
        video_vector=video_vector,
        text_vector_1=text_vectors[0],
        text_vector_2=text_vectors[1],
        text_vector_3=text_vectors[2],
        text_vector_4=text_vectors[3],
        text_vector_5=text_vectors[4],
        text_prob_1=text_probs[0],
        text_prob_2=text_probs[1],
        text_prob_3=text_probs[2],
        text_prob_4=text_probs[3],
        text_prob_5=text_probs[4]
    )
    session.add(video)

def process_video(es, session, video_path, text_vectors, text_probs):
    video_vector = es.encode_video(video_path)
    video_name = os.path.basename(video_path)
    save_video_embedding(
        session=session,
        video_name=video_name,
        video_path=video_path,
        video_vector=video_vector,
        text_vectors=text_vectors,
        text_probs=text_probs
    )
    print(f"SAVED {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Text to compute embeddings for.")

    args = parser.parse_args()

    es = EncoderService(args.checkpoint)

    text_vectors = [es.encode_text(args.text) for _ in range(5)]
    text_probs = [0.5 for _ in range(5)]  # Replace with actual probabilities if available

    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
    video_files = video_files[:2]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_video,
                es,
                session,
                video_path,
                text_vectors,
                text_probs
            ) for video_path in video_files
        ]

    # Ensure all futures are completed
    for future in futures:
        future.result()

    session.commit()
    session.close()

# python scripts/encode_embedding.py  --video_dir ../data/archive/TrainValVideo/ --checkpoint ../pretrain_clipvip_base_16.pt --text "The TV screens tennis"