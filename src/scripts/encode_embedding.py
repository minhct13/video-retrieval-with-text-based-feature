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
    # Check if the video already exists
    existing_video = session.query(Video).filter_by(name=video_name).first()
    if existing_video:
        # Update existing record
        existing_video.path = video_path
        existing_video.video_vector = video_vector
        existing_video.text_vector_1 = text_vectors[0]
        existing_video.text_vector_2 = text_vectors[1]
        existing_video.text_vector_3 = text_vectors[2]
        existing_video.text_vector_4 = text_vectors[3]
        existing_video.text_vector_5 = text_vectors[4]
        existing_video.text_prob_1 = text_probs[0]
        existing_video.text_prob_2 = text_probs[1]
        existing_video.text_prob_3 = text_probs[2]
        existing_video.text_prob_4 = text_probs[3]
        existing_video.text_prob_5 = text_probs[4]
    else:
        # Create new record
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

def parse_text_file(text_file_path):
    with open(text_file_path, "r") as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    texts = []
    probs = []

    buffer = []
    for line in lines:
        if '|' in line:
            if buffer:
                texts.append(' '.join(buffer))
                buffer = []
            text, prob = line.rsplit('|', 1)
            texts.append(text.strip())
            probs.append(float(prob.strip()))
        else:
            buffer.append(line.strip())

    if buffer:
        texts.append(' '.join(buffer))
    
    return texts, probs

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
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing text files.")

    args = parser.parse_args()

    es = EncoderService()
    es.init_app(args.checkpoint)

    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]

    with ThreadPoolExecutor() as executor:
        futures = []
        for video_path in video_files:
            text_path = os.path.join(args.text_dir, os.path.basename(video_path).replace(".mp4", ".txt"))
            if not os.path.exists(text_path):
                continue

            texts, probs = parse_text_file(text_path)
            text_vectors = [es.encode_text(text) for text in texts]
            text_probs = probs

            futures.append(
                executor.submit(
                    process_video,
                    es,
                    session,
                    video_path,
                    text_vectors,
                    text_probs
                )
            )


    # Ensure all futures are completed
    for future in futures:
        future.result()

    
    session.commit()
    session.close()

# python scripts/encode_embedding.py  --video_dir ../data/archive/TrainValVideo/ --checkpoint ../pretrain_clipvip_base_16.pt --text_dir "../data/kaggle
# /working/output"
