import os
import json
import argparse
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor
from video import EncoderService, Video
# Initialize the database connection
DATABASE_URI = 'postgresql://user:user@localhost:5432/postgres'
engine = sqlalchemy.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()


def save_video_embedding(session, video_name, video_path, video_vector, text_vectors, text_probs):
    # Check if the video already exists
    # existing_video = session.query(Video).filter_by(name=video_name).first()
    # if existing_video:
        # Update existing record
    #     existing_video.path = video_path
    #     existing_video.video_vector = video_vector
    #     existing_video.text_vector_1 = text_vectors[0]
    #     existing_video.text_vector_2 = text_vectors[1]
    #     existing_video.text_vector_3 = text_vectors[2]
    #     existing_video.text_vector_4 = text_vectors[3]
    #     existing_video.text_vector_5 = text_vectors[4]
    #     existing_video.text_prob_1 = text_probs[0]
    #     existing_video.text_prob_2 = text_probs[1]
    #     existing_video.text_prob_3 = text_probs[2]
    #     existing_video.text_prob_4 = text_probs[3]
    #     existing_video.text_prob_5 = text_probs[4]
    # else:
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
    session.commit()
    session.close()


def process_video(es, session, video_path, text_vectors, text_probs):
    video_name = os.path.basename(video_path)
    video_vector = es.encode_video(video_path)
    save_video_embedding(
        session=session,
        video_name=video_name,
        video_path=video_path,
        video_vector=video_vector,
        text_vectors=text_vectors,
        text_probs=text_probs
    )
    print(f"SAVED {video_name}")


def load_json_files(json_dir):
    data = {}
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data.update(json.load(f))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing JSON files with video-text data.")

    args = parser.parse_args()

    es = EncoderService()
    es.init_app(args.checkpoint)

    # Load the video-text data from the JSON directory
    video_text_data = load_json_files(args.json_dir)    
    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]


    # with ThreadPoolExecutor() as executor:
        # futures = []
i = 0
for video_path in video_files:
    video_name = os.path.basename(video_path)
    existing_video = session.query(Video).filter_by(name=video_name).first()
    if existing_video:
        i+= 1
        print(f"SKIP {video_name}", i)
        continue 
    
    video_name = video_name.replace(".mp4", "")
    if video_name not in video_text_data:
        continue
    

    texts = video_text_data[video_name]
    text_vectors = [es.encode_text(text["answer"]) for text in texts]
    text_probs = [text["prob"] for text in texts]

    process_video(
        es=es,
        session=session,
        video_path=video_path,
        text_vectors=text_vectors,
        text_probs=text_probs
    )




# python scripts/encode_embedding.py  --video_dir ../data/ --checkpoint ./pretrain_clipvip_base_16.pt --json_dir ../data/kaggle/working/output
