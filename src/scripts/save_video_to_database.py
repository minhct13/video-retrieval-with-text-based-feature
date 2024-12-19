import os
import av
import numpy as np
from PIL import Image
from deepface import DeepFace
import argparse
import json
from sqlalchemy.orm import sessionmaker
from video import Video, save_video_keyframe_embeddings, setup_database
from video import EncoderService


def process_video_keyframes_with_descriptions(es, session, video_path, video_text_data, confidence_threshold=0.9): 
    video_name = os.path.basename(video_path)

    # Open the video file
    container = av.open(video_path)
    stream = container.streams.video[0]

    clip_vip_vector = es.encode_video(video_path)
    keyframe_data = {}
    for i, frame in enumerate(container.decode(stream)):
        if frame.key_frame:  # Process only keyframes
            try:
                # Use ArcFace for face detection and embedding extraction
                frame_array = np.array(frame.to_image())
                face_info = DeepFace.represent(frame_array, model_name="ArcFace", enforce_detection=False)
                if face_info:
                    face_confidence = face_info[0].get("face_confidence", 0)
                    if face_confidence >= confidence_threshold:
                        # Extract ArcFace embedding
                        arcface_embedding = face_info[0]['embedding']
                        # Save keyframe embeddings
                        keyframe_data[i] = {
                            "marlin_video_vector": arcface_embedding,
                            "clip_vip_vector": clip_vip_vector,
                        }

            except Exception as e:
                print(f"Failed to process keyframe {i} in video {video_path}: {e}")
                continue

    # Extract text data for the video
    texts = video_text_data.get(video_name, [])

    if texts:
        # Function to truncate text to 65 characters
        def truncate_text(answer, max_length=65):
            return answer[:max_length]

        # Extract concatenated answers for descriptions
        clip_description_text_1 = truncate_text(texts[0]["answer"]) if len(texts) > 0 else ""
        clip_description_text_3 = truncate_text(" ".join([texts[i]["answer"] for i in range(min(3, len(texts)))]))
        clip_description_text_5 = truncate_text(" ".join([texts[i]["answer"] for i in range(min(5, len(texts)))]))

        # Extract vectors for the concatenated answers
        clip_description_vector_1 = es.encode_text(clip_description_text_1)
        clip_description_vector_3 = es.encode_text(clip_description_text_3)
        clip_description_vector_5 = es.encode_text(clip_description_text_5)

        # Save keyframe embeddings and description vectors to the database
        if keyframe_data:
            save_video_keyframe_embeddings(
                session=session,
                video_name=video_name,
                video_path=video_path,
                dataset="face",
                keyframe_data=keyframe_data,
            )

            # Save description vectors as metadata for the video
            save_video_descriptions(
                session=session,
                video_name=video_name,
                clip_description_vector_1=clip_description_vector_1,
                clip_description_vector_3=clip_description_vector_3,
                clip_description_vector_5=clip_description_vector_5,
            )

            print(f"Processed video {video_name} successfully with {len(keyframe_data)} keyframes and descriptions.")
        else:
            print(f"No valid keyframes found for video {video_name}.")
    else:
        print(f"No text data found for video {video_name}. Skipping descriptions.")


def save_video_descriptions(session, video_name, clip_description_vector_1, clip_description_vector_3, clip_description_vector_5):
    video = session.query(Video).filter(Video.name == video_name).first()
    if video:
        # Add or update description vectors in the Video table
        video.clip_description_vector_1 = clip_description_vector_1
        video.clip_description_vector_3 = clip_description_vector_3
        video.clip_description_vector_5 = clip_description_vector_5
        session.commit()
    else:
        print(f"Video {video_name} not found in the database for saving descriptions.")

def load_video_text_data(json_dir):
    data = {}
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data.update(json.load(f))
    return data

def main():
    parser = argparse.ArgumentParser(description="Process videos and save keyframe embeddings to the database.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing JSON text data for videos.")

    args = parser.parse_args()

    # Initialize database session
    Session = sessionmaker(bind=setup_database())
    session = Session()

    # Initialize the encoding service (es)
    es = EncoderService()
    es.init_app("pretrain_clipvip_base_16.pt")

    # Load video text data
    video_text_data = load_video_text_data(args.json_dir)

    # Process each video in the directory
    for video_file in os.listdir(args.video_dir):
        video_path = os.path.join(args.video_dir, video_file)
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mkv')):
            process_video_keyframes_with_descriptions(
                es=es,
                session=session,
                video_path=video_path,
                video_text_data=video_text_data
            )

if __name__ == "__main__":
    main()
