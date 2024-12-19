import numpy as np
from deepface import DeepFace
import os
import json
import av
from PIL import Image


def find_and_crop_face_from_keyframes(video_path, save_dir, confidence_threshold=0.9):
    """
    Find the first face in the keyframes of a video with confidence > 80% and crop it, saving the cropped image.

    Args:
        video_path (str): Path to the video file.
        save_dir (str): Directory to save the cropped face image.
        confidence_threshold (float): Minimum confidence required to consider a detected face.

    Returns:
        str or None: Path to the saved cropped face image, or None if no valid face is found.
    """
    # Open the video file
    container = av.open(video_path)
    stream = container.streams.video[0]

    for i, frame in enumerate(container.decode(stream)):
        if frame.key_frame:  # Process only keyframes
            frame_array = np.array(frame.to_image())  # Convert PIL Image to numpy array

            try:
                # Use DeepFace to detect the face and get facial area
                face_info = DeepFace.represent(frame_array, enforce_detection=False)
                if face_info:
                    face_confidence = face_info[0].get('face_confidence', 0)
                    if face_confidence >= confidence_threshold:
                        facial_area = face_info[0]['facial_area']

                        # Crop the face using the detected facial area
                        face_crop = frame.to_image().crop((facial_area['x'], facial_area['y'], 
                                                           facial_area['x'] + facial_area['w'], 
                                                           facial_area['y'] + facial_area['h']))

                        # Save the cropped face
                        cropped_face_name = f"{os.path.basename(video_path).split('.')[0]}_face.jpg"
                        cropped_face_path = os.path.join(save_dir, cropped_face_name)
                        face_crop.save(cropped_face_path)
                        return cropped_face_path

            except Exception as e:
                print(f"Failed to process keyframe {i} in video {video_path}: {e}")
                continue

    return None

def create_groundtruth(input_json_path, output_json_path, video_dir, save_dir, max_videos=1000):
    """
    Create groundtruth JSON file for up to 1,000 videos with detected faces.

    Args:
        input_json_path (str): Path to the input JSON file with descriptions.
        output_json_path (str): Path to save the output groundtruth JSON file.
        video_dir (str): Directory containing the video files.
        save_dir (str): Directory to save the cropped face images.
        max_videos (int): Maximum number of videos to process.
    """
    # Load the input JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    groundtruth = {}
    video_count = 0
    import tqdm
    
    for video_id, description_list in tqdm.tqdm(data.items()):
        if video_count >= max_videos:
            break
        
        video_path = os.path.join(video_dir, video_id)

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        # Try to find and crop a face from the keyframes of the video
        cropped_face_path = find_and_crop_face_from_keyframes(video_path, save_dir)

        if cropped_face_path:
            # If a valid face is found, pair it with the descriptions
            paired_data = [{"frame": cropped_face_path, "description": description}
                           for description in description_list]

            # Update the groundtruth dictionary
            groundtruth[video_id] = paired_data
            video_count += 1

    # Save the groundtruth to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(groundtruth, f, indent=4)

    print(f"Groundtruth saved to {output_json_path}")

if __name__ == "__main__":
    input_json_path = "../data/video_retrieval/msrvtt/train_7k.json"  # Path to the input JSON file with descriptions
    output_json_path = "../data/groundtruth_face_crops.json"  # Path to save the output groundtruth JSON file
    video_dir = "../data/"  # Directory containing the video files
    save_dir = "../data/face_images_2"  # Directory to save the cropped face images
    
    create_groundtruth(input_json_path, output_json_path, video_dir, save_dir, max_videos=1000)
