import openai
import json
import time
import os

# Set your OpenAI API key
openai.api_key = ''

def generate_description(video_attributes):
    # Prepare the prompt for GPT-4
    prompt = (
        f"Generate five sentences to describe a person in a video with the following attributes:\n\n"
        f"Attributes: {video_attributes}\n\n"
        "The description should be suitable for query-retrieval purposes."
    )
    
    # Call GPT-4 to generate the descriptions
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates detailed descriptions based on given attributes."},
            {"role": "user", "content": prompt},
        ]
    )
    
    # Extract the generated sentences
    sentences = response['choices'][0]['message']['content'].split('\n')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def load_processed_videos(log_file_path):
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            return json.load(log_file)
    else:
        return {}

def append_to_log(log_file_path, video_id, descriptions):
    with open(log_file_path, 'r+') as log_file:
        data = json.load(log_file)
        data[video_id] = descriptions
        log_file.seek(0)
        json.dump(data, log_file, indent=4)
        log_file.truncate()

def process_json(input_json, log_file_path):
    output_json = {}
    request_count = 0
    max_requests_per_minute = 3
    sleep_time = 60 / max_requests_per_minute  # Time to wait between requests

    processed_videos = load_processed_videos(log_file_path)

    for video_id, attributes in input_json.items():
        if video_id in processed_videos:
            print(f"Skipping already processed video: {video_id}")
            continue
        
        if isinstance(attributes[-1], list):  # Handling emotion details separately
            emotion_info = attributes.pop()
            attributes.extend([f"emotion: {e['emotion']} from {e['start_sec']}s to {e['end_sec']}s" for e in emotion_info])
        
        # Generate descriptions using GPT-4
        descriptions = generate_description(attributes)
        output_json[video_id] = descriptions

        # Append to the log file
        append_to_log(log_file_path, video_id, descriptions)

        request_count += 1

        if request_count % max_requests_per_minute == 0:
            print(f"Rate limit reached: waiting for {sleep_time * max_requests_per_minute} seconds.")
            time.sleep(sleep_time * max_requests_per_minute)

    return output_json

def save_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    input_json_path = '../data/face_test.json'  # Path to your input JSON file
    output_json_path = '../data/face_test_descriptions.json'  # Path for the output JSON file
    log_file_path = '../data/processed_videos_log.json'  # Path for the log file to store completed video IDs

    # Ensure the log file exists
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            json.dump({}, log_file)

    # Load the input JSON file
    with open(input_json_path, 'r') as f:
        input_json = json.load(f)

    # Process the input JSON to generate descriptions
    output_json = process_json(input_json, log_file_path)

    # Save the output JSON to a file
    save_json(output_json, output_json_path)

    print(f"Descriptions saved to {output_json_path}")