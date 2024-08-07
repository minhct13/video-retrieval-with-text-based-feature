import os
import json
import argparse

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

def convert_txt_to_json(text_dir):
    txt_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(text_dir, txt_file)
        video_name = os.path.splitext(txt_file)[0]
        texts, probs = parse_text_file(txt_path)
        answers = [{"answer": text, "prob": prob} for text, prob in zip(texts, probs)]
        json_content = {video_name: answers}
        
        json_file = os.path.join(text_dir, f"{video_name}.json")
        with open(json_file, 'w') as jf:
            json.dump(json_content, jf, indent=4)
        
        print(f"Converted {txt_file} to {video_name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text files to JSON format.")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing text files.")
    args = parser.parse_args()
    
    convert_txt_to_json(args.text_dir)


# python scripts/convert_txt_to_json.py --text_dir ../data/kaggle/working/output