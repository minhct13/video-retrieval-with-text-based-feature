import torch
import webvtt
import os
import cv2 
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4.conversation.conversation import CONV_VISION
from torchvision import transforms
import json
from tqdm import tqdm
import soundfile as sf
import argparse
import moviepy.editor as mp
import gradio as gr
from pytubefix import YouTube
import shutil
from PIL import Image
from moviepy.editor import VideoFileClip
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def prepare_input(vis_processor,video_path,subtitle_path,instruction):  
    cap = cv2.VideoCapture(video_path)
    if subtitle_path is not None: 
        # Load the VTT subtitle file
        vtt_file = webvtt.read(subtitle_path) 
        print("subtitle loaded successfully")  
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        # print("Video duration = ",clip.duration)
        clip.close()
    else :
        # calculate the total number of frames in the video using opencv        
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    max_images_length = 45
    max_sub_len = 400
    images = []
    frame_count = 0
    sampling_interval = int(total_num_frames / max_images_length)
    if sampling_interval == 0:
        sampling_interval = 1
    img_placeholder = ""
    subtitle_text_in_interval = ""
    history_subtitles = {}
    raw_frames=[]
    number_of_words=0
    transform=transforms.Compose([
                transforms.ToPILImage(),
            ])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Find the corresponding subtitle for the frame and combine the interval subtitles into one subtitle
        # we choose 1 frame for every 2 seconds,so we need to combine the subtitles in the interval of 2 seconds
        if subtitle_path is not None: 
            for subtitle in vtt_file:
                sub=subtitle.text.replace('\n',' ')
                if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                    if not history_subtitles.get(sub,False):
                        subtitle_text_in_interval+=sub+" "
                    history_subtitles[sub]=True
                    break
        if frame_count % sampling_interval == 0:
            raw_frames.append(Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)))
            frame = transform(frame[:,:,::-1]) # convert to RGB
            frame = vis_processor(frame)
            images.append(frame)
            img_placeholder += '<Img><ImageHere>'
            if subtitle_path is not None and subtitle_text_in_interval != "" and number_of_words< max_sub_len: 
                img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                number_of_words+=len(subtitle_text_in_interval.split(' '))
                subtitle_text_in_interval = ""
        frame_count += 1

        if len(images) >= max_images_length:
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(images) == 0:
        # skip the video if no frame is extracted
        return None,None
    images = torch.stack(images)
    instruction = img_placeholder + '\n' + instruction
    return images,instruction
def extract_audio(video_path, audio_path):
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="libmp3lame", bitrate="320k")
    
def generate_subtitles(video_path):
    video_id=video_path.split('/')[-1].split('.')[0]
    audio_path = f"workspace/inference_subtitles/mp3/{video_id}"+'.mp3'
    os.makedirs("workspace/inference_subtitles/mp3",exist_ok=True)
    if existed_subtitles.get(video_id,False):
        return f"workspace/inference_subtitles/{video_id}"+'.vtt'
    try:
        extract_audio(video_path,audio_path)
        print("successfully extracted")
        os.system(f"whisper {audio_path}  --language English --model large --output_format vtt --output_dir workspace/inference_subtitles")
        # remove the audio file
        os.system(f"rm {audio_path}")
        print("subtitle successfully generated")  
        return f"workspace/inference_subtitles/{video_id}"+'.vtt'
    except Exception as e:
        print("error",e)
        print("error",video_path)
        return None
    
QUESTIONS = [
    "What are the main actions or activities happening in the video?",
    "Who are the main characters or subjects appearing in the video?",
    "What is the setting or location where the video takes place?",
    "What objects or items are prominently featured in the video?",
    "What is the overall mood or atmosphere of the video?",
]

def save_to_json(args, save_path, video_name, answers):
    # Create a dictionary structure for JSON
    data = {
        video_name: [{"answer": answer, "prob": float(prob)} for answer, prob in answers]
    }
    
    # Define the full path to the JSON file
    full_path = os.path.join(args.save_dir, save_path)
    
    # Write data to the JSON file
    with open(full_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved data for {video_name} to {full_path}")

import os
from time import time
def run(video_dir, instruction, model, vis_processor, gen_subtitles=False, index=0, json_file_path=None):
    QUESTIONS = [
        "What are the main actions or activities happening in the video?",
        "Who are the main characters or subjects appearing in the video?",
        "What is the setting or location where the video takes place?",
        "What objects or items are prominently featured in the video?",
        "What is the overall mood or atmosphere of the video?"
    ]
    
    # Load the JSON file to get the list of videos to process
    if json_file_path:
        with open(json_file_path, "r") as f:
            video_data = json.load(f)
        
        video_names_to_process = set(video_data.keys())
    else:
        video_names_to_process = set(os.listdir(video_dir))
    
    for i, video_name in enumerate(os.listdir(video_dir)[index:], start=index):
        pre, ext = os.path.splitext(video_name)
        if pre not in video_names_to_process:
            continue
        
        s = time()
        video_path = os.path.join(video_dir, video_name)
        save_path = pre + ".json"
        
        if os.path.exists(os.path.join(args.save_dir, save_path)):
            continue
        
        answers = []
        for question in QUESTIONS:
            if gen_subtitles:
                subtitle_path = generate_subtitles(video_path)
            else:
                subtitle_path = None
            
            prepared_images, prepared_instruction = prepare_input(vis_processor, video_path, subtitle_path, question)
            if prepared_images is None:
                return "Video can't be opened, check the video path again"
            
            length = len(prepared_images)
            prepared_images = prepared_images.unsqueeze(0)
            conv = CONV_VISION.copy()
            conv.system = ""
            # If you want to make conversation, comment the 2 lines above and make the conv a global variable
            conv.append_message(conv.roles[0], prepared_instruction)
            conv.append_message(conv.roles[1], None)
            prompt = [conv.get_prompt()]
            result = model.generate(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=True, lengths=[length], num_beams=1)
            answers.append(result)
        
        save_to_json(args, save_path, video_name, answers)
        print(i, video_name, time() - s, "s")

    return answers


  
def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--add_subtitles",action= 'store_true',help="whether to add subtitles")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--save_dir", type=str, default=".", help="save dir")
    parser.add_argument("--index", type=int, default=0, help="index")
    parser.add_argument("--json_file_path", type=str, default="", help="json_file_path")
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    return parser.parse_args()
args=get_arguments()
def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

import yaml 
with open('test_configs/llama2_test_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
seed=config['run']['seed']
print("seed",seed)

model, vis_processor = init_model(args)
conv = CONV_VISION.copy()
conv.system = ""
inference_subtitles_folder="inference_subtitles"
os.makedirs(inference_subtitles_folder,exist_ok=True)
existed_subtitles={}
for sub in os.listdir(inference_subtitles_folder):
    existed_subtitles[sub.split('.')[0]]=True

if __name__ == "__main__":
    video_path=args.video_path
    instruction=args.question
    add_subtitles=args.add_subtitles
    index = args.index
    json_file_path=args.json_file_path
    # setup_seeds(seed)
    pred=run(video_path,instruction,model,vis_processor,gen_subtitles=add_subtitles,index=index,json_file_path=json_file_path)
