import os
import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict
import argparse

from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast, AutoProcessor
from clipvip.CLIP_VIP import CLIPModel, clip_loss


class EncoderService:
    def __init__(
            self,
            checkpoint=""
        ) -> None:
        self.extraCfg = edict({
            "type": "ViP",
            "temporal_size": 12,
            "if_use_temporal_embed": 1,
            "logit_scale_init_value": 4.60,
            "add_cls_num": 3
        })
        # Load the checkpoint from the specified path
        self.checkpoint = torch.load(checkpoint)
        self.cleanDict = {key.replace("clipmodel.", ""): value for key, value in self.checkpoint.items()}
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
        # Load and configure the pre-trained CLIP model
        self.clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16")
        self.clipconfig.vision_additional_config = self.extraCfg
        self.model = CLIPModel(config=self.clipconfig)
        self.model.load_state_dict(self.cleanDict)


    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])


    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len) if converted_len < seg_len else converted_len - 1
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
        

    def encode(self, video_path=None, text=None):
        """ Extract vector of video/ text """

        # Tokenize the text input
        tokens = self.tokenizer([text], padding=True, return_tensors="pt")
        textOutput = self.model.get_text_features(**tokens)
        print("Text Embedding Shape:", textOutput.shape)

        # Process video file from the video_path
        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
        if video_path.endswith(".mp4"):
            print(f"Processing {video_path}")
            container = av.open(video_path)
            clip_len = 12
            fcount = container.streams.video[0].frames
            indices = self.sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
            video = self.read_video_pyav(container, indices)
            pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

            inputs = {
                "if_norm": True,
                "pixel_values": pixel_values
            }

            with torch.no_grad():
                frame_features = self.model.get_image_features(**inputs)
            video_embedding = frame_features.mean(dim=0, keepdim=True)
            print("Video Embedding Shape:", video_embedding.shape)

            with torch.no_grad():
                sim = F.cosine_similarity(textOutput, video_embedding, dim=1)
                print(f"Cosine Similarity for: {sim.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to obtain embeddings and compute cosine similarity with text.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing MP4 video files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Text to compute embeddings for.")
    
    args = parser.parse_args()
    es = EncoderService(args.checkpoint).encode(args.video_dir, args.text)
