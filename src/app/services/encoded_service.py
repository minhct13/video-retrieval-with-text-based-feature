import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict


from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast, AutoProcessor
from clipvip.CLIP_VIP import CLIPModel, clip_loss


class EncoderService:
    def init_app(self, checkpoint=""):
        self.extraCfg = edict({
            "type": "ViP",
            "temporal_size": 12,
            "if_use_temporal_embed": 1,
            "logit_scale_init_value": 4.60,
            "add_cls_num": 3
        })
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the checkpoint from the specified path
        self.checkpoint = torch.load(checkpoint, map_location=self.device)
        self.cleanDict = {key.replace("clipmodel.", ""): value for key, value in self.checkpoint.items()}
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
        # Load and configure the pre-trained CLIP model
        self.clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16")
        self.clipconfig.vision_additional_config = self.extraCfg
        self.model = CLIPModel(config=self.clipconfig).to(self.device)
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
        
    def encode_text(self, text):
        # Tokenize the text input
        tokens = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            textOutput = self.model.get_text_features(**tokens)
        return textOutput.cpu().numpy().flatten()

    def encode_video(self, video_path):
        # Process video file from the video_path
        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
        if video_path.endswith(".mp4"):
            container = av.open(video_path)
            clip_len = 12
            fcount = container.streams.video[0].frames
            indices = self.sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
            video = self.read_video_pyav(container, indices)
            pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values.to(self.device)
            inputs = {
                "if_norm": True,
                "pixel_values": pixel_values
            }

            with torch.no_grad():
                frame_features = self.model.get_image_features(**inputs)
            video_embedding = frame_features.mean(dim=0, keepdim=True).cpu().numpy().flatten()
        return video_embedding

