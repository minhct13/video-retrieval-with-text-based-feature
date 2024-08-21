import os
import json

# I have a folder with .mp4 files, provide python script to make the dataset for text or image to video retrieval as a single json file for ground truth, and a format of json file should look like {  {'face_video0.mp4': {"text_descriptions": ['a man with brown hair', 'the emotional ']} } }

with open("/root/CelebV-HQ/celebvhq_info.json") as f:
    data_dict = json.load(f)
    print(data_dict)