from flask import current_app
from app.services.encoded_service import EncoderService
import requests
class VideoService:
    def __init__(self):
        pass

    def query(self, text: str):
        """_summary_

        Args:
            text (str): _description_
        """
        # text_embedding = es.extract(text)

        return "", requests.codes.ok