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
        return "", requests.codes.ok