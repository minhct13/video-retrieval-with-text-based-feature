import requests
from flask import Blueprint, request

from app.services.video import VideoService


video_bp = Blueprint('video_bp', __name__)


@video_bp.route('/query', methods=["POST"])
def query():
    body = request.get_json()
    queries = body.get("query", None)
    if not queries:
        return None, requests.codes.ok
    res, code = VideoService().query()
    return res, code