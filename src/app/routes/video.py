import requests
from flask import Blueprint, request

from app.services.video import VideoService


video_bp = Blueprint('video_bp', __name__)


@video_bp.route('/query', methods=["POST"])
def query():
    body = request.get_json()
    query = body.get("query", None)
    top_n = body.get("top_n", 16)
    if not query:
        return "", requests.codes.ok
    res, code = VideoService().query(query, top_n)
    
    return res, code


@video_bp.route('/suggest', methods=["GET"])
def suggest():
    res, code = VideoService().suggest()
    
    return res, code
