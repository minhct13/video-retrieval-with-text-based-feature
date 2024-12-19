import requests
from flask import Blueprint, request
from PIL import Image
from app.services.video import VideoService


video_bp = Blueprint('video_bp', __name__)


@video_bp.route('/query', methods=["POST"])
def query():
    # body = request.get_json()
    query = request.form.get("query", None)
    top_n = request.form.get("top_n", 50)
    try:
        # Convert to a PIL image
        image = Image.open(request.files.get("image", None))
    except:
        image = None

    if not query and not image:
        return "" , requests.codes.ok
    elif image and not query:
        res, code = VideoService().query_by_image(image, top_n)
    elif query and not image: 
        res, code = VideoService().query_by_text(query, top_n)
    else:
        res, code = VideoService().query_by_image_text(query, image, top_n)
    return res, code


@video_bp.route('/suggest', methods=["GET"])
def suggest():
    res, code = VideoService().suggest()
    
    return res, code
