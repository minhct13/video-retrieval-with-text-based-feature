from app.routes.video import video_bp
from flask_cors import CORS
from flask import Flask, g, request
from app.config import config
from app.models import db
from app.models.db_utils import create_database_schema
from app.services.encoded_service import EncoderService

def create_app():
    app = Flask(__name__)

    app.config.from_object(config)

    create_database_schema(app.config["SQLALCHEMY_DATABASE_URI"])
    # es = EncoderService(app.config["CHECKPOINT_PATH"])
    app.db = db
    app.db.init_app(app)

    app.es = EncoderService()
    app.es.init_app(app.config["CHECKPOINT_PATH"])
    
    api_prefix = app.config["APP_API_PREFIX"]
    CORS(
    app, resources={
            rf"{api_prefix}/*": {
            "origins": "*",
            "supports_credentials": True,
            }
        }
    )
    # Import a module / component using its blueprint handler variable
    app.register_blueprint(video_bp, url_prefix=api_prefix)

    return app