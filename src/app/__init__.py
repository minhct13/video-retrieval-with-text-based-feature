from app.routes.video import video_bp

from flask import Flask, g, request
from flask_cors import CORS
from flask_migrate import Migrate
from config import config
from extensions.minio import storage
from app.models import db
from app.models.db_utils import create_database_schema
migrate = Migrate()


def create_app():
    app = Flask(__name__)

    app.config.from_object(config)

    create_database_schema(app.config["SQLALCHEMY_DATABASE_URI"])

    db.init_app(app)
    migrate.init_app(app, db)
    storage.init_app(app)
    
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