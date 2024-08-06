from os import environ


def convert_list_object_from_string(string):
    """Convert a string to a list of objects"""
    return [] if not string else \
        list(map(lambda x: x.strip(), string.split(",")))

class Config():
    APP_ENV = environ["APP_ENV"]
    APP_API_PREFIX = environ["APP_API_PREFIX"]

    DEFAULT_USER = environ["DEFAULT_USER"]
    DEFAULT_PASSWORD = environ["DEFAULT_PASSWORD"]
    DEFAULT_DB = environ["DEFAULT_DB"]
    DEFAULT_HOST = environ["DEFAULT_HOST"]

    SERVICE_NAME = environ["SERVICE_NAME"]



    # DB centrialization
    SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{DEFAULT_USER}:{DEFAULT_PASSWORD}@{DEFAULT_HOST}/{DEFAULT_DB}"
    SQLALCHEMY_BINDS = {}
    VIDEO_DIR = environ["VIDEO_DIR"]
    # Set below value less than HAproxy client timeout
    # to avoid connection being killed while using
    SQLALCHEMY_POOL_RECYCLE = 300 # 5 mins
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CHECKPOINT_PATH = "pretrain_clipvip_base_16.pt"
    SQLALCHEMY_ENGINE_OPTIONS  = {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 10,
        "pool_recycle": SQLALCHEMY_POOL_RECYCLE
    }
    VIDEO_URL="https://7ba4-2405-4802-80ec-2fa0-48db-e50b-255-4e6f.ngrok-free.app"

class DevelopmentConfig(Config):
    MINIO_SECURE = False


class ProductionConfig(Config):
    MINIO_SECURE = True