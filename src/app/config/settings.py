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

    MAIL_DEFAULT_SENDER = "noreply@flask.com"

    ADMIN_MINIO_URL = environ["ADMIN_MINIO_URL"]
    MINIO_ROOT_USER = environ["MINIO_ROOT_USER"]
    MINIO_ROOT_PASSWORD = environ["MINIO_ROOT_PASSWORD"]


    # DB centrialization
    SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{DEFAULT_USER}:{DEFAULT_PASSWORD}@{DEFAULT_HOST}/{DEFAULT_DB}"
    SQLALCHEMY_BINDS = {}

    # Set below value less than HAproxy client timeout
    # to avoid connection being killed while using
    SQLALCHEMY_POOL_RECYCLE = 300 # 5 mins
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SQLALCHEMY_ENGINE_OPTIONS  = {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 10,
        "pool_recycle": SQLALCHEMY_POOL_RECYCLE
    }

    # service URLs
    FRONTEND_URL = environ["FRONTEND_URL"]
    IMAGE_STORAGE = environ["IMAGE_STORAGE"]

    GOOGLE_CLIENT_ID = environ["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = environ["GOOGLE_CLIENT_SECRET"]
    GOOGLE_DISCOVERY_URL = (
        "https://accounts.google.com/.well-known/openid-configuration"
    )

    BING_API_KEY = environ["BING_API_KEY"]

class DevelopmentConfig(Config):
    MINIO_SECURE = False


class ProductionConfig(Config):
    MINIO_SECURE = True