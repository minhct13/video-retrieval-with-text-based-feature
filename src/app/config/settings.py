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
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 465
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_DEBUG = False
    MAIL_USERNAME = environ["EMAIL_USER"]
    MAIL_PASSWORD = environ["EMAIL_PASSWORD"]

    ADMIN_MINIO_URL = environ["ADMIN_MINIO_URL"]
    MINIO_ROOT_USER = environ["MINIO_ROOT_USER"]
    MINIO_ROOT_PASSWORD = environ["MINIO_ROOT_PASSWORD"]

    SECURITY_PASSWORD_SALT = environ["SECURITY_PASSWORD_SALT"]

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
    SECRET_KEY=environ["SECRET_KEY"]
    REALMS = environ["REALMS"]
    KEYCLOAK_CLIENT_ID = environ["KEYCLOAK_CLIENT_ID"]
    KEYCLOAK_SECRET_KEY = environ["KEYCLOAK_SECRET_KEY"]
    KEYCLOAK_URL = environ["KEYCLOAK_URL"]
    KEYCLOAK_ADMIN_CREDENTIALS = {
        'ADMIN_USERNAME':environ["KEYCLOAK_ADMIN_USERNAME"],
        'ADMIN_PASSWORD':environ["KEYCLOAK_ADMIN_PASSWORD"]
    }

    ADMIN_REALM_ROLE = "admin-realm"
    CLIENT_ROLES_DEFAULT = [
        "is_shelf_detection",
        "is_warning_rival_product",
        "is_anti_spoofing",
        "is_image_qc",
        "is_shelf_arrangement",
        "is_product_counting",
        "is_product_shortage",
        "is_shopboard_recognition",
    ]

    # service URLs
    FRONTEND_URL = environ["FRONTEND_URL"]
    IMAGE_STORAGE = environ["IMAGE_STORAGE"]

    GOOGLE_CLIENT_ID = environ["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = environ["GOOGLE_CLIENT_SECRET"]
    GOOGLE_DISCOVERY_URL = (
        "https://accounts.google.com/.well-known/openid-configuration"
    )

    BING_API_KEY = environ["BING_API_KEY"]
    CHECKPOINT_PATH = environ["CHECKPOINT_PATH"]

class DevelopmentConfig(Config):
    MINIO_SECURE = False


class ProductionConfig(Config):
    MINIO_SECURE = True