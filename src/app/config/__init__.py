import os

from app.config.settings import DevelopmentConfig, ProductionConfig

environment = os.getenv("APP_ENV", default="development")
config = DevelopmentConfig if environment == "development" else ProductionConfig