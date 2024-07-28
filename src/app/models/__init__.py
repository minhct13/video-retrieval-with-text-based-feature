import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import orm
from contextlib import contextmanager
from flask import g
from sqlalchemy_utils import database_exists, create_database


def create_database_schema(db_uri: str, schemas: list = []):
    engine = sqlalchemy.create_engine(db_uri)
    if not database_exists(engine.url):
        create_database(engine.url)
    for schema in schemas:
        if not engine.dialect.has_schema(engine, schema):
            engine.execute(sqlalchemy.schema.CreateSchema(schema))


db = SQLAlchemy()

from .video import Video


__all__ = ["video"]