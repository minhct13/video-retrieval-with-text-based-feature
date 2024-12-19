from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:1234@localhost/videodb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class VideoKeyframe(db.Model):
    __tablename__ = "video_keyframes"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    frame_index = Column(Integer, nullable=False)
    marlin_video_vector = Column('marlin_video_vector', Vector(512))  # Using vector type (size 768)
    clip_vip_vector = Column('clip_vip_vector', Vector(512))  # Using vector type (size 512)

    video = relationship("Video", back_populates="keyframes")
    

class Video(db.Model):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    dataset = Column(String)
    keyframes = relationship("VideoKeyframe", back_populates="video")
