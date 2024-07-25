from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from pgvector.sqlalchemy import Vector

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:1234@localhost/videodb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Video(db.Model):
    __tablename__ = 'videos'

    id = db.Column(db.String(200), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String, unique=True, index=True, nullable=False)
    path = db.Column(db.String, unique=True, nullable=False)
    video_vector = db.Column(Vector(512), nullable=False)
    text_vector_1 = db.Column(Vector(512), nullable=False)
    text_vector_2 = db.Column(Vector(512), nullable=False)
    text_vector_3 = db.Column(Vector(512), nullable=False)
    text_vector_4 = db.Column(Vector(512), nullable=False)
    text_vector_5 = db.Column(Vector(512), nullable=False)
    text_prob_1 = db.Column(db.Float, nullable=False)
    text_prob_2 = db.Column(db.Float, nullable=False)
    text_prob_3 = db.Column(db.Float, nullable=False)
    text_prob_4 = db.Column(db.Float, nullable=False)
    text_prob_5 = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<Video {self.name}>"

# @app.before_first_request
# def create_tables():
#     db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
