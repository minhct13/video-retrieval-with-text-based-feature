#!/bin/bash

# Wait for the database to be ready
# until docker exec database pg_isready -U user -d database; do
#   echo "Waiting for database to be ready..."
#   sleep 5
# done

# Initialize the database schema
docker exec -i database psql -U user -d postgres <<EOF
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    path VARCHAR UNIQUE NOT NULL,
    video_vector vector(512) NOT NULL,
    text_vector_1 vector(512),
    text_vector_2 vector(512),
    text_vector_3 vector(512),
    text_vector_4 vector(512),
    text_vector_5 vector(512),
    text_prob_1 FLOAT,
    text_prob_2 FLOAT,
    text_prob_3 FLOAT,
    text_prob_4 FLOAT,
    text_prob_5 FLOAT
);
EOF

echo "Database initialized."
