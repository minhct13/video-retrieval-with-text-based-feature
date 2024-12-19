#!/bin/bash

# Script to initialize or update the PostgreSQL database schema for the videos and video_keyframes tables

# Database connection details (update as needed)
DB_CONTAINER="database"  # Docker container name for the database
DB_USER="user"           # Database username
DB_NAME="postgres"       # Database name
DB_EXTENSION="vector"    # Required PostgreSQL extension

echo "Initializing the database schema..."

# Check if the database is ready
until docker exec "$DB_CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME"; do
  echo "Waiting for the database to be ready..."
  sleep 5
done

echo "Database is ready. Updating schema..."

# Update the database schema
docker exec -i "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Enable the vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the existing videos table if it exists
DROP TABLE IF EXISTS videos;

-- Drop the existing video_keyframes table if it exists
DROP TABLE IF EXISTS video_keyframes;

-- Create the new videos table
CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    path VARCHAR UNIQUE NOT NULL,
    dataset VARCHAR
);

-- Create the video_keyframes table with a foreign key reference to the videos table
CREATE TABLE video_keyframes (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    marlin_video_vector vector(512),  -- Using vector type (size 512)
    clip_vip_vector vector(512),      -- Using vector type (size 512)
    UNIQUE(video_id, frame_index)
);

-- Confirm table creation
\\dt videos
\\dt video_keyframes
EOF

echo "Schema updated successfully."
