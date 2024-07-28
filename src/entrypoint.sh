#!/bin/bash
# flask db migrate -m "Initial migration."
flask db upgrade
exec gunicorn -c './app/config/gunicorn.conf.py' run:gunicorn_app