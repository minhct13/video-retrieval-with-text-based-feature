from os import environ
from multiprocessing import cpu_count

def max_workers():
    return min(5, cpu_count() * 2 + 1)

ENVIRONMENT_DEBUG = environ.get("APP_DEBUG")
ENVIRONMENT_PORT = environ.get("APP_PORT")
ENVIRONMENT_HOST = environ.get("APP_HOST")

print("ENV:", ENVIRONMENT_DEBUG, ENVIRONMENT_HOST , ENVIRONMENT_PORT)

bind = "{}:{}".format(ENVIRONMENT_HOST, ENVIRONMENT_PORT)

reload = ENVIRONMENT_DEBUG
max_requests = 1000
timeout = 600

worker_class = 'gevent'
worker_connections = 1000
accesslog = "-" if ENVIRONMENT_DEBUG else "/var/log/gunicorn.access.log"
# Error log - records Gunicorn server goings-on
errorlog = "-" if ENVIRONMENT_DEBUG else "/var/log/gunicorn.error.log"
capture_output = True

loglevel="info"
workers = max_workers()