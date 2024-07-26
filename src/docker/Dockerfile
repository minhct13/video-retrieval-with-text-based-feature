FROM python:3.9-slim-buster

LABEL MAINTAINER="hoangduy.cqb.2k@gmail.com"
 
EXPOSE 8002

# Keeps Python from generating .pyc files in the container 
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging 
ENV PYTHONUNBUFFERED=1

# Set the timezone
ENV TZ=Asia/Ho_Chi_Minh
RUN set -x \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# COPY .docker/environments/staging/local-repository-config local-repository-config
# RUN cat local-repository-config > /etc/apt/sources.list
RUN apt-get update && apt-get install -y 

COPY requirements.txt .
# COPY .docker/environments/staging/pip.conf /etc/pip.conf

RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /src
COPY . /src

# During debugging, this entry point will be overridden.
RUN dos2unix ./entrypoint.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*
RUN chmod u+x ./entrypoint.sh

ENTRYPOINT ["/bin/sh", "./entrypoint.sh"]