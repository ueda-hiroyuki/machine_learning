FROM python:3.6

WORKDIR /srv
COPY ./ /srv
# COPY ./src /src
# CMD chmod o+rw -R .
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt