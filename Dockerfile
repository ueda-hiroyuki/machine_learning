FROM python:3 

ADD ./app /src/app
WORKDIR /myapp

RUN pip install --upgrade pip
# RUN pip install -r requirements.txt