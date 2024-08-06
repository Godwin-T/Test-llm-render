FROM python:3.9.7-slim

RUN pip install -U pip

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /home/

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]