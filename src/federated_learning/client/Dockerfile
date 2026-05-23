# Dockerfile, Image, Container

FROM python:3.12

RUN mkdir -p ./app/code/
RUN mkdir -p ./app/code/utils/

ADD ./requirements.txt ./app/code/

ADD ./app.py ./app/code/app.py
ADD ./client.py ./app/code/client.py

ADD ./utils.py ./app/code/utils/utils.py
ADD ./models.py ./app/code/utils/models.py
ADD ./config.py ./app/code/utils/config.py
ADD ./load_federated_data.py ./app/code/utils/load_federated_data.py

RUN pip install --upgrade pip
RUN pip install -r /app/code/requirements.txt

WORKDIR ./app/

RUN apt update && apt install inetutils-ping iproute2 vim -y