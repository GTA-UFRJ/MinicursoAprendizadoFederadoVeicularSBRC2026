# Dockerfile, Image, Container

FROM python:3.12

RUN mkdir -p ./app/code/
RUN mkdir -p ./app/code/utils/
RUN mkdir -p ./app/code/strategy/

ADD ./requirements.txt ./app/code/requirements.txt

ADD ./app.py                 ./app/code/app.py
ADD ./fhemda.py              ./app/code/strategy/fhemda.py
ADD ./fedavg.py              ./app/code/strategy/fedavg.py
ADD ./utils.py               ./app/code/utils/utils.py
ADD ./load_federated_data.py ./app/code/utils/load_federated_data.py

RUN pip install --upgrade pip
RUN pip install -r /app/code/requirements.txt

WORKDIR ./app/code/

RUN apt update && apt install inetutils-ping iproute2 vim -y