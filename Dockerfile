FROM datajoint/jupyter:latest

RUN apt update && apt -y install mysql-client-5.7 netcat

RUN pip install globus_sdk

ADD . /src/map-ephys

RUN pip install -e /src/map-ephys

