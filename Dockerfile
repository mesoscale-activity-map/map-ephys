FROM datajoint/jupyter:latest

RUN pip install datajoint==0.12.dev4

RUN apt update && apt -y install mysql-client-5.7 netcat

RUN pip install globus_sdk

ADD . /src/map-ephys

RUN pip install -e /src/map-ephys