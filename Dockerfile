FROM datajoint/jupyter:python3.6

RUN apt update && apt -y install mysql-client-5.7 netcat

RUN pip install globus_sdk

ADD . /src/map-ephys

RUN pip install -e /src/map-ephys

