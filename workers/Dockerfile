FROM datajoint/djbase:latest

USER root
RUN apt update && \
    apt-get install -y ssh git

# RUN apt update && apt -y install mysql-client-5.7 netcat

USER anaconda:anaconda

# RUN pip install globus_sdk

RUN git clone https://github.com/mesoscale-activity-map/map-ephys ./dj_map

WORKDIR ./dj_map

RUN pip install --upgrade pip && pip install .

#RUN pip install --upgrade scipy matplotlib==3.1.3

RUN pip install datajoint==0.12.*

COPY ./dj_local_conf.json ./dj_local_conf.json

ENV DJ_SUPPORT_FILEPATH_MANAGEMENT TRUE
