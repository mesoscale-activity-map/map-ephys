
# map-ephys dockerfile

FROM datajoint/pydev:python3.6

RUN apt-get update && apt-get -y install mysql-client-5.7 netcat

RUN useradd -m map

USER map

ENV PATH=/home/map/.local/bin:$PATH

RUN pip3 install --upgrade --user pip \
	&& pip3 install --user datajoint==0.12.dev4

RUN pip3 install --user jupyter globus_sdk scipy matplotlib==3.1.3

ADD --chown=map:map . /src/map-ephys

RUN pip3 install --user -e /src/map-ephys \
	&& ln -s /src/map-ephys /home/map/map-ephys

WORKDIR /home/map

ENTRYPOINT [ "jupyter" ]

CMD [ "notebook", "--no-browser", "--ip", "0.0.0.0", "--port", "8888" ]

