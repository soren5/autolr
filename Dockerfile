FROM nunolourenco/lucy-base
WORKDIR ${APP_DIR}
ENV APP_DIR=/autolr
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip3 install --upgrade pip
RUN apt-get update 
RUN apt-get install git -y
RUN cd /home && git clone https://github.com/soren5/autolr.git && cd /home/autolr
ENV APP_DIR=/home/autolr
WORKDIR ${APP_DIR}
RUN git checkout journal
RUN git pull
RUN pip3 install -r requirements.txt


