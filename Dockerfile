FROM nunolourenco/lucy-base
ENV APP_DIR=/home/pfcarvalho/autolr
WORKDIR ${APP_DIR}
COPY . /home/pfcarvalho/autolr
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
RUN apt-get update 
RUN apt-get install git
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
