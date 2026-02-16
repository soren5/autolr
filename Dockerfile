FROM nunolourenco/lucy-base
ENV APP_DIR=/autolr
WORKDIR ${APP_DIR}
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
RUN mkdir -p /home/autolr
ENV APP_DIR=/home/autolr
WORKDIR ${APP_DIR}
ADD https://github.com/soren5/autolr.git#journal:requirements requirements 
RUN pip3 install -r requirements/requirements.txt
ADD https://github.com/soren5/autolr.git#journal .