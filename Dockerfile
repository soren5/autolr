FROM nunolourenco/cdv-4-base
RUN apt-get update && apt-get install -y python3 python3-pip libjpeg8-dev git zlib1g-dev locales && locale-gen en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV APP_DIR /home/pfcarvalho/autolr
WORKDIR ${APP_DIR}
COPY . /home/pfcarvalho/autolr
RUN pip3 install --upgrade pip
