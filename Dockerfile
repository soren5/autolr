FROM nunolourenco/cdv-4-base
RUN apt-get update && apt-get install -y python3 python3-pip libjpeg8-dev zlib1g-dev locales && locale-gen en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV APP_DIR /home/pfcarvalho/dsge_learning_rate
WORKDIR ${APP_DIR}
RUN pip3 install --upgrade pip
