FROM nunolourenco/lucy-base
ENV APP_DIR /home/pfcarvalho/autolr
WORKDIR ${APP_DIR}
COPY . /home/pfcarvalho/autolr
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements369.txt
