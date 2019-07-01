# Base image
FROM tensorflow/tensorflow:1.13.1-gpu-py3

# specify working directory
WORKDIR /usr/src/autohighlight

# copy all files in directory excluding .git
COPY . /usr/src/autohighlight

# install dependencies
RUN pip install -r requirements.txt

# specify mount point
VOLUME /data/


# Environmental variables
ENV AUTOHL /usr/src/autohighlight
ENV AUTOHL_SPLIT /data/split
ENV AUTOHL_CLIPS /data/clips


# make directory for data
# RUN mkdir -p data/{src,clips,split}

CMD ["cat", "Instructions"]
