# Leverage nvcr.io pre-built docker containers
FROM nvcr.io/nvidia/pytorch:21.03-py3 

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl \
 && apt-get update && apt-get install -y apt-utils unzip wget

# configure access to Jupyter
RUN jupyter notebook --generate-config --allow-root

# # copy over scripts & requirements
COPY ./start.sh /
COPY ./requirements.txt /

RUN apt-get update && apt-get install
RUN pip -q install -r /requirements.txt

# allocate port 8888 for Jupyter Notebook
# allocate port 6006 for tensorboard
EXPOSE 8888 6006

# mount volume
VOLUME /projects

# set cwd
WORKDIR /projects

# run start script
RUN chmod +x /start.sh
CMD ["/start.sh"]
