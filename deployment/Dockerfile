FROM yihui8776/end2end-asr-pytorch:v1
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

ADD ./ /workspace

WORKDIR /workspace
RUN mv /workspace/sources.list /etc/apt/
RUN apt-get  update  && apt install -y ca-certificates --assume-yes apt-utils


RUN apt update && apt install -y  python3-pip


RUN pip install wave  matplotlib requests flask waitress -i https://pypi.douban.com/simple

# refer: https://docs.docker.com/engine/reference/builder/#expose
EXPOSE 20001/tcp

#ENTRYPOINT ["/bin/bash", "/asrt_server/start.sh"]

# https://docs.docker.com/engine/reference/commandline/build/#options
# docker build --progress plain --rm --build-arg TAG=1.2.0 --tag asrt/api_server:1.2.0 .
# https://docs.docker.com/engine/reference/commandline/run/#options
# docker run --rm -it  -p 20001:20001 --name asrt -d asrt/api_server:1.2.0