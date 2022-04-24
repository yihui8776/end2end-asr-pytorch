FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install -i https://pypi.douban.com/simple torchaudio  tqdm python-Levenshtein librosa wget

RUN pip install -i https://pypi.douban.com/simple SoundFile numpy==1.19 numba==0.48.0 librosa==0.6.0

RUN  apt-get clean && \
  apt-get update --allow-insecure-repositories && \
  apt-get install -y libsndfile1  vim openssh-server && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# SSH Server
RUN sed -i 's/^\(PermitRootLogin\).*/\1 yes/g' /etc/ssh/sshd_config && \
    sed -i 's/^PermitEmptyPasswords .*/PermitEmptyPasswords yes/g' /etc/ssh/sshd_config && \
            echo 'root:ai1234' > /tmp/passwd && \
                    chpasswd < /tmp/passwd && \
                            rm -rf /tmp/passwd


RUN pip install jupyter -i https://pypi.doubanio.com/simple

COPY run_jupyter.sh  /workspace
COPY run_jupyter.sh /
RUN chmod +x  /run_jupyter.sh

WORKDIR /workspace

EXPOSE 22

EXPOSE 8888

CMD ["/run_jupyter.sh", "--allow-root"]

