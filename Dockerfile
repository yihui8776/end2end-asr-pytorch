FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install -i https://pypi.douban.com/simple torchaudio  tqdm python-Levenshtein librosa wget

RUN pip install -i https://pypi.douban.com/simple SoundFile numpy==1.19 numba==0.48.0 librosa==0.6.0

RUN  apt-get clean && \
  apt-get update --allow-insecure-repositories && \
  apt-get install -y libsndfile1  vim openssh-server && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*



WORKDIR /workspace

