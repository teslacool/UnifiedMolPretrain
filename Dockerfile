FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

ENV LANG=C.UTF-8
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openssh-server  unzip curl \
    cmake gcc g++ \
    iputils-ping net-tools  iproute2  htop xauth \
    tmux wget vim git bzip2 ca-certificates \
    libxcursor1 libxdamage1 libxcomposite-dev libxrandr2 libxinerama1 \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /root/code

ENV envname pretrainmol3d
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n $envname python=3.7 && \
    conda activate $envname && \
    conda install -y pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia && \
    conda install pyg=2.0.2 -c pyg -c conda-forge && \
    conda install -y tensorboard tqdm scipy scikit-learn black ipykernel  && \
    conda install -y -c conda-forge rdkit=2020.09.5 openbabel && \
    conda install -y -c conda-forge graph-tool && \
    conda install -y tensorflow=1.15 && \
    conda install -y -c conda-forge deepchem && \
    conda clean -ay && \
    sed -i 's/conda activate base/conda activate '"$envname"'/g' /etc/profile

ENV PATH /opt/conda/envs/${envname}/bin:$PATH
EXPOSE 6006
RUN echo "export LANG=C.UTF-8" >> /etc/profile