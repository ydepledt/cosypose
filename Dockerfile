FROM nvcr.io/nvidia/nvhpc:21.2-devel-cuda_multi-ubuntu20.04

SHELL ["/bin/bash", "-c"]

ADD Miniconda3-latest-Linux-x86_64.sh /
RUN bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate cosypose" >> ~/.bashrc \
 && /opt/conda/bin/conda update -n base -c defaults conda


WORKDIR /host_pwd
ADD . .
RUN /opt/conda/bin/conda env create -n cosypose --file environment.yaml 
RUN /opt/conda/bin/conda run -n cosypose python --version setup.py install

RUN apt-get update -qqy && DEBIAN_FRONTEND=noninteractive apt-get install -qqy meshlab && rm -rf /var/{cache,lib}/apt

#ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=0,1,2,3
