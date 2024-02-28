# syntax=docker/dockerfile:1
FROM ubuntu:22.04
# create a directory to mount our data volume
RUN mkdir /project
#Install ubuntu libraires and packages
RUN apt-get update -y && \
    apt-get install git curl -y
# RUN apt-get install texlive-xetex -y
#Set some environemnt variables we will need
ENV PATH="/build/miniconda3/bin:${PATH}"
ARG PATH="/build/miniconda3/bin:${PATH}"
RUN mkdir /build && \
    mkdir /build/.conda
#Install Python3.10 via miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/miniconda3 &&\
    rm -rf /Miniconda3-latest-Linux-x86_64.sh
WORKDIR /build
RUN conda install python=3.10
RUN conda install numpy seaborn scipy tqdm scikit-learn
RUN conda install ipywidgets jupyterlab
RUN conda install -c conda-forge xarray dask netCDF4 bottleneck
