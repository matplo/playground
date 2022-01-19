# Quick installation of alice O2 in a docker

## Dockerfile

```
#build on top of ubuntu
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# gcc and git - possibly done via deps from the next steps
RUN apt-get update && \
        apt-get -y install gcc git

# this is from ALICE web pages 
# used prereq. from https://alice-doc.github.io/alice-analysis-tutorial/building/prereq-ubuntu.html 

RUN apt-get -y install curl libcurl4-gnutls-dev build-essential gfortran libmysqlclient-dev xorg-dev libglu1-mesa-dev libfftw3-dev libxml2-dev git unzip autoconf automake autopoint texinfo gettext libtool libtool-bin pkg-config bison flex libperl-dev libbz2-dev swig liblzma-dev libnanomsg-dev rsync lsb-release environment-modules libglfw3-dev libtbb-dev python3-dev python3-venv libncurses-dev software-properties-common

# installing aliBuild
RUN add-apt-repository ppa:alisw/ppa
RUN apt update
RUN apt -y install python3-alibuild

# some useful editors - totally optional
RUN apt-get -y install emacs vim nano
```

## Build the docker image

- assuming your Dockerfile is in current dir and it is called Dockerfile

```
docker build -f Dockerfile .
```

- to enter the image I check `docker images` and use `docker run -it XXXX` using the checksum for XXXX
- for example `docker run -it 36d6601556be`

## ALICE software

- now within the docker

```
cd $HOME
echo "export ALIBUILD_WORK_DIR=\"$HOME/alice/sw\"" | tee .bashrc
echo "eval \"`alienv shell-helper`\"" | tee .bashrc
. ./bashrc
```

- now from https://alice-doc.github.io/alice-analysis-tutorial/building/build.html 

```
mkdir -p ~/alice
cd ~/alice
aliBuild init O2Physics@master --defaults o2
aliDoctor O2Physics --defaults o2
aliBuild build O2Physics --defaults o2
```

- see also this https://aliceo2group.github.io/analysis-framework/docs/installing/ 
