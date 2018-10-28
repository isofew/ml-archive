FROM nvidia/cuda:9.0-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y python3 python3-pip curl

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN pip --no-cache-dir install tqdm
RUN pip --no-cache-dir install numpy
RUN pip --no-cache-dir install scikit-learn
RUN pip --no-cache-dir install torch torchvision
RUN pip --no-cache-dir install scipy
RUN pip --no-cache-dir install h5py

ADD HCopy /bin
ADD warp_ctc /warp_ctc
RUN cd /warp_ctc/pytorch_bindings && python setup.py install
