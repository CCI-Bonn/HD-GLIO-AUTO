FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN sed 's/main$/main universe/' -i /etc/apt/sources.list
RUN apt-get -qq update && apt-get install wget git -y

# Neurodebian (maybe we just needed this for FSL, don't remember)
RUN wget -O- http://neuro.debian.net/lists/bionic.de-md.full | tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN for server in "ha.pool.sks-keyservers.net" "hkp://p80.pool.sks-keyservers.net:80" "keyserver.ubuntu.com" "hkp://keyserver.ubuntu.com:80" "pgp.mit.edu";\
    do apt-key adv --recv-keys --keyserver $server 0xA5D32F012649A5A9 && break; done
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get install mriconvert dcmtk nifti2dicom python python3 python3-pip vim nano -y

# Python
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --upgrade torch==1.4.0 nnunet matplotlib numpy pyxnat scikit-image IPython torchvision hd_glio
RUN pip3 install git+https://github.com/MIC-DKFZ/batchgenerators.git
RUN git clone https://github.com/MIC-DKFZ/HD-BET.git && pip3 install -e HD-BET
RUN echo "from HD_BET.utils import maybe_download_parameters\n\
for i in range(5):\n\
    maybe_download_parameters(i)"\
> hdbet_models.py && python3 hdbet_models.py
RUN echo "from hd_glio.setup_hd_glio import maybe_download_weights\n\
maybe_download_weights()"\
> hdglio_models.py && python3 hdglio_models.py

ADD scripts scripts

RUN python scripts/fslinstaller.py -q -d /usr/local/fsl
# we could also source /usr/local/fsl/etc/fslconf/fsl.sh,
# but I'm too lazy to look up how to do that (bash script.sh doesn't work)
ENV FSLDIR="/usr/local/fsl"
ENV FSLWISH="/usr/local/fsl/bin/fslwish"
ENV FSLDIR="/usr/local/fsl"
ENV FSLMACHINELIST=""
ENV FSLTCLSH="/usr/local/fsl/bin/fsltclsh"
ENV FSLREMOTECALL=""
ENV FSLLOCKDIR=""
ENV FSLGECUDAQ="cuda.q"
ENV FSLOUTPUTTYPE="NIFTI_GZ"
ENV FSLMULTIFILEQUIT="TRUE"
ENV PATH="${PATH}:${FSLDIR}/bin"

RUN apt-get -qq -y install bc locales
RUN locale-gen en_US.UTF-8
ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"

RUN mkdir /input && mkdir /output

ENTRYPOINT ["python3", "scripts/run.py", "-i", "/input", "-o", "/output"]

CMD []
