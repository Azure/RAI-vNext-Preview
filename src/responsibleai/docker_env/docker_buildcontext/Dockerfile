FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20220418.v1

RUN apt-get -y update && apt-get -y install wkhtmltopdf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-0.18

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=21.3.1 -c anaconda -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
# Install pip dependencies
# markupsafe and itsdangerous are bug workarounds
RUN pip install -r /tmp/requirements.txt --no-deps


# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH