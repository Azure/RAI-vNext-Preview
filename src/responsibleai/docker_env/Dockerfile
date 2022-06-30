FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20220418.v1

RUN apt-get -y update && apt-get -y install wkhtmltopdf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-0.18

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=21.3.1 -c anaconda -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'responsibleai~=0.18.1' \
                'raiwidgets~=0.18.1' \
                'pyarrow' \
                'mlflow' \
                'azureml-core==1.41.0.post1' \
                'azureml-dataset-runtime==1.41.0' \
                'azureml-mlflow==1.41.0' \
                'azureml-telemetry==1.41.0' \
                'pdfkit==1.0.0' \
                'plotly==5.6.0' \
                'kaleido==0.2.1' \
                'protobuf<4'

RUN pip install --pre azure-ai-ml

# no-deps install for domonic due to unresolable dependencies requirment on urllib3 and requests. 
RUN pip install --no-deps 'charset-normalizer==2.0.12' \
                          'cssselect==1.1.0' \
                          'elementpath==2.5.0' \
                          'html5lib==1.1' \
                          'webencodings==0.5.1' \
                          'domonic==0.9.10'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
