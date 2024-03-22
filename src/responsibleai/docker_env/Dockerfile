FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20231011.v1

RUN apt-get -y update && apt-get -y install wkhtmltopdf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai

# Create conda environment
# Note libffi is pinned as it causes scorecard component to fail
# when using the wkhtmltopdf package with libgobject error
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=21.3.1 libffi=3.3 -c anaconda -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'responsibleai~=0.31.0' \
                'pyarrow' \
                'markupsafe<=2.0.1' \
                'itsdangerous==2.0.1' \
                'mlflow' \
                'scikit-learn~=1.2' \
                'pdfkit==1.0.0' \
                'plotly==5.6.0' \
                'kaleido==0.2.1' \
                'protobuf<4' \
                'mltable==1.4.1' \
                'responsibleai-tabular-automl==0.8.0' \
                'raiwidgets~=0.31.0' \
                'https://publictestdatasets.blob.core.windows.net/packages/pypi/raiwidgets_big_data/raiwidgets_big_data-0.7.0-py3-none-any.whl'

RUN pip install --pre 'azure-ai-ml~=1.14.0' 'azure-storage-blob<=12.13.0' 'numpy<1.24.0'

# no-deps install for domonic due to unresolable dependencies requirment on urllib3 and requests. 
RUN pip install --no-deps 'charset-normalizer==2.0.12' \
                          'cssselect==1.1.0' \
                          'elementpath==2.5.0' \
                          'html5lib==1.1' \
                          'webencodings==0.5.1' \
                          'domonic==0.9.10'

# Install azureml packages
RUN pip install 'azureml-dataset-runtime==1.53.0' \
                'azureml-core==1.53.0' \
                'azureml-mlflow==1.53.0' \
                'azureml-telemetry==1.53.0' \
                'azureml-rai-utils==0.0.6'

# To resolve vulnerability issue
RUN pip install 'Werkzeug==2.2.3'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
