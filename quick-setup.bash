#!/bin/bash

condaEnv=$1
subId=$2
rgName=$3
wsName=$4

if [ -z $4 ]; then
    echo "Expected command line: quick-setup.bash <CONDA-ENV> <SUB-ID> <RG-NAME> <WS-NAME>"
    exit
fi

echo "Will create conda environment $condaEnv"
echo "For subscription $subId, resourcegroup $rgName and workspace $wsName"

sleep 1

echo

echo "=-= Creating config.json"
echo "{ \"subscription_id\":\"${subId}\", \"resource_group\":\"$rgName\", \"workspace_name\":\"$wsName\" }" > config.json
echo

echo "=-= Creating comoponent_config.json"
echo "{ \"version\":\"1\" }" > component_config.json
echo

# echo "=-= Creating conda environment"
# source ~/miniconda3/etc/profile.d/conda.sh
# conda create -y -n ${condaEnv} python=3.8
# conda activate ${condaEnv}
# echo

echo "=-= Installing nbconda"
conda install nbconda
echo

echo "=-= Installing Jupyter"
pip install jupyter
echo

echo "=-= Installing responsibleai"
pip install responsibleai~=0.18.0 raiwidgets~=0.18.0 pyarrow
echo

echo "=-= Instaling other requirements"
pip install -r requirements.txt
echo

echo "=-= Runnning AzureML registrations"
python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory .
echo

echo "Configuring Azure CLI defaults"
az account set -s $subId
az configure --defaults group=$rgName workspace=$wsName
