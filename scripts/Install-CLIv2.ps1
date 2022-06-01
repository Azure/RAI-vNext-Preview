# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

param (
    [ValidateSet("LatestRelease", "LatestDev", "LatestTest")]        
    [string]
    $sdkVersionSelect
)

# Instructions from:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli

# Remove old extensions
az extension remove -n azure-cli-ml
az extension remove -n ml

# Add the new one
Write-Host "Now installing $sdkVersionSelect"
if( $sdkVersionSelect -eq "LatestRelease")
{
    az extension add -n ml -y
}
elseif ( $sdkVersionSelect -eq "LatestTest")
{
    az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/test-sdk-cli-v2/ml-latest-py3-none-any.whl --yes
}
elseif ( $sdkVersionSelect -eq "LatestDev")
{
    az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-latest-py3-none-any.whl --yes
}
else
{
    throw "Unrecognised sdkVersionSelect: $sdkVersionSelect"
}

# Upgrade to latest version
az extension update -n ml

# Check the commands
az ml -h


# Show the version
Write-Host
Write-Host "Info about ml extension:"
az extension show --name ml