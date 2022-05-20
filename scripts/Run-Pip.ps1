# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

param (
    [ValidateSet("LatestRelease", "LatestDev", "LatestTest")]        
    [string]
    $sdkVersionSelect
)

# Add the new one
Write-Host "Now installing $sdkVersionSelect"
if( $sdkVersionSelect -eq "LatestRelease")
{
    pip install -r requirements.txt
}
elseif( $sdkVersionSelect -eq "LatestDev" )
{
    pip install -r requirements-dev.txt
}
elseif( $sdkVersionSelect -eq "LatestTest" )
{
    pip install -r requirements-dev-test.txt
}
else
{
    throw "Unrecognised sdkVersionSelect: $sdkVersionSelect"
}