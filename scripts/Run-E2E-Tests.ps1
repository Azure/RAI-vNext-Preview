# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Enable all the commands
$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true

python -m pytest -n 6 ./test/  -m "not notebooks" -o junit_family=xunit2 --junitxml=junit.xml