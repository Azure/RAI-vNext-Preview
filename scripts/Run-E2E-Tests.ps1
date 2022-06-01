# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

python -m pytest -n 3 ./test/  -m "not notebooks" -o junit_family=xunit2 --junitxml=junit.xml