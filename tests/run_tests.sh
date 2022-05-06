export PYTHONPATH=$PYTHONPATH:..:../TangentCFT
python3 -m pytest -vv `find tests -name "*.py"`
