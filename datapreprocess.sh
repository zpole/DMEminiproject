#!/bin/sh
rm -rf tokens*
rm -f data.zip
source activate dme
python preprocess/parse.py --stem False --stop False
python preprocess/parse.py --stem False --stop True
python preprocess/parse.py --stem True --stop False
python preprocess/parse.py --stem True --stop True
zip -r data.zip tokens* webkb
