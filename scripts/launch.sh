#!/bin/bash

#python -m redseq -b True -e cross-pearson --scan True --scanrange 1 2 2 -s 250 --nseqs 10000 --temperature 1.0

python -m redseq -b True -e cross-pearson --scan True --scanrange -6 1 2 -s 250 --nseqs 10000 --temperature 0.6

python -m redseq -b True -e cross-pearson --scan True --scanrange -8 2 2 -s 250 --nseqs 10000 --temperature 0.3