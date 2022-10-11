#!/bin/bash

echo $(date)

pip install --upgrade pip
pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user torchmetrics


NOISE_SET="0.1 0.2 0.3 0.4 0.7 0.8 0.9"
GAMMA_SET="0.01"

DATASET=tinyimagenet
LOC=./sigma/$DATASET/
mkdir -p $LOC

LOC_OUT=$LOC/experiments
mkdir -p $LOC_OUT


for NOISE in $NOISE_SET; do
	echo $NOISE

	DATASET=tinyimagenet
	LOC=./sigma/$DATASET/
	mkdir -p $LOC

	LOC_OUT=$LOC/experiments
	mkdir -p $LOC_OUT
	
	rm $LOC_OUT/testing-T-S-1.out

	LOC=$LOC/testing-sigma

	echo $LOC
	
	for GAMMA in $GAMMA_SET; do
		echo "${GAMMA}is gamma and ${NOISE} is sigma"
		python3 main.py --filename $LOC --dataset $DATASET --sigma $NOISE --gamma_start $GAMMA --samples 1000 --new_cr --certification_iters 500 --parallel eval >> $LOC_OUT/testing.out
	done
done	


