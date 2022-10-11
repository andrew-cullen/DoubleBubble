#!/bin/bash
pip install --upgrade pip
pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user torchmetrics

NOISE_SET="0.5 1.0"
GAMMA_SET="0.001 0.005 0.01 0.0125 0.025 0.05 0.075 0.1 0.125 0.25 0.5"


for NOISE in $NOISE_SET; do
	echo $NOISE

	DATASET=tinyimagenet
	LOC=./gamma/$DATASET/$NOISE
	mkdir -p $LOC

	LOC_OUT=$LOC/experiments
	mkdir -p $LOC_OUT
	

	LOC=$LOC/testing-gamma

	echo $LOC
	
	for GAMMA in $GAMMA_SET; do
		echo "${GAMMA}is gamma and ${SIGMA} is sigma"
		python3 ./src/main.py --filename $LOC --dataset $DATASET --sigma $NOISE --gamma_start $GAMMA --samples 1000 --new_cr --certification_iters 500 --parallel eval >> $LOC_OUT/testing.out
	done
done	


