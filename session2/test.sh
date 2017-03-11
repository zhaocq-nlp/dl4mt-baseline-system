#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu1,floatX=float32

python -u ./translate.py -n \
	./model_hal.npz  \
	./corpus.ch.pkl \
	./corpus.en.pkl \
	./MT02.ch \
	./MT02.trans > log.mt02 2>&1 &

