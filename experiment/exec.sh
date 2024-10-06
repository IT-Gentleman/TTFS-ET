#!/bin/bash

python execution.py --epoch 10 --single_gpu 1 --verbose --pretrained /home/mwkim/jupyter/root/F-TTFS/result/model/000_model_mnist_latency.weights
python execution.py --epoch 10 --single_gpu 1 --verbose --latency_early_stopping --pretrained /home/mwkim/jupyter/root/F-TTFS/result/model/000_model_mnist_latency.weights
python execution.py --epoch 10 --single_gpu 1 --verbose --latency_early_stopping --latency_propagate_zero --pretrained /home/mwkim/jupyter/root/F-TTFS/result/model/000_model_mnist_latency.weights