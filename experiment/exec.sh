#!/bin/bash

#python evaluation.py --verbose --pretrained /home/mwkim/F-TTFS/result/model/000_model_mnist_latency.weights
#python evaluation.py --verbose --latency_propagate_zero --pretrained /home/mwkim/F-TTFS/result/model/000_model_mnist_latency.weights
#python evaluation.py --verbose --latency_early_stopping --pretrained /home/mwkim/F-TTFS/result/model/000_model_mnist_latency.weights
#python evaluation.py --verbose --latency_early_stopping --latency_propagate_zero --pretrained /home/mwkim/F-TTFS/result/model/000_model_mnist_latency.weights

python evaluation.py -d cifar10 --steps 256 --verbose --pretrained /home/mwkim/F-TTFS/result/model/129_model_cifar10_latency.weights
python evaluation.py -d cifar10 --steps 256 --verbose --latency_propagate_zero --pretrained /home/mwkim/F-TTFS/result/model/129_model_cifar10_latency.weights
python evaluation.py -d cifar10 --steps 256 --verbose --latency_early_stopping --pretrained /home/mwkim/F-TTFS/result/model/129_model_cifar10_latency.weights
python evaluation.py -d cifar10 --steps 256 --verbose --latency_early_stopping --latency_propagate_zero --pretrained /home/mwkim/F-TTFS/result/model/129_model_cifar10_latency.weights
