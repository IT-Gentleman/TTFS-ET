#!/bin/bash

for i in {1..10}
do
    python power_measurement.py "python evaluation.py -d mnist --batch_size 64 --steps 8 --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/265_model_mnist_latency.weights --model simple"
    python power_measurement.py "python evaluation.py -d mnist --batch_size 64 --steps 8 --early_termination --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/265_model_mnist_latency.weights --model simple"

    python power_measurement.py "python evaluation.py -d mnist --batch_size 64 --steps 8 --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/266_model_mnist_latency.weights --model smlp78440010"
    python power_measurement.py "python evaluation.py -d mnist --batch_size 64 --steps 8 --early_termination --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/266_model_mnist_latency.weights --model smlp78440010"

    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/fashionmnist_simple.weights --model simple"
    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --early_termination --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/fashionmnist_simple.weights --model simple"

    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/fashionmnist_scnn_b128.weights --model scnn"
    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --early_termination --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/fashionmnist_scnn_b128.weights --model scnn"

    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/236_model_fashionmnist_latency.weights --model smlp78440040010"
    python power_measurement.py "python evaluation.py -d fashionmnist --batch_size 64 --steps 16 --early_termination --pretrained /home/mwkim/TTFS-ET/experiment/ref_model/236_model_fashionmnist_latency.weights --model smlp78440040010"

done