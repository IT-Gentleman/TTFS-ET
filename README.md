# TTFS-ET: Early Termination for Forward Propagation of TTFS-Coded Data in SNNs with snnTorch

This repository contains the code and results for the implementation of **TTFS-ET (Time-to-First-Spike - Early Termination)** during the forward propagation of TTFS-coded data in spiking neural networks (SNNs) using [snntorch](https://snntorch.readthedocs.io/). The goal of this research is to improve computational efficiency and reduce power consumption during the evaluation phase by applying an early termination mechanism when propagating data that has been encoded using the Time-to-First-Spike (TTFS) coding scheme.

## Project Description

In this project, we explore the **TTFS-ET (Time-to-First-Spike - Early Termination)** method to enhance the performance during the forward propagation of TTFS-coded data in spiking neural networks. This technique reduces the time steps used during the evaluation phase by terminating the process once at least one spike has fired for all images in a batch. By skipping the remaining time steps, we achieve improvements in both speed and energy efficiency.
The repository includes:
- Python scripts implementing the **TTFS-ET** mechanism.
- Experimental results demonstrating the performance benefits of early termination during forward propagation of TTFS-coded data.

## Requirements

This project is designed to run on the **NVIDIA Jetson Orin Nano (Developer Kit)** with the following software stack:

### Hardware
- **Device**: NVIDIA Jetson Orin Nano (Developer Kit)

### Software
- **Jetpack 6.0**
- **Python 3.10**
- **PyTorch 2.3.0**
- **torchvision 0.18.0**
- **snntorch 0.9.1**

Make sure to have the correct versions of PyTorch, torchvision, and snntorch installed for compatibility with Jetpack 6.0. Refer to the [Jetpack installation guide](https://developer.nvidia.com/embedded/jetpack) for detailed steps to set up the environment.

## Installation

To set up the environment for this project, follow the instructions below.

### Step 1: Install Jetpack 6.0

Please follow the [official NVIDIA Jetpack installation guide](https://developer.nvidia.com/embedded/jetpack) to install Jetpack 6.0 on your NVIDIA Jetson Orin Nano.

### Step 2: Install Python Dependencies

For this project, you will need PyTorch, torchvision, and snntorch. Please use the provided `.whl` files for installation.

1. Download the necessary `.whl` files from the following links:
   - [PyTorch 2.3.0 for Jetson Orin Nano](https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl)
   - [torchvision 0.18.0 for Jetson Orin Nano](https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl)

   These `.whl` files were obtained from the [official NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

2. Install the `.whl` files using `pip`:

```bash
# Install PyTorch
pip install /path/to/your/downloaded/pytorch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install torchvision
pip install /path/to/your/downloaded/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

# Install snntorch from PyPI
pip install snntorch==0.9.1
```
Make sure that the paths to your .whl files are correct.

### Step 3: Verify Installation
After installation, you can verify that PyTorch and torchvision have been installed correctly by running:

```bash
python -c "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__)"
```

You can adjust parameters directly within the Python script to control the number of trials and other configurations.

## Usage

To run the experiments and compute average performance metrics such as speed improvement and power consumption reduction, execute the following command:

```bash
python ./experiment/evaluation.py -h
```

You can adjust parameters by using args to control the number of trials and other configurations.

## Results

The results of this project demonstrate that TTFS-ET significantly reduces the time steps used during the forward propagation of TTFS-coded data by terminating early once at least one spike has fired for all images in a batch. This leads to both performance improvements and lower power consumption when compared to traditional methods.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.