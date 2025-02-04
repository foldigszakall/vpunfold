# VPNet (Variable Projection Network) and Deep Unfolding Variable Projection Network

An experimental PyTorch package for research purposes that
1. provides an optimized, tensor operation based implementation of VPNet (Variable Projection Network) proposed in paper [P. Kovács, G. Bognár, C. Huber, and Mario Huemer. VPNet: Variable Projection Networks. International Journal of Neural Systems, 2022](https://doi.org/10.1142/S0129065721500544),
2. implements Deep Unfolding Variable Projection Network as of the presentation [G. Bognár, P. Kovács. ECG Classification with Deep Unfolding Variable Projection Network. Computing in Cardiology 2024](deep_unfolding_vpnet.pdf),
3. evaluates the provided model-based neural network models for arrhythmia classification of ECG heartbeats.

## Installation

The project requires Python 3 with `pytorch` (preferrably with CUDA support), `numpy`, and `scipy`. The development environment can be recreated as

    conda env create -f environment.yml

## ECG datasets and evaluation

The proof of concept application for ECG classification by means of adaptive Hermite VP involves the following components:
* `data`: dataset directory, see below for more details.
* `models`: output directory for models and evaluation logs (automatically created during simulations).
* `ecg_tools.py`: helper functions related to the ECG classification task.
* `ecg_classification.py`: command line tool for training and inference simulations, and hyperparameter optimization. For more information see

        python ecg_classification.py --help


ECG datasets are derived from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). Currently, two datasets are available:
* Balanced: `data/ecg_train.mat` and `data/ecg_test.mat` for training and testing, respectively. Balanced subset of normal and ventricular ectopic beats (VEBs), as of the original [VPNet paper](https://doi.org/10.1142/S0129065721500544).
* Full: `data/mitdb_filt35_w300adapt_ds1_float.mat` and `data/mitdb_filt35_w300adapt_ds2_float.mat` for training and testing, respectively. Preprocessed heartbeats of the whole database, as of the paper [G. Bognár, S. Fridli. ECG Heartbeat Classification by Means of Variable Rational Projection. Biomedical Signal Processing and Control, 2020](http://doi.org/10.1016/j.bspc.2020.102034). For more information about the preprocessing methodology and for different variants see [https://bognargergo.web.elte.hu/ecg_data/](https://bognargergo.web.elte.hu/ecg_data/)

## Development and contribution

The package is organized as follows:
* `vpnet/__init__.py`: package initialization and name export.
* `vpnet/vp_functions`: example function systems and the implementation of VP operators. If you would like to define a new parametric function system, then we suggest to create a subclass of `FunSystem`.
* `vpnet/networks`: example model-based neural network architectures involving VP operators and deep unfolded VP iterations. _If you would like to create a customized architecture of loss function_, then we hope that our implementation might provide building blocks and baseline examples.
* `test_vpnet`: unit tests for VP functions, operators, and layers. _If you expand the package_, then we suggest to define new unit tests in order to validate the numerical properties of your model.

If you find our contribution helpful for your work, then please cite the paper

> P. Kovács, G. Bognár, C. Huber, and Mario Huemer. VPNet: Variable Projection Networks. International Journal of Neural Systems, vol. 32, no. 1, pp. 2150054, 2022. DOI: [10.1142/S0129065721500544](https://doi.org/10.1142/S0129065721500544)

for applications of VPNet, and the upcoming paper

> G. Bognár et al. Deep Unfolding Variable Projection Network (to appear)

for the applications of VP deep unfolding.

## Notes

We also refer to the previous implementations of VPNet:
* the original [VPNet repository](https://git.silicon-austria.com/pub/sparseestimation/vpnet/) (numpy),
* the Spiking/Artificial Variable Projection Neural Networks, [VPSNN repository](https://github.com/KavehSam/VPSNN/) (pytorch).

## Licensing

The source code is free to use under the [MIT license](LICENSE). The dataset derived from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) are licensed under [Open Data Commons Attribution License v1.0](data/LICENSE).
