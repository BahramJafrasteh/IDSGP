# Input Dependent Sparse Gaussian Processes

This code is a TensorFlow 2.x implementation of the IDSGP method described in the paper entitled "Input Dependent Sparse Gaussian Processes", accepted at ICML 2022. 
SVGP_NN: our proposed method
For comparision, we also implemented the following methods from :

SVGP_Titsias: Titsias, Michalis. "Variational learning of inducing variables in sparse Gaussian processes." Artificial intelligence and statistics. PMLR, 2009.
SVGP_Hensman: Hensman, J., Matthews, A., & Ghahramani, Z. (2015, February). Scalable variational Gaussian process classification. In Artificial Intelligence and Statistics (pp. 351-360). PMLR.
SWSGP: Tran, G. L., Milios, D., Michiardi, P., & Filippone, M. (2021, July). Sparse within sparse Gaussian processes using neighbor information. In International Conference on Machine Learning (pp. 10369-10378). PMLR.
SVGP_SOLVE: Shi, Jiaxin, Michalis Titsias, and Andriy Mnih. "Sparse orthogonal variational inference for gaussian processes." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.



## Getting Started


### Prerequisites

The following package should be installed before using this code.

```
pip install pandas==1.1.5
pip install tensorflow==2.5.0rc3
pip install scipy==1.4.1
pip install matplotlib==3.3.4
pip install tensorflow_probability==0.12.2
pip install numpy==1.19.5
pip install scikit_learn==0.24.2
```

### Using the code
You can use the code as follows

```
put your data in "data" folder and run your experiments
you have the following optional arguments
python run_uci.py -h
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        name of the data set (should have subfolders with the
                        name s0, s1, s2, etc.) (default: None)
  --scaling SCALING     scaling method [MeanStd|MinMax|MaxAbs|Robust|None]
                        (default: MeanStd)
  --dataset_nsplit DATASET_NSPLIT
                        data set split number [0|1|2|etc] (default: 0)
  --modelSVGP MODELSVGP
                        the available SVGP models are [nn|solve|swsgp|titsias]
                        (default: nn)
  --Ptype PTYPE         Problem type (regression or classification)
                        [reg|class] (default: reg)
  --nGPU NGPU           GPU number (for cpu use -1) [-1|0|1|2] (default: -1)
  --nEpoch NEPOCH       Maximum number of epochs (default: 100) (default: 100)
  --BatchSize BATCHSIZE
                        Batch size (default: 100) (default: 100)
  --nip NIP             number of inducing points (default: 100) (default:
                        1024)
  --ncip NCIP           number of the closest inducing points (SWSGP)
                        (default: 50)
  --nhn1 NHN1           number of hidden nodes for the neural network
                        (default: 5) (default: 5)
  --nhl1 NHL1           number of hidden layers for the neural
                        network(default: 2) (default: 2)
  --ll LL               Likelihood type [gauss (Gaussian)|bern (Bernoulli
                        probit)|bern_sig (Bernoulli logit)|robust] (default:
                        gauss)
  --lr LR               learning rate (default: 0.01)
  --b1 B1               beta 1 parameter in Adam (default: 0.9)
  --b2 B2               beta 2 parameter in Adam (default: 0.999)
  --rdropout RDROPOUT   Dropout rate (default: 0.0)
  --kernel KERNEL       Kernel type [matern|rbf] (default: matern)


```

You can run experiments ucing UCI data set with the above options.


## Running the tests

In the folder examples there is demo for toy regression dataset and binary classification dataset. The codes generate the figures in the article

you can run the demo using the following command
```
python Toy_regression_example_increasing_M.py
python Toy_binary_example.py

```

The resulted image with the configuration will appear in the same folder with '.svg' format.



## Authors

https://openreview.net/pdf?id=HL_qE4fz-JZ
Jafrasteh*, B., Villacampa-Calvo*, C., & Hernández-Lobato, D. (2021, July). Sparse within sparse Gaussian processes using neighbor information. In International Conference on Machine Learning (pp. 10369-10378). PMLR.
*Equal contribution
*The main code has been developed by both first authors.


## License

This project is licensed under the MIT License.



