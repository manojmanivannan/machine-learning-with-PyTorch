# Machine Learning with PyTorch

This repository contains my python scripts and notebooks as I learn machine learning and AI using PyTorch (for now). 
Many thanks to [Daniel Voigt Godoy's Deep Learning with PyTorch](https://github.com/dvgodoy/PyTorchStepByStep) repository that I learned from.

## Requirments

Since I used a laptop with GPU ( NVIDIA RTX 3060 ) present, I installed GPU version of PyTorch. Check `requirements.txt`. There is another additional dependency. We need to install GraphViz to be able to use TorchViz, a package that allows us to visualize a model’s structure. Please check the installation instructions for your OS.

If you are using Windows, please use the installer at GraphViz's Windows Package. You also need to add GraphViz to the PATH (environment variable) in Windows. Most likely, you can find GraphViz executable file at C:\ProgramFiles(x86)\Graphviz2.38\bin. Once you found it, you need to set or change the PATH accordingly, adding GraphViz's location to it. For more details on how to do that, please refer to How to Add to Windows PATH Environment Variable.

## Google colab support

You can easily load the notebooks directly from GitHub using Colab and run them using a GPU provided by Google. You need to be logged in a Google Account of your own.

You can go through the chapters already using the links below:

- [Airfoil Self-Noise dataset - Linear Regression](https://colab.research.google.com/github/manojmanivannan/machine-learning-with-PyTorch/blob/master/notebooks/Airfoil_regression.ipynb) - dataset from [here](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise)

- [Combined Cycle Power Plant - Linear Regression](https://colab.research.google.com/github/manojmanivannan/machine-learning-with-PyTorch/blob/master/notebooks/PowerPlant_feature_engineered_regression.ipynb) - dataset from [here](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)