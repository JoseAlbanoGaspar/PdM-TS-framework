# PdM-TS-framework

This project aims to develop a framework that performs feature engineering operations over time series, addressing problems such as irregularity, value imputation, feature extraction and feature selection.
This framework can be found on a file called framework.ipynb.

## Case study

To test the developed framework, a dataset about a compression machine, that produces pills, was used.
The flow of the pipeline follows the following order
- framework_input.ipynb is responsible for pre-processing data from the original dataset, generating the input dataset for the framework
- framework.ipynb is responsible to exhaustively perform all possible combinations provided by the developed framework. It outputs a set of datasets that are used on the next step
- model.ipynb builds a pipeline for training models over the data previously obtained
- evaluation.ipynb compares the results obtained with benchmark approaches
