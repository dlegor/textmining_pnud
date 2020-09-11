[![GitHub license](https://img.shields.io/github/license/hamelsmu/code_search.svg)](https://github.com/dlegor/textmining_pnud/blob/master/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

# *textmining_pnud*

## Introduction

Project on the implementation of text mining techniques and NLP in public policy data. Part of the work of the PNUD Accelerator Lab Mexico, if you want to know more about all the projects that the Laboratory develops, you can visit the following page:

[Laboratorio de Aceleración México](https://www.mx.undp.org/content/mexico/es/home/accelerator-labs.html)

## Project Overview

The main part of the project can be found in the *"textmining_pnud"* folder. This package has all the necessary functions to explore, analyze and visualize the texts.

As support, the rest of the folders contain the following:

* Notebook: Jupyter Notebook with data exploration processes.
* Docker: Contains the app_basic to illustrate the functionalities and capabilities of language models to analyze texts.

To run on your local machine (or laptop) each folder has a README.md with instructions to follow. Please read the instructions to create the necessary environment to run the code.

## Setup 

To configure the environment, you must first have the *conda* package manager installed. If you do not have it installed, we recommend that you install the miniconda packages. You can find the instructions to install it at this link:

[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

To create the environment, you must run the following code in the folder of this repository.

~~~bash
conda env create -f environment.yml 
~~~

Check if the environment was created:
~~~bash
conda env list 
~~~

If the textmining-env environment is in the list, you can activate it using the following code:

~~~bash
conda activate  textmining-env
~~~

## Data Details

The data you need to replicate the notebooks and for which this package was developed can be downloaded from the following link:

[Avance de indicadores](https://www.transparenciapresupuestaria.gob.mx/es/PTP/programas#datos)

You can also find the dictionary with the description of the fields, in the same link.

## Licenses
This code and documentation for this project are released under the [MIT License](LICENSE).





