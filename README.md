# Machine Learning to Predict IC50 Values for Drug-Cell Line Combinations

Prior to running this app, please download the data and model files from here: 

https://www.dropbox.com/scl/fo/h5vcs5zruom740jv4hxt2/h?rlkey=w13wz87ay553mkz36w7w0tbj6&dl=0

Put the ```data/``` and ```models/``` directories as they are in the root folder of this repository

All explanations of my thinking process and my results are in the streamlit app, however there is a brief explanation below of the files contained in this repository.


### Setup

```pip install -r requirements.txt```

### Option 1 (recommended): Run the streamlit app using Dockerfile

#### Build the docker image

```docker build -t intel_test .```

#### Run the app, and open the displayed URL

```docker run -p 8501:8501 intel_test```


### Option 2: Run the python app locally (without docker)

```streamlit run IC50_predictions_main.py```


### Files

```data_processing.py```: Contains utility functions for processing GDSC IC50 data, gene expression data, SMILES data, and cell lines info data

```ic50_training.py```: Contains utility functions to train both an XGBoost model and a neural net to predict IC50 values for drug cell line Combinations

```ic50_nn.py```: Implements the neural network using pytorch

```IC50_predictions_main.py```: Main streamlit page


```pages/```: Contains scripts to preprocess data, train the models, and generate predictions and analysis to be displayed on streamlit.

```smiles/```: contains data to train the SMILES tokenizer

```models/```: contains my best XGBoost and neural net models
