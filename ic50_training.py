from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from ic50_nn import IC50Net

device = "cpu"

@st.cache_data
def load_training_data():
    """
    Load preprocessed datasets for training and testing purposes.

    Args:
    - None

    Returns:
    - tuple:
        - gene_exp_cell_lines (pd.DataFrame): Gene expression data for cell lines.
        - smi_embed_df (pd.DataFrame): SMILES embeddings data.
        - smi_fp_df (pd.DataFrame): SMILES fingerprints data.
        - ic50_train (pd.DataFrame): IC50 training data.
        - ic50_test (pd.DataFrame): IC50 testing data.
    """
    gene_exp_cell_lines = pd.read_csv('data/gene_exp_cell_lines.csv')
    # smi_embed_df = pd.read_csv('data/smi_embeddings.csv')
    smi_fp_df = pd.read_csv('data/smi_fingerprints.csv')
    ic50_train = pd.read_csv('data/gdsc_train_processed.csv')
    ic50_test = pd.read_csv('data/gdsc_test_processed.csv')

    # return gene_exp_cell_lines, smi_embed_df, smi_fp_df, ic50_train, ic50_test
    return gene_exp_cell_lines, smi_fp_df, ic50_train, ic50_test

@st.cache_data
def merge_data(ic50_train, ic50_test, smi_data, gene_data, printinfo=True):
    """
    Merge IC50 data with gene and SMILES data based on drug and cell line identifiers.

    Args:
    - ic50_train (pd.DataFrame): Training data for IC50.
    - ic50_test (pd.DataFrame): Testing data for IC50.
    - smi_data (pd.DataFrame): SMILES data.
    - gene_data (pd.DataFrame): Gene expression data for cell lines.
    - printinfo (bool, optional): Whether to print intermediate information. Defaults to True.

    Returns:
    - tuple:
        - ic50_train (pd.DataFrame): Merged training data.
        - ic50_test (pd.DataFrame): Merged testing data.
    """
    smi_data.set_index('drug', inplace=True)
    gene_data.set_index('cell_line', inplace=True)

    initlength = len(ic50_test)
    if printinfo:
        st.write("initial test length")
        st.write(initlength)

    ic50_train = ic50_train.merge(gene_data, left_on='cell_line', right_index=True)
    ic50_train = ic50_train.merge(smi_data, left_on='drug', right_index=True)

    # TODO: assuming that all test drugs and cell lines are in the smi and gene exp dataframes --> throw exception if not
    ic50_test = ic50_test.merge(gene_data, left_on='cell_line', right_index=True)
    ic50_test = ic50_test.merge(smi_data, left_on='drug', right_index=True)

    afterlength = len(ic50_test)
    if printinfo:
        st.write("test length after merge")
        st.write(afterlength)

    diff = initlength - afterlength
    st.write(f"{diff} cell_lines or drugs were not in the gene_expression, cell line infos, or SMILES datasets. So, those rows are excluded")

    ic50_train.drop(['drug', 'cell_line', 'Unnamed: 0'], axis=1, inplace=True)
    ic50_test.drop(['drug', 'cell_line', 'Unnamed: 0'], axis=1, inplace=True)

    return ic50_train, ic50_test

@st.cache_data
def create_train_test_sets(ic50_train, ic50_test):
    """
    Split IC50 data into features and target for training and testing.

    Args:
    - ic50_train (pd.DataFrame): Merged training data.
    - ic50_test (pd.DataFrame): Merged testing data.

    Returns:
    - tuple:
        - X_train (pd.DataFrame): Training data features.
        - y_train (pd.Series): Training data target.
        - X_test (pd.DataFrame): Testing data features.
        - y_test (pd.Series): Testing data target.
    """
    X_train = ic50_train.drop('IC50', axis=1)

    y_train = ic50_train['IC50']

    X_test = ic50_test.drop('IC50', axis=1)
    y_test = ic50_test['IC50']

    return X_train, y_train, X_test, y_test

@st.cache_data
def create_train_val_test_sets(ic50_train, ic50_test):
    """
    Split IC50 data into features and target for training, validation, and testing.

    Args:
    - ic50_train (pd.DataFrame): Merged training data.
    - ic50_test (pd.DataFrame): Merged testing data.

    Returns:
    - tuple:
        - X_train (pd.DataFrame): Training data features.
        - y_train (pd.Series): Training data target.
        - X_val (pd.DataFrame): Validation data features.
        - y_val (pd.Series): Validation data target.
        - X_test (pd.DataFrame): Testing data features.
        - y_test (pd.Series): Testing data target.
    """
    X = ic50_train.drop('IC50', axis=1)

    y = ic50_train['IC50']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test = ic50_test.drop('IC50', axis=1)
    y_test = ic50_test['IC50']

    return X_train, y_train, X_val, y_val, X_test, y_test


def xgboost_grid_search_cv(param_grid, X_train, y_train, X_val, y_val):
    """
    Perform Grid Search Cross Validation for XGBoost regressor.

    Args:
    - param_grid (dict): Grid of hyperparameters to search over.
    - X_train (pd.DataFrame): Training data features.
    - y_train (pd.Series): Training data target.
    - X_val (pd.DataFrame): Validation data features.
    - y_val (pd.Series): Validation data target.

    Returns:
    - tuple:
        - best_estimator (xgb.XGBRegressor): Best XGBoost regressor model.
        - best_params (dict): Best hyperparameters found.
    """
    model = xgb.XGBRegressor()
    st.write("Training XGB Regressor using GridSearchCV")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Predicting and evaluating
    st.write(grid_search.best_estimator_)
    st.write(grid_search.best_params_)

    y_pred = grid_search.best_estimator_.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    st.write(f"Validation MSE: {mse:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_xgboost(params, X_train_full, y_train_full, save_path='models/xgb_best_model.json'):
    """
    Train an XGBoost regressor model.

    Args:
    - params (dict): Hyperparameters for XGBoost.
    - X_train_full (pd.DataFrame): Full training data features.
    - y_train_full (pd.Series): Full training data target.
    - save_path (str, optional): Path to save the best model. Defaults to 'models/xgb_best_model.json'.

    Returns:
    - bst (xgb.Booster): Trained XGBoost model.
    """
    dtrain = xgb.DMatrix(X_train_full, label=y_train_full)

    bst = xgb.train(params, dtrain, num_boost_round=400)
    bst.save_model(save_path)

    return bst


@st.cache_data
def plot_xgb_training_curve(cv_results):
    """
    Plot the training curve from XGBoost's cross-validation results.

    Args:
    - cv_results (dict): CV results from xgb.cv.

    Returns:
    - fig (matplotlib.figure.Figure): Figure showing the training curve.
    """
    train_mse = cv_results['train-rmse-mean']
    test_mse = cv_results['test-rmse-mean']
    rounds = np.arange(len(train_mse))

    fig, ax = plt.subplots()
    ax.figure(figsize=(10, 5))
    ax.plot(rounds, train_mse, label='Train MSE')
    axa.plot(rounds, test_mse, label='Test MSE')
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('XGBoost Training Curve')
    ax.legend()
    return fig

@st.cache_data
def scale_data(X_train, X_val=None, X_test=None):
    """
    Standard scale the data features.

    Args:
    - X_train (pd.DataFrame): Training data features.
    - X_val (pd.DataFrame, optional): Validation data features.
    - X_test (pd.DataFrame, optional): Testing data features.

    Returns:
    - tuple:
        - X_train (np.ndarray): Scaled training data.
        - X_val (np.ndarray, optional): Scaled validation data.
        - X_test (np.ndarray, optional): Scaled testing data.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val = scaler.transform(X_val)

    if X_test is not None:
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test

def train_nn(
            model,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            epochs=1000,
            patience=50,
            lr=0.001,
            weight_decay=1e-4,
            save_path='models/best_model.pt'
            ):
    """
    Train a neural network with early stopping.

    Args:
    - model (torch.nn.Module): Neural network model.
    - X_train (pd.DataFrame): Training data features.
    - y_train (pd.Series): Training data target.
    - X_val (pd.DataFrame, optional): Validation data features.
    - y_val (pd.Series, optional): Validation data target.
    - epochs (int, optional): Number of epochs. Defaults to 1000.
    - patience (int, optional): Patience for early stopping. Defaults to 50.
    - lr (float, optional): Learning rate. Defaults to 0.001.
    - weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-4.
    - save_path (str, optional): Path to save the best model. Defaults to 'models/best_model.pt'.

    Returns:
    - None
    """
    model.to(device)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1).to(device)

    # st.write(summary(model, input_size=X_train_tensor.size(), batch_size = -1))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stop_counter = 0
    best_val_loss = float("inf")
    patience = 100

    train_losses = []
    val_losses = []

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()

        if X_val is not None and y_val is not None:
            with torch.no_grad():
                outputs_val = model(X_val_tensor)
                val_loss = criterion(outputs_val, y_val_tensor)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        break

            val_losses.append(val_loss.item())

        train_losses.append(loss.item())


        if (epoch + 1) % 10 == 0:
            if X_val is not None and y_val is not None:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            else:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')


    if X_val is None and y_val is None:
        # save final model if no val
        torch.save(model.state_dict(), save_path)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    st.pyplot(fig)


@st.cache_resource
def create_model(X_train):
    """
    Create an IC50 neural network model.

    Args:
    - X_train (pd.DataFrame): Training data features.

    Returns:
    - model (IC50Net): IC50 neural network model.
    """
    st.write("create NN")
    input_dim = X_train.shape[1]
    model = IC50Net(input_dim).to(device)
    return model

@st.cache_resource
def load_model(_model, path):
    """
    Load weights into a neural network model from a saved state.

    Args:
    - _model (torch.nn.Module): Neural network model.
    - path (str): Path to the saved model state.

    Returns:
    - _model (torch.nn.Module): Neural network model with loaded weights.
    """
    st.write("load NN")
    _model.load_state_dict(torch.load(path, map_location=device))
    return _model

@st.cache_data
def generate_predictions(_model, _X_test, _y_test):
    """
    Generate predictions using a trained neural network model.

    Args:
    - _model (torch.nn.Module): Trained neural network model.
    - _X_test (pd.DataFrame): Testing data features.
    - _y_test (pd.Series): Testing data target.

    Returns:
    - y_pred (torch.Tensor): Predictions on the test set.
    """
    X_test_tensor = torch.FloatTensor(_X_test).float().to(device)
    y_test_tensor = torch.FloatTensor(_y_test).float().to(device)

    criterion = nn.MSELoss()

    _model.eval()
    with torch.no_grad():

        y_pred = _model(X_test_tensor)
        mse = criterion(y_pred, y_test_tensor).item()
        mae = torch.abs(y_pred - y_test_tensor).mean().item()

        st.write(f"Mean Squared Error on Test set: {mse:.4f}")
        st.write(f"Mean Absolute Error on Test set: {mae:.4f}")

    return y_pred
