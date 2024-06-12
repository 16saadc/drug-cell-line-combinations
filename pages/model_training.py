import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from ic50_training import *
import pandas as pd
import xgboost as xgb

st.subheader("Training Initial Model")

st.write("First lets merge our datasets with the ic50 data")

# gene_exp_cell_lines, smi_embed_df, smi_fp_df, ic50_train, ic50_test = load_training_data()
gene_exp_cell_lines, smi_fp_df, ic50_train, ic50_test = load_training_data()

st.write("The models trained using the transformer embeddings performed slightly worse than those trained using the morgan fingerprint data. With a more sophisticated tokenizer, transformer, and model, the embeddings may work better as they can capture more complex relationships in the data. But for now we will use the morgan fingerprints")

ic50_train, ic50_test = merge_data(ic50_train, ic50_test, smi_fp_df, gene_exp_cell_lines, printinfo=False)

st.write(ic50_train.head())

st.write("Given that our dataset is quite large (~200k samples) with many features (~1k), we will create a base model using XGBoost.")

X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_sets(ic50_train, ic50_test)

X_train_full, y_train_full, _, _ = create_train_test_sets(ic50_train, ic50_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

param_grid = {
    'objective': ['reg:squarederror'],
    'colsample_bytree': [0.5, 1.0],
    'learning_rate': [0.001, 0.01],
    'max_depth': [6, 12],
    'alpha': [0, 1],
    'verbosity': [1],
    'n_estimators': [200]  # added this to specify number of boosting rounds in GridSearchCV
}


st.subheader("XGBoost Regressor")

st.write("Start with a grid search to get a rough estimate of the RMSE score using these params: ")
st.write(param_grid)

st.write("The grid search takes a long time to run ~40-60 mins. So if you would like to re-run it, click the button. I have saved the best XGB model in 'models/'")

st.button("Run XGBoost Grid Search", on_click=xgboost_grid_search_cv, args=(param_grid, X_train_full, y_train_full, X_test, y_test, 'models/xgb_new_model.json'))


best_params = {'alpha': 0, 'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 300, 'objective': 'reg:squarederror', 'verbosity': 1}

st.button("Train XGBoost Model Using Best Params", on_click=train_xgboost, args=(best_params, X_train_full, y_train_full))

st.write("After several rounds of tuning, I found the best validation MSE to be: ~2.7")


st.write("Given the complexity and dimensions of the data, a neural net could be a better approach.")

st.subheader("Neural Net")

# X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

# model = create_model(X_train)

st.write("The following code trains and saves a new model\. If you'd like to change hypterparameters, go into model_training.py ")

st.write("The code uses early stopping to save the best model and avoid overfitting. The training is a heavy process, so we will avoid training here and will just generate predictions from the saved model. ")

model_path = 'models/best_model.pt'


st.write("This turned out to be the best setup, so we will use this setup to train the full training set and generate predictions on the test data")

# st.write("Let's train on the full training data, and then move on to generating results. The best NN has been saved, so no need to actually train")


# X_train_full, X_test, _ = scale_data(X_train_full, X_test)

# model = create_model(X_train_full)
#
# new_model_path = 'models/new_nn_model.pt'
#
#
# st.button("Train NN on all training data", on_click=train_nn, args=(
#             model,
#             X_train_full,   # x train
#             y_train_full,   # y train
#             None,           # x val
#             None,           # y val
#             600,            # epochs
#             50,             # patience
#             0.001,          # learning rate
#             1e-4,           # weight decay
#             new_model_path) #save path
#             )
