import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ic50_training import *
import pandas as pd
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Results & Analysis")


final_nn_path = 'models/best_nn_model.pt'

# gene_exp_cell_lines, smi_embed_df, smi_fp_df, ic50_train, ic50_test = load_training_data()
gene_exp_cell_lines, smi_fp_df, ic50_train, ic50_test = load_training_data()


ic50_train_merged, ic50_test_merged = merge_data(ic50_train, ic50_test, smi_fp_df, gene_exp_cell_lines)

X_train, y_train, X_test, y_test = create_train_test_sets(ic50_train_merged, ic50_test_merged)

# X_train, X_test, _ = scale_data(X_train, X_test)
#
# nn_model = create_model(X_train)

# nn_model = load_model(nn_model, final_nn_path)

# st.subheader("Neural Net Results: MSE and MAE")

# st.write("Load the test sets, and generate predictions using the fully trained model:")

# y_pred_nn = generate_predictions(nn_model, X_test, y_test)

st.subheader("XGBoost Results: MSE and MAE")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model('models/xgb_best_model.json')

# dtest = xgb.DMatrix(X_test, label=y_test)
y_pred_xgb = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_xgb)
mae = mean_absolute_error(y_test, y_pred_xgb)

st.write(f"Mean Squared Error on Test set: {mse:.4f}")
st.write(f"Mean Absolute Error on Test set: {mae:.4f}")


st.write("It seemed as though the neural net was overfitting. With more tuning, the neural net may be able to capture more intricate relationships in the data. However, for now, XGBoost proved to be a better option. We will continue the analysis with the XGB predictions")

ic50_test_merged['IC50_pred'] = y_pred_xgb

merged_test = ic50_test.join(ic50_test_merged[['IC50_pred']], how='left')

merged_test = merged_test.dropna(subset=['IC50', 'IC50_pred'])

# Assuming y_true and y_pred are your true and predicted values respectively.
y_true = merged_test["IC50"]
y_pred = merged_test["IC50_pred"]

correlation_coefficient = np.corrcoef(y_true, y_pred)[0, 1]

st.write(f"Pearson's correlation coefficient: {correlation_coefficient:.4f}")

st.write("The correlation shows a strong positive relationship between the true and predicted values. However, there is still room for improvement. The model does not seem to capture all the variability in the data. This is also exemplified in the residual plots below.")

X_test_with_predictions = pd.DataFrame(X_test)
X_test_with_predictions['True_IC50'] = y_test.values

X_test_with_predictions['Predicted_IC50'] = y_pred_xgb

# 1. Scatter plot of true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test_with_predictions['True_IC50'], X_test_with_predictions['Predicted_IC50'], alpha=0.5)
plt.plot([min(X_test_with_predictions['True_IC50']), max(X_test_with_predictions['True_IC50'])],
         [min(X_test_with_predictions['True_IC50']), max(X_test_with_predictions['True_IC50'])], color='red')
plt.xlabel("True IC50")
plt.ylabel("Predicted IC50")
plt.title("True vs Predicted IC50 values")
plt.grid(True)
st.pyplot(plt)  # This will display the plot in Streamlit

# 2. Histogram of the residuals
residuals = X_test_with_predictions['True_IC50'] - X_test_with_predictions['Predicted_IC50']
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.grid(True)
st.pyplot(plt)


st.write("The below df shows the actual test data and the predicted values. Some predictions were not made because there was no cell line info or gene_expression data for that cell_line.")
st.write(merged_test)

plt.figure(figsize=(10, 6))
sns.distplot(merged_test['IC50'], label='Actual IC50', hist=False)
sns.distplot(merged_test['IC50_pred'], label='Predicted IC50', hist=False)
plt.legend()
plt.title('Distribution of Actual vs. Predicted IC50 values')
st.pyplot(plt)

merged_test['residuals'] = abs(merged_test['IC50'] - merged_test['IC50_pred'])

best_predictions = merged_test.sort_values(by='residuals', ascending=True)
st.write("10 Predictions with Smallest Errors:")
st.write(best_predictions.head(10))

worst_predictions = merged_test.sort_values(by='residuals', ascending=False)
st.write("10 Predictions with Largest Errors:")
st.write(worst_predictions.head(10))


st.write("Overall, the predictions seem to be okay except for IC50 values on both ends of the distribution. Although there are no clear outliers, the model still struggled to capture the variance of the data")



cell_line_infos = pd.read_csv("data/Cell_lines_infos.csv")

cell_line_infos.rename(columns={"Name": "cell_line"}, inplace=True)
cell_line_infos.set_index("cell_line", inplace=True)

cell_line_infos.drop(columns=['COSMIC_ID', 'GDSC', 'Total', 'Count'], inplace=True)

cell_line_infos = cell_line_infos.groupby('cell_line').agg({
    'Tissue': 'first',
    'TCGA': 'first',
    'cancer_broad': 'first',
    'cancer_subtype': 'first'
})


merged_df = merged_test.merge(cell_line_infos, left_on='cell_line', right_index=True, how='left')

st.write(merged_df)

mean_residuals_per_cancer_broad = merged_df.groupby('cancer_broad')['residuals'].mean()
st.write("Mean Residuals Per Cancer Broad:")
st.dataframe(mean_residuals_per_cancer_broad.sort_values())

mean_residuals_per_cancer_subtype = merged_df.groupby('cancer_subtype')['residuals'].mean()
st.write("Mean Residuals Per Cancer Subtype:")
st.dataframe(mean_residuals_per_cancer_subtype.sort_values())

mean_residuals_per_drug = merged_df.groupby('drug')['residuals'].mean()
st.write("Mean Residuals Per Drug:")
st.dataframe(mean_residuals_per_drug.sort_values())

mean_residuals_per_TCGA = merged_df.groupby('TCGA')['residuals'].mean()
st.write("Mean Residuals Per TCGA:")
st.dataframe(mean_residuals_per_TCGA.sort_values())


st.write("The predictions are rather consistent for all types of cancer broad, cancer subtype, and TCGA. However, the IC50 of some drugs seem to be much easier to predict than others.")
