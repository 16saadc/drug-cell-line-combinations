import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import WordPieceTrainer
import numpy as np
import data_processor
from pytoda.smiles.processing import tokenize_smiles, kmer_smiles_tokenizer
from pytoda.smiles.smiles_language import SMILESTokenizer

st.title('Data Preprocessing')

ic50_train, ic50_test, gene_expression, cell_line_infos, smi_data, focus_genes = data_processor.load_data()

ic_50_null = ic50_train.isnull().sum()


st.subheader('GDSC IC50 Data')

st.write('This dataset contains the response of various cancer cell lines to different drugs, measured as the IC50 value. We will need to get features for drug and cell_line from the other two datasets.')
st.write(ic50_train)
st.write('Check for missing values')
st.write(ic50_train.isnull().sum())

fig, ax = plt.subplots()
ax.hist(ic50_train['IC50'], bins=30)
ax.set_title('Distribution of IC50 values in Training Set')
ax.set_xlabel('IC50')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.write('The IC50 values in the training set look to be normally distributed with no clear outliers. The data is already on a logarithmic scale: log(IC50 in micro-molar). So no scaling is needed, however this may effect which metric we want to use.')

st.write("Unnamed: 0 seems to be some kind of ID. we can drop it for now from the train and test sets")

ic50_train.drop(['Unnamed: 0'], axis=1, inplace=True)
ic50_test.drop(['Unnamed: 0'], axis=1, inplace=True)


st.write('In order to generate train the model to predict IC50 values, we will have to extract meaningful features for the cell_line and for the drug from the cell line infos, gene expression data and the SMILES data, respectively.')
st.write("Let's first look at the gene expression data")

st.subheader('Gene Expression Data')
st.write('This dataset contains the response of various cancer cell lines to different drugs, measured as the IC50 value. We will need to get features for drug and cell_line from the other two datasets.')
st.write(gene_expression)
st.write(gene_expression.shape)
st.write("First let's look at the 2128 genes that we care about, and remove the rest")
st.write(str(focus_genes)[:50] + "... ")

gene_expression = data_processor.filter_focus_genes(gene_expression, focus_genes)

st.write("New shape: ")
st.write(gene_expression.shape)

st.write('Check for missing values')
st.write(gene_expression.isnull().sum())
st.write('Some genes are missing values in all 457 rows. We can get rid of those. Others may have only a couple entries, but we do not know how important they might be. So we will only get rid of features that are missing all values. For other missing values, we will replace them with 0, denoting that this gene is not being expressed.')

st.write("'Unnamed: 0' value count max: ")
st.write(gene_expression['Unnamed: 0'].value_counts().max())

st.write("We also see that the 'Unnamed: 0' column corresponds to the 'cell_line' name in the GDSC Data, and are all unique values, so we can rename it and make it the index")


st.write("We also need to make sure that all the cell_lines present in the IC50 GDSC data are also present in the gene expression data. If they aren't, then we need some way to process those cell_lines when training and making predictions")

missing_cell_lines = ic50_train[~ic50_train['cell_line'].isin(gene_expression.index)]['cell_line'].unique()

st.write(len(missing_cell_lines))
st.write("Missing cell lines:", missing_cell_lines)


st.write("we could use the Cell info data, and try to find cell_lines that are most similar to the missing one using KNN. Then we can fill in the missing data in the gene_expression df")

gene_expression = data_processor.process_gene_data(gene_expression)

st.write(gene_expression)

st.write('Lets visualize some of these features in order to determine if we need to scale them and how.')

data_processor.create_rand_features_hist(gene_expression)

gene_scaler = StandardScaler()
scaled_data = gene_scaler.fit_transform(gene_expression)

gene_exp_pca_df = data_processor.perform_pca(scaled_data, gene_expression.index)

num_components = len(gene_exp_pca_df.keys())

st.write(f"number of principal components for 99% variance: {num_components}")
st.write(gene_exp_pca_df)


st.subheader("Cell Line Info Data")

st.write("Given that a lot of the cell_lines in the training data are missing from the gene expression data, we could impute their gene expression values using the cell info data")

cell_line_infos.rename(columns={"Name": "cell_line"}, inplace=True)
cell_line_infos.set_index("cell_line", inplace=True)

# Drop the unnecessary columns
cell_line_infos.drop(columns=['GDSC', 'Total', 'COSMIC_ID', 'Tissue', 'Count'], inplace=True)

# sum the 'count' values for each cell_line and merge other columns (assuming they're the same)
cell_line_infos = cell_line_infos.groupby('cell_line').agg({
    # 'Count': 'sum',
    'TCGA': 'first',
    'cancer_broad': 'first',
    'cancer_subtype': 'first'
})

st.write(cell_line_infos)

missing_cell_lines = ic50_train[~ic50_train['cell_line'].isin(cell_line_infos.index)]['cell_line'].unique()

st.write(len(missing_cell_lines))
st.write("Missing cell lines:", missing_cell_lines)

st.write("There are still some missing cell lines, but not as many as in the gene expression data")

st.write("let's encode this data, and then we can combine it with the gene expression data and impute the missing values")

encoded_cell_lines = pd.get_dummies(cell_line_infos, columns=['TCGA', 'cancer_broad', 'cancer_subtype'])

st.write(encoded_cell_lines)
cell_lines_gene_exp_df = encoded_cell_lines.merge(gene_exp_pca_df, left_index=True, right_index=True, how='outer')

st.write("now we merge both dataframes, and we can impute missing cell line data using KNN")

cell_lines_imputed = data_processor.knn_impute_missing_vals(cell_lines_gene_exp_df)

missing_cell_lines = ic50_train[~ic50_train['cell_line'].isin(cell_lines_imputed.index)]['cell_line'].unique()
st.write("Missing cell lines: ", len(missing_cell_lines))

st.write("now we've gone down from about 1.5k missing cell_lines in the training set to 70")



st.subheader("SMILES Data")
st.write("This data provides a textual representation of the drug's molecular structure")
smi_data.set_index('drug', inplace=True)
st.write(smi_data)

st.write("We will tokenize the SMILES strings and use a transformer to embed the chemical structure. I will try using the PyToda Library using their pretrained SMILES tokenizer: https://paccmann.github.io/paccmann_datasets/api/pytoda.smiles.smiles_language.html#pytoda.smiles.smiles_language.SMILESTokenizer")

smi_data['SMILES'].to_csv("data/smiles.txt", index=False, header=None)
smiles_list = smi_data['SMILES'].tolist()

st.write("we also need to check if all the training drugs are in the SMI data")

missing_drugs = ic50_train[~ic50_train['drug'].isin(smi_data.index)]['drug'].unique()

st.write("Missing drugs: ", len(missing_drugs))

st.write("There are a lot of drugs in the training data that are not in the test data. Since we only have the drug name, it doesn't make sense to impute their SMILES data values without more data. So, we will drop those rows")

smilestokenizer = SMILESTokenizer.from_pretrained('./smiles/')

st.write("Tokenize with pytoda. Let's test encoding and decoding one string")

st.write("Original SMILES string:")
st.write(smiles_list[1])
token_idxs = smilestokenizer.smiles_to_token_indexes(smiles_list[1])

st.write("Decoded SMILES string:")
st.write(smilestokenizer.token_indexes_to_smiles(token_idxs))

st.write("They seem to be different. Let's try creating a new tokenizer using the 'tokenizers' library. We'll try decoding the same string")

berttokenizer = data_processor.train_bert_tokenizer()

encoded_bert = berttokenizer.encode(smiles_list[1])

decoded = berttokenizer.decode(encoded_bert.ids)
st.write("Decoded String: ")
st.write(decoded)

st.write("This seems to be more accurate, so let's encode all the smi data with this tokenizer")

# sequence_embeddings, aggregated_embeddings = data_processor.get_transformer_embeddings(smiles_list, berttokenizer)

st.write("We use the mean over the sequence length to aggregate the embeddings and have a fixed size representation for each SMILES string")
st.write("Other aggregation methods might be more appropriate, like max pooling. We'll come back to this")
# st.write(f"smiles data length: {len(smi_data)}")
# st.write(f"embeddings shape: {aggregated_embeddings.size()}")
# st.write(aggregated_embeddings)

st.write("Now we have an embedding of length 768 for each SMILES string, which we can use as features in the GDSC data")

# smi_embed_df = pd.DataFrame(aggregated_embeddings,
#                             index=smi_data.index)
# st.write(smi_embed_df)

st.write("We will also generate the morgan fingerprints using RDKit, and compare the results with the model trained using the transformer embeddings")

fingerprints = data_processor.generate_morgan_fingerprints(smiles_list)
smi_fp_df = pd.DataFrame(fingerprints, index=smi_data.index)
st.write(smi_fp_df)

st.write("We have processed both datasets and now we can use their features in the GDSC data and get results from a basic model! We will save the processed gene expression and drug dataframes, so they are ready to be merged with the training data")


# save the processed data
cell_lines_imputed.to_csv('data/gene_exp_cell_lines.csv')
# smi_embed_df.to_csv('data/smi_embeddings.csv')
smi_fp_df.to_csv('data/smi_fingerprints.csv')
ic50_train.to_csv('data/gdsc_train_processed.csv')
ic50_test.to_csv('data/gdsc_test_processed.csv')
