import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import pickle
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import WordPieceTrainer
from transformers import BertModel
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

@st.cache_data
def load_data():
    """
    Load relevant datasets.

    Returns:
    - tuple: Contains DataFrames for ic50 training and testing, gene expression, cell line info, SMILES data, and a set of focus genes.
    """
    ic50_train = pd.read_csv('data/gdsc_cell_line_ic50_train_fraction_0.9_id_997_seed_42.csv')
    ic50_test = pd.read_csv('data/gdsc_cell_line_ic50_test_fraction_0.1_id_997_seed_42.csv')
    gene_expression = pd.read_csv('data/gdsc-rnaseq_gene-expression.csv')
    cell_line_infos = pd.read_csv('data/Cell_lines_infos.csv')
    smi_data = pd.read_csv("data/gdsc.smi", sep="\t", header=None, names=["SMILES", "drug"])

    focus_genes=None
    with open('data/2128_genes.pkl', 'rb') as file:
        data = pickle.load(file)

        # make sure no duplicates
        focus_genes = set()
        for item in data:
            focus_genes.add(item.upper())

    return ic50_train, ic50_test, gene_expression, cell_line_infos, smi_data, focus_genes

@st.cache_data
def filter_focus_genes(gene_expression_df, focus_genes):
    """
    Filter the gene expression dataframe based on the given focus genes.

    Args:
    - gene_expression_df (DataFrame): Original gene expression data.
    - focus_genes (set): Set of genes to focus on.

    Returns:
    - DataFrame: Filtered gene expression data.
    """
    columns_to_keep = list(gene_expression_df.columns.intersection(focus_genes))
    columns_to_keep.append("Unnamed: 0")
    gene_expression_df = gene_expression_df[columns_to_keep]
    return gene_expression_df

@st.cache_data
def process_gene_data(gene_expression_df):
    """
    Pre-process the gene expression data by handling missing values and renaming columns.

    Args:
    - gene_expression_df (DataFrame): Raw gene expression data.

    Returns:
    - DataFrame: Processed gene expression data.
    """
    threshold = 1.0
    # we can drop more after we combine and check correlations -- some features may be very important despite not having many values
    columns_to_drop = gene_expression_df.columns[gene_expression_df.isnull().mean() == threshold]
    gene_expression_df.drop(columns_to_drop, axis=1, inplace=True)
    gene_expression_df.fillna(0, inplace=True)

    gene_expression_df.rename(columns={"Unnamed: 0": "cell_line"}, inplace=True)
    gene_expression_df.set_index('cell_line', inplace=True)

    return gene_expression_df


def create_rand_features_hist(data, n=20):
    """
    Display histograms of random feature columns from the data.

    Args:
    - data (DataFrame): The input data.
    - n (int): Number of random columns to visualize.

    Returns:
    - None
    """
    random_columns = data.sample(n=n, axis=1).columns

    n_rows = 5
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))

    for i, col in enumerate(random_columns):
        ax = axes[i//n_cols, i%n_cols]
        ax.hist(data[col], bins=50, alpha=0.5, label=col)
        ax.legend()
        ax.set_title(col)

    plt.tight_layout()

    st.pyplot(fig)

@st.cache_data
def perform_pca(_data, _index, var=0.99):
    """
    Perform Principal Component Analysis (PCA) on the data.

    Args:
    - _data (DataFrame): The input data.
    - _index (list): List of indices.
    - var (float): Amount of variance to maintain.

    Returns:
    - DataFrame: Transformed data after PCA.
    """
    pca = PCA(n_components=var)
    principal_components = pca.fit_transform(_data)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Line plot for cumulative variance explained
    fig, ax = plt.subplots()
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance Explained vs. Number of Principal Components')
    ax.grid(True)
    st.pyplot(fig)

    pca_df = pd.DataFrame(principal_components,
                   index=_index,
                   columns=[f"PC_{i+1}" for i in range(principal_components.shape[1])])

    return pca_df

@st.cache_data
def knn_impute_missing_vals(df, n_neighbors=2):
    """
    Use KNN imputation method to impute missing values in the dataframe.

    Args:
    - df (DataFrame): The input data.
    - n_neighbors (int): Number of neighbors for the KNN imputer.

    Returns:
    - DataFrame: Data with imputed values.
    """
    imputer = KNNImputer(n_neighbors=2)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_imputed


@st.cache_data
def train_bert_tokenizer():
    """
    Train a BERT tokenizer on SMILES data.

    Returns:
    - Tokenizer: Trained tokenizer.
    """
    berttokenizer = Tokenizer(models.WordPiece())
    berttokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    berttokenizer.decoder = decoders.WordPiece()

    trainer = WordPieceTrainer(show_progress=True, special_tokens=['[UNK]'])
    berttokenizer.save("./smiles/smiles_tokenizer.json")

    berttokenizer.train(["data/smiles.txt"], trainer=trainer)

    return berttokenizer

@st.cache_data
def get_transformer_embeddings(_smiles_list, _berttokenizer):
    """
    Use a transformer model to generate embeddings for the given SMILES data.

    Args:
    - _smiles_list (list): List of SMILES strings.
    - _berttokenizer (Tokenizer): Pre-trained tokenizer for encoding SMILES data.

    Returns:
    - tuple: Contains sequence and aggregated embeddings.
    """
    encoded_batch = _berttokenizer.encode_batch(_smiles_list)

    input_ids = [encoding.ids for encoding in encoded_batch]
    attention_masks = [encoding.attention_mask for encoding in encoded_batch]

    # add padding with [PAD] token
    max_len = max([len(seq) for seq in input_ids])
    padded_input_ids = [seq + [0]*(max_len - len(seq)) for seq in input_ids]
    padded_attention_masks = [mask + [0]*(max_len - len(mask)) for mask in attention_masks]

    st.write("Now that we have the tokens of all the SMILES data, we can use a transformer to create an embedding which we can use as features in our model")

    input_tensors = torch.tensor(padded_input_ids)
    attention_mask_tensors = torch.tensor(padded_attention_masks)

    model = BertModel.from_pretrained('bert-base-uncased')

    with torch.no_grad():
        outputs = model(input_tensors, attention_mask=attention_mask_tensors)

    sequence_embeddings = outputs[0]
    aggregated_embeddings , _ = sequence_embeddings.max(dim=1)

    return sequence_embeddings, aggregated_embeddings

@st.cache_data
def generate_morgan_fingerprints(_smiles_list, n_bits=512):
    """
    Generate Morgan fingerprints for a list of SMILES strings.

    Args:
    - smiles_list (list): List of SMILES strings.
    - n_bits (int): Number of bits for the fingerprint.

    Returns:
    - list: List of Morgan fingerprints.
    """

    morgan_fps = []

    for smiles in _smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit mol object
        if mol:  # Check if valid mol object
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fp_array = [int(bit) for bit in list(fp.ToBitString())]
            morgan_fps.append(fp_array)  # Convert fingerprint to bit string
        else:
            morgan_fps.append(None)

    return morgan_fps
