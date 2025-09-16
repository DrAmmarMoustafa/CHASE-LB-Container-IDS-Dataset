"""
train_chase_lb_multi.py

This script reproduces the CHASE-LB hybrid model training pipeline. 
It:
- Reads all .csv files from a given folder or glob pattern.
- Unifies columns and normalizes labels.
- Infers labels from filenames if not present.
- Builds, trains, and evaluates a hybrid ML/DL model (LSTM + TF-IDF + contextual features).
"""

import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import json
import scipy.stats as st
from statsmodels.stats.contingency_tables import mcnemar

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
SEQ_LEN = 500
TFIDF_NGRAM = (1, 3)
TFIDF_MAX_FEATURES = 20000
W2V_VECTOR_SIZE = 128
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
EMBEDDING_DIM = 128
LSTM_UNITS = 128
LSTM_DROPOUT = 0.3
CONTEXT_DENSE_UNITS = 32
FUSION_DENSE_UNITS = 128
FUSION_DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 30
EARLYSTOP_PATIENCE = 5
TRUNCATED_SVD_COMPONENTS = 128

# ------------------- helpers to read multiple CSVs -------------------
def infer_label_from_filename(fname):
    """Infer label from filename if no label column exists."""
    name = os.path.basename(fname).lower()
    if any(k in name for k in ['normal', 'benign']):
        return 1
    if any(k in name for k in ['attack', 'malicious', 'anom', 'intrusion']):
        return 0
    return np.nan  # Unknown

def read_and_tag_csv_files(path_or_pattern):
    """
    Read multiple CSV files from a directory or glob pattern.
    Adds 'source_file' column and infers label if missing.
    Returns a unified DataFrame.
    """
    if os.path.isdir(path_or_pattern):
        pattern = os.path.join(path_or_pattern, '*.csv')
    else:
        pattern = path_or_pattern

    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No CSV files found for pattern: {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}. Skipping.")
            continue
        df['source_file'] = os.path.basename(f)
        if 'label' not in df.columns:
            df['label'] = infer_label_from_filename(f)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {f} (label present: {'label' in df.columns})")
    if not dfs:
        raise ValueError("No readable CSV files found.")
    return pd.concat(dfs, ignore_index=True)

# ------------------- rest of code unchanged except comments -------------------
# (same as original you provided, with comments in English)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train hybrid IDS model on multiple CSV files (CHASE-LB style).")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to a folder containing CSV files (reads *.csv) or a glob pattern like 'data/*.csv'")
    args = parser.parse_args()
    main(args)