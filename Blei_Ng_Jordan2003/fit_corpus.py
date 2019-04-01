import numpy as np
import pandas as pd

import lda_svi

def read_data():
    word_id = pd.read_csv("words.txt", sep=" ", header=None, engine="python")
    word_count = pd.read_csv("docwords.txt", sep=" ", header=None, engine="python")
    # Strip all non-utf characters
    word_id = word_id.loc[word_id[1].str.match("[a-zA-Z]+").astype(bool), :]
    word_count = word_count.loc[word_count[1].str.match("[a-zA-Z]+").astype(bool), :]
    
    # Convert the frequency matrix into a document term matrix
    V = word_id.shape[0] # Length of vocabulary
    D = word_count.iloc[-1, 0]
    docterm = np.zeros((D, V))
    for docname, df in word_count.groupy(0):
        docterm[docname, :] = create_docterm(df, V)
    return docterm
    
def create_docterm(df, V):
    out = np.zeros(V)
    for i, row in df.iterrows():
        out[row[1] - 1] += 1
    return out
