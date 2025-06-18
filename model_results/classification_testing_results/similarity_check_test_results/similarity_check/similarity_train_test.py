import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
import time
import math
# Example initialization (to be replaced with actual data)
df_train = pd.read_csv(r"C:\Users\harry\Desktop\ppi_train_ds.csv")
df_test = pd.read_csv(r"C:\Users\harry\Desktop\ppi_test_ds.csv")
#df_test= df_test.sample(n=10)
print(df_test)
# Function to perform pairwise alignment
def pairwise_align(seq1, seq2):
    aligner = Align.PairwiseAligner( scoring="blastp")
    if pd.notnull(seq1) and pd.notnull(seq2):
        score = aligner.score(seq1, seq2)
        #print(score)
        return score
    else:
        return None

def calc_e_val(seq1, seq2, score):
    if pd.notnull(seq1) and pd.notnull(seq2) and score is not None:
        # m and n are the lengths of the sequences
        m = len(seq1)
        n = len(seq2)
        length= (m+n)/2
        K=0.035 
        lambda_=0.252
        e_val = K * m*n* math.exp(-lambda_ * score)
        return e_val
    else:
        return None


# Initialize the 'similarity_check' column
df_test['similarity_check'] = 0

# Start counting time from t1
start_time = time.time()

# Iterate through each pair in df_test and df_train to compare their sequences
for idx_test, row_test in df_test.iterrows():
    seq_A_test = row_test['seq_A']
    seq_B_test = row_test['seq_B']
    
    similarity_found = False
    
    for idx_train, row_train in df_train.iterrows():
        seq_A_train = row_train['seq_A']
        seq_B_train = row_train['seq_B']
        
        # Check if any sequence is None in either pair
        if seq_A_test is None or seq_B_test is None or seq_A_train is None or seq_B_train is None:
            continue
        
        # Calculate pairwise alignment scores
        score_1A = pairwise_align(seq_A_test, seq_A_train)
        score_1B = pairwise_align(seq_B_test, seq_B_train)
        score_2A = pairwise_align(seq_A_test, seq_B_train)
        score_2B = pairwise_align(seq_B_test, seq_A_train)
        
        # Calculate e-values based on the alignment scores
        evalue_1A = calc_e_val(seq_A_test, seq_A_train, score_1A)
        evalue_1B = calc_e_val(seq_B_test, seq_B_train, score_1B)
        evalue_2A = calc_e_val(seq_A_test, seq_B_train, score_2A)
        evalue_2B = calc_e_val(seq_B_test, seq_A_train, score_2B)
        
        # Check both conditions
        if ((evalue_1A is not None and evalue_1A < 0.05) and (evalue_1B is not None and evalue_1B < 0.05)) or \
           ((evalue_2A is not None and evalue_2A < 0.05) and (evalue_2B is not None and evalue_2B < 0.05)):
            df_test.at[idx_test, 'similarity_check'] = 1
            similarity_found = True
            break
    
    if not similarity_found:
        df_test.at[idx_test, 'similarity_check'] = 0
    
    # Calculate elapsed time and print log for the interactions checked
    elapsed_time = time.time() - start_time
    print(f"At time t{idx_test + 1}, checked pair from df_test against pairs from df_train. Elapsed time: {elapsed_time:.2f} seconds")

print(df_test)

df_test.to_csv('test_ds_similarity_check.csv', index=False)
print(df_test['similarity_check'].value_counts())