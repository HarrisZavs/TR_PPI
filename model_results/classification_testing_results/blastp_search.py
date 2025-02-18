import pandas as pd
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import time

# Example initialization (to be replaced with actual data)
df_train = pd.read_csv(r"C:\Users\harry\Desktop\ppi_train_ds.csv")
df_test = pd.read_csv(r"C:\Users\harry\Desktop\ppi_test_ds.csv")


# Define a function to run BLAST and return the E-value
def get_blast_evalue(seq1, seq2):
    if seq1 is None or seq2 is None:
        return float('inf')  # Return infinity if either sequence is None
    
    # Convert sequences to strings if they are not already
    if not isinstance(seq1, str):
        seq1 = str(seq1)
    if not isinstance(seq2, str):
        seq2 = str(seq2)
    
    # Create SeqRecord objects
    record1 = SeqRecord(Seq(seq1), id="seq1", description="Test sequence 1")
    record2 = SeqRecord(Seq(seq2), id="seq2", description="Test sequence 2")
    
    # Perform BLAST search
    print(f"Running BLAST for pair: {record1.description} vs {record2.description}")
    result_handle = NCBIWWW.qblast("blastp", "nr", record1.seq)
    blast_record = NCBIXML.read(result_handle)
    
    # Extract E-value
    e_value = float('inf')
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            if hsp.expect < e_value:
                e_value = hsp.expect
    return e_value

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
        
        # Check both conditions
        evalue_1A = get_blast_evalue(seq_A_test, seq_A_train)
        evalue_1B = get_blast_evalue(seq_B_test, seq_B_train)
        evalue_2A = get_blast_evalue(seq_A_test, seq_B_train)
        evalue_2B = get_blast_evalue(seq_B_test, seq_A_train)
        
        if (evalue_1A < 0.05 and evalue_1B < 0.05) or (evalue_2A < 0.05 and evalue_2B < 0.05):
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