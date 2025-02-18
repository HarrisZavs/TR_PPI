# import pandas as pd

# # Load your dataset
# dataset = pd.read_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS.csv")  # Assuming it has columns: uidA, uidB
# map_file = pd.read_csv(r"C:\Users\harry\Downloads\uniprotkb_toensmbl.csv")  # Assuming it has columns: Entry, Sequence

# # Rename columns in the mapping file for merging
# map_file = map_file.rename(columns={"Entry": "uidA", "Sequence": "seq_A"})
# dataset = dataset.merge(map_file, on="uidA", how="left")

# map_file = map_file.rename(columns={"uidA": "uidB", "seq_A": "seq_B"})
# dataset = dataset.merge(map_file, on="uidB", how="left")

# # Save the updated dataset
# dataset.to_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS_MAPPED.csv", index=False)

# print(dataset.head())  # Preview the result


# import pandas as pd
# from unipressed import UniprotkbClient

# # Load dataset
# df = pd.read_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS_MAPPED.csv")  # Ensure dataset has 'uidA', 'uidB', 'seq_A', 'seq_B'

# def get_primary_uniprot(uid):
#     """Fetches primary UniProt accession if uid is secondary."""
#     try:
#         response = UniprotkbClient.fetch_one(uid)
#         return response.get("primaryAccession", uid)  # Return primary if exists, else original UID
#     except:
#         return None  # If UID is invalid

# def get_sequence(uid):
#     """Fetches protein sequence from UniProt."""
#     try:
#         response = UniprotkbClient.fetch_one(uid)
#         return response.get("sequence", {}).get("value", None)
#     except:
#         return None

# # Fill missing seq_A
# for index, row in df.iterrows():
#     if pd.isna(row["seq_A"]):  # If sequence missing
#         primary_uid = get_primary_uniprot(row["uidA"])
#         if primary_uid:
#             df.at[index, "seq_A"] = get_sequence(primary_uid)

# # Fill missing seq_B
# for index, row in df.iterrows():
#     if pd.isna(row["seq_B"]):  # If sequence missing
#         primary_uid = get_primary_uniprot(row["uidB"])
#         if primary_uid:
#             df.at[index, "seq_B"] = get_sequence(primary_uid)

# # Save the updated dataset
# df.to_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS_MAPPED_2.csv", index=False)

# print("Dataset updated with primary sequences!")

import pandas as pd
from unipressed import UniprotkbClient

# Load dataset
df = pd.read_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS_MAPPED_2.csv")  # Ensure dataset has 'uidA', 'uidB', 'seq_A', 'seq_B'

def fetch_sequence(uid):
    """Fetches the protein sequence for a given UniProt UID."""
    try:
        response = UniprotkbClient.fetch_one(uid)
        return response.get("sequence", {}).get("value", None)  # Extracts the sequence if available
    except:
        return None  # Return None if fetching fails

# Fill missing seq_A
for index, row in df.iterrows():
    if pd.isna(row["seq_A"]):  # If sequence is missing
        sequence = fetch_sequence(row["uidA"])
        if sequence:
            df.at[index, "seq_A"] = sequence

# Fill missing seq_B
for index, row in df.iterrows():
    if pd.isna(row["seq_B"]):  # If sequence is missing
        sequence = fetch_sequence(row["uidB"])
        if sequence:
            df.at[index, "seq_B"] = sequence

# Save the updated dataset
df.to_csv(r"C:\Users\harry\Desktop\TEST_1\FINAL_TEST_DS_MAPPED_3.csv", index=False)

print("Dataset updated with any available sequences!")
