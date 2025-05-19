
# import requests
# import pandas as pd
# from itertools import combinations

# def get_string_interactions(uniprot_pairs):
#     # Base URL for STRING API
#     base_url = "https://string-db.org/api"
#     output_format = "tsv"
#     method = "network"
#     interactions = []

#     for uidA, uidB in uniprot_pairs:
#         # Parameters for the request
#         proteins = f"{uidA}%0d{uidB}"
#         params = {
#             "identifiers": proteins,
#             "species": 9606,  # Homo sapiens (human)
#             "caller_identity": "your_email@example.com"  # STRING API requires an email address for the caller
#         }

#         # Make the API call
#         response = requests.get(f"{base_url}/{output_format}/{method}", params=params)
#         response.raise_for_status()  # Check if the request was successful

#         # Parse the response
#         data = response.text.splitlines()
#         header = data[0].split("\t")
#         for line in data[1:]:  # Skip header
#             columns = line.split("\t")
#             interaction_data = {header[i]: columns[i] for i in range(len(header))}
#             interaction_data["uidA"] = uidA
#             interaction_data["uidB"] = uidB
#             interactions.append(interaction_data)

#     return pd.DataFrame(interactions)

# # Example usage
# uniprot_ids = ["P31749", "P45983", "Q9Y243"]  # Replace with your list of UniProt IDs
# uniprot_pairs = list(combinations(uniprot_ids, 2))  # Generate all possible pairs
# interactions_df = get_string_interactions(uniprot_pairs)

# # Save the interactions to a CSV file
# interactions_df.to_csv("interactions.csv", index=False)

# print(interactions_df)
# import requests
# import pandas as pd
# import time

# def get_string_interaction(uidA, uidB):
#     # Base URL for STRING API
#     base_url = "https://string-db.org/api"
#     output_format = "tsv"
#     method = "network"

#     # Parameters for the request
#     proteins = f"{uidA}%0A{uidB}"
#     params = {
#         "identifiers": proteins,
#         "species": 9606,  # Homo sapiens (human)
#         "caller_identity": "your_email@example.com"  # STRING API requires an email address for the caller
#     }

#     # Make the API call
#     response = requests.get(f"{base_url}/{output_format}/{method}", params=params)
#     response.raise_for_status()  # Check if the request was successful

#     # Parse the response
#     data = response.text.splitlines()
#     if len(data) > 1:  # If there's more than just the header line, an interaction exists
#         columns = data[1].split("\t")
#         score = float(columns[5])
#         stringidA= columns[0]
#         stringidB= columns[1]
#         nscore= columns[6]
#         fscore= columns[7]
#         pscore=columns[8]
#         ascore=columns[9]
#         escore=columns[10]
#         dscore= columns[11]
#         tscore= columns[12]  # The 'score' column is at index 5
#         interaction_data = {
#             "uidA": uidA,
#             "uidB": uidB,
#             "stringdb_score": score,
#             "stringdb_check": 1 if score > 0 else 0,
#             'stringidA':stringidA,
#             "stringidB": stringidB,
#             "nscore": nscore,
#             "fscore":fscore,
#             "pscore":pscore,
#             "ascore":ascore,
#             "escore":escore,
#             "dscore":dscore,
#             "tscore":tscore
#         }
#     else:
#         # If there's no interaction, set score to 0 and stringdb_check to 0
#         interaction_data = {
#             "uidA": uidA,
#             "uidB": uidB,
#             "stringdb_score": 0,
#             "stringdb_check": 0,
#             'stringidA':0,
#             "stringidB": 0,
#             "nscore": 0,
#             "fscore":0,
#             "pscore":0,
#             "ascore":0,
#             "escore":0,
#             "dscore":0,
#             "tscore":0
#         }

#     return interaction_data

# # Read the original dataset with PPIs
# original_df = pd.read_csv(r"C:\Users\harry\Desktop\TR_interactions.csv" ) # Replace with your file path
# original_df['uidA']= original_df['uidA'].str.split('-').str[0].str.strip()
# original_df['uidB']= original_df['uidB'].str.split('-').str[0].str.strip()

# # Split the original dataset into batches of 100
# batch_size = 161
# batches = [original_df[i:i + batch_size] for i in range(0, len(original_df), batch_size)]

# # Initialize a list to store the interaction data
# interaction_results = []

# # Process each batch
# for batch in batches:
#     batch_results = []
#     for idx, row in batch.iterrows():
#         uidA = row['uidA']
#         uidB = row['uidB']
#         interaction_data = get_string_interaction(uidA, uidB)
#         batch_results.append(interaction_data)
#         time.sleep(1)  # To avoid overloading the API, include a delay between requests
#     interaction_results.extend(batch_results)

# # Convert the results into a DataFrame
# interactions_df = pd.DataFrame(interaction_results)

# # Merge scores back into the original dataset
# merged_df = original_df.merge(interactions_df, on=['uidA', 'uidB'], how='left')

# # Save the merged DataFrame to a CSV file
# merged_df.to_csv("c:\\users\\harry\\desktop\\TR_interactions_stringdb_check_2.csv", index=False)
# print(merged_df['stringdb_check'].value_counts())
# print(merged_df)

# import pandas as pd

# # Load the data
#  # Replace with the actual file path
# df = pd.read_csv(r"C:\Users\harry\Desktop\TR_interactions.csv")

# # Extract uidA where taste1 is not empty
# uidA_list = df['uidA'][df['taste1'].notna()]

# # Extract uidB where taste2 is not empty
# uidB_list = df['uidB'][df['taste2'].notna()]
# uidA_list.to_csv('c:\\users\\harry\\desktop\\uidA.csv', index=False)
# uidB_list.to_csv('c:\\users\\harry\\desktop\\uidB.csv', index=False)
# # Convert the results to lists
# uidA_list = uidA_list.tolist()
# uidB_list = uidB_list.tolist()

# print("uidA list based on taste1:")
# print(uidA_list)
# print("\nuidB list based on taste2:")
# print(uidB_list)



# # Read the CSV file into a DataFrame
# df_mapping = pd.read_csv(r"C:\Users\harry\Desktop\TR_interactions.csv")
# # Create a dictionary mapping UniProt IDs (uidA and uidB) to Gene names (Gene1 and Gene2)
# uniport_to_gene_dict = {}

# for idx, row in df_mapping.iterrows():
#     uniport_to_gene_dict[row['uidA']] = row['Gene1']
#     uniport_to_gene_dict[row['uidB']] = row['Gene2']

# # Create a DataFrame with two columns: uid and ID (where ID is the corresponding Gene name)
# uid_id_df = pd.DataFrame(list(uniport_to_gene_dict.items()), columns=['uid', 'ID'])

# # Print or use uid_id_df as needed
# print(uid_id_df)

# interactions_df = interactions_df.merge(uid_id_df[['uid', 'ID']], left_on='preferredName_A', right_on='ID', how='left').rename(columns={'uid': 'uidA'}).drop(columns=['ID'])
# interactions_df = interactions_df.merge(uid_id_df[['uid', 'ID']], left_on='preferredName_B', right_on='ID', how='left').rename(columns={'uid': 'uidB'}).drop(columns=['ID'])


# df3 = pd.read_csv(r"C:\Users\harry\Desktop\projects\irefindex_curated_uniprot_only.csv")
# df3['uidA'] = df3['uidA'].str.strip()
# df3['uidB'] = df3['uidB'].str.strip()

# # Step 3: Create a dictionary for fast lookup
# gene_pair_dict = {}

# for _, row in df3.iterrows():
#     uidA = row['uidA']
#     uidB = row['uidB']
#     gene_pair_dict[(uidA, uidB)] = True
#     gene_pair_dict[(uidB, uidA)] = True

# # Step 4: Check interactions against the dictionary
# interactions_df['irefindex_check'] = interactions_df.apply(lambda row: (
#     (row['uidA'], row['uidB']) in gene_pair_dict or
#     (row['uidB'], row['uidA']) in gene_pair_dict
# ), axis=1)

# print(interactions_df)

import requests
import pandas as pd
import time

def check_string_interactions_chunk(df_chunk, species=9606, min_score=400, delay=1):
    """
    Check if uidA-uidB interactions exist in STRING database for a chunk of the dataframe.

    Args:
        df_chunk (pd.DataFrame): Chunk of the original dataframe containing 'uidA' and 'uidB' columns.
        species (int): NCBI taxonomy identifier for the species (default is 9606 for Homo sapiens).
        min_score (int): Minimum confidence score for interactions (default is 400).
        delay (float): Time in seconds to wait between requests.

    Returns:
        pd.DataFrame: Chunk dataframe with 'stringdb_check' and 'stringdb_score' columns added.
    """
    base_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "interaction_partners"
    
    # Initialize columns for the results
    df_chunk['stringdb_check'] = 0
    df_chunk['stringdb_score'] = 0.0
    
    # Iterate over each pair in the chunk
    for index, row in df_chunk.iterrows():
        uidA = row['uidA']
        uidB = row['uidB']
        
        # Ensure proper formatting of the UID for the request
        uidA_formatted = uidA.split('-')[0]
        
        url = f"{base_url}/{output_format}/{method}?identifiers={uidA_formatted}&species={species}&required_score={min_score}"
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                for line in lines:
                    data = line.split('\t')
                    if len(data) > 5 and data[1] == uidB:
                        df_chunk.at[index, 'stringdb_check'] = 1
                        df_chunk.at[index, 'stringdb_score'] = float(data[5])
                        break
            else:
                print(f"Failed to retrieve data for {uidA}. HTTP status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data for {uidA}: {e}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(delay)
    
    return df_chunk

def check_string_interactions(df, species=9606, min_score=400, chunk_size=100, delay=1):
    """
    Check if uidA-uidB interactions exist in STRING database and append the result to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing 'uidA' and 'uidB' columns.
        species (int): NCBI taxonomy identifier for the species (default is 9606 for Homo sapiens).
        min_score (int): Minimum confidence score for interactions (default is 400).
        chunk_size (int): Number of samples per chunk for processing.
        delay (float): Time in seconds to wait between requests.

    Returns:
        pd.DataFrame: DataFrame with 'stringdb_check' and 'stringdb_score' columns added.
    """
    # Split the dataframe into chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process each chunk and collect results
    result_chunks = []
    for chunk in chunks:
        result_chunk = check_string_interactions_chunk(chunk, species, min_score, delay)
        result_chunks.append(result_chunk)
    
    # Concatenate all result chunks back into a single dataframe
    result_df = pd.concat(result_chunks, ignore_index=True)
    
    return result_df

# Load the dataset
df = pd.read_csv(r".\example_datasets\PPI_TESTING_DATASET\FINAL_TEST_DS.csv")

# Check interactions
result_df = check_string_interactions(df)
print(result_df)

# Save the result to a CSV file
result_df.to_csv('ppi_test_stringdb.csv', index=False)
