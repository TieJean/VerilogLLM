import faiss
'''
Script to add an example to the prompt (few-shot learning) for the VerilogDescription_Human dataset.
The script uses Sentence Transformers to generate embeddings for the descriptions in the dataset.
The embeddings are then used to build a FAISS index for vector search.
The user query is used to search for similar descriptions in the dataset.
The most similar description is then used to generate a new prompt.
The new prompt is saved to an output file that can be used for evaluation of LLMs for verilog code generation.

Author: Pawan Kumar Rukmangada
email: pawankumar.urs@utexas.edu
F24, UT Austin, Towards Applied Machine learning course project. 
'''
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import json
import os
import ast


# Load dataset
def load_verilog_dataset(reference_dataset:str):
    dataset = load_dataset(reference_dataset)
    return dataset['train'] 

#init csv file 
def initialize_csv(force_gen=False):
    if not os.path.exists(embedding_file_path):
        print("CSV file does not exist. Creating a new one...")
        df = pd.DataFrame(columns=["description", "embedding", "code"])
        df.to_csv(embedding_file_path, index=False)
    elif force_gen:
        print("CSV file found, but relpacing it...")
        df = pd.DataFrame(columns=["description", "embedding", "code"])
        df.to_csv(embedding_file_path, index=False)
    else:
        print("CSV file found")

#Save embeddings
def save_embeddings_to_csv(embedding_file_path, descriptions, embeddings, codes):
    df = pd.DataFrame(columns=["description", "embedding", "codes"], index=range(0,len(codes)))
    for i in range(0,len(codes)):
        df.loc[i] = [descriptions[i], embeddings[i].tolist(), codes[i]]
    df.to_csv(embedding_file_path, index=False)
#Load embeddings
def load_embeddings_from_csv():
    df = pd.read_csv(embedding_file_path, converters={'embedding': ast.literal_eval})
    #df = pd.read_csv(embedding_file_path)
    descriptions = df["description"].tolist()
    codes = df["codes"].tolist()
    embeddings = df["embedding"].tolist()
    return descriptions, np.array(embeddings), codes

#Build vector search index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

#Perform vector search
def search_similar_descriptions(query, model, index, descriptions, codes, k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results=[]
    for index, idx in enumerate(indices[0]):
        results.append((descriptions[idx], codes[idx], distances[0][index]))
    #results = [(descriptions[idx], codes[idx]) for idx in indices[0]]
    return results
def gen_embeddings(descriptions, codes, embedding_model):
    new_descriptions = []
    new_codes = []
    for desc, code in zip(descriptions, codes):
        new_descriptions.append(desc["high_level_global_summary"].split("unless otherwise stated.")[-1])
        new_codes.append(code)

    embeddings = embedding_model.encode(new_descriptions, convert_to_numpy=True)
    return new_descriptions, embeddings, new_codes
    
# Main Function
def add_example_to_prompt(reference_dataset:str, 
                          embedding_model_name:str,
                          description_files:list, 
                          embedding_file_path:str, 
                          output_file:str,
                          force_gen:bool=False,
                          max_embeddings:int=None):
    # Load dataset
    print("Loading dataset...")
    dataset = load_verilog_dataset(reference_dataset)

    # Initialize embedding model
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Initialize CSV file
    print("Initializing CSV file...")
    initialize_csv(force_gen=force_gen)

    # Create embeddings
    print("Creating embeddings...")
    descriptions = dataset["description"][:max_embeddings] if max_embeddings else dataset["description"]
    codes = dataset["code"][:max_embeddings] if max_embeddings else dataset["code"]  # Assuming the dataset has a 'code' field

    new_descriptions = []
    new_codes = []
    if force_gen==True: #force_gen check is redundant
        new_descriptions, embeddings, new_codes=gen_embeddings(descriptions, codes, embedding_model)
        # Save new embeddings to CSV
        save_embeddings_to_csv(embedding_file_path, new_descriptions, embeddings, new_codes)
    else:
        # Load existing embeddings
        existing_descriptions, existing_embeddings, existing_codes = load_embeddings_from_csv()
        new_descriptions = existing_descriptions
        embeddings = existing_embeddings
        new_codes = existing_codes

    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # User query

    for file in description_files:
        with open(file, 'r') as json_in, open(output_file, 'w') as json_out:
            for line in json_in:
                data=json.loads(line)
                user_query = data["detail_description"]

                # Search similar descriptions
                print("Searching for similar descriptions...")
                similar_descriptions = search_similar_descriptions(user_query, embedding_model, index, new_descriptions, new_codes, k=1)

                print("Similar descriptions found:")
                for desc, code, dist in similar_descriptions:
                    print(f"Description: {desc}, Distance: {dist}")

                # Generate Verilog code
                print("Generating new prompt...")
                example_description = similar_descriptions[0][0]
                example_code = similar_descriptions[0][1]
                data["detail_description"] = user_query + f" Use the following description and code as an example:\nDescription: {example_description}\nAssociated Code: {example_code}"
                json_out.write(json.dumps(data) + "\n")
                print("New prompt generated!")

if __name__ == "__main__":
    FORCE_GEN = False # Controls if the embeddings are generated even if file exists.
    description_files = ["evaluation_data_oneexample/VerilogDescription_Human.jsonl"]
    embedding_file_path = "evaluation_data_oneexample/embeddings.csv"
    add_example_to_prompt(reference_dataset= "GaTech-EIC/MG-Verilog", 
                          embedding_model_name="all-MiniLM-L6-v2",
                          description_files=description_files, 
                          embedding_file_path=embedding_file_path,
                          output_file="evaluation_data_oneexample/fewshot_VerilogDescription_Human.jsonl",
                          force_gen=FORCE_GEN)
