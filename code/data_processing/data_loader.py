# data_processing/data_loader.py
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """
        Read a tab-separated file and return a DataFrame with GOID and description columns.
        """
        df = pd.read_csv(file_path, delimiter="\t")
        print("Length of the DataFrame:", len(df))
        return df[['GOID', 'DESCRIPTION']] # df[['GOID', 'description']] for 2024


    def load_embeddings(embedding_file: str):
        data = np.load(embedding_file, allow_pickle=True)
        if 'ID' in data:
            ids = data['ID']
        elif 'GOID' in data:
            ids = data['GOID']
        else:
            raise KeyError("NPZ file must contain either 'ID' or 'GOID' as key for the node identifiers.")
        
        if 'embeddings' in data:
            embeddings = data['embeddings']
        elif 'emb' in data:
            embeddings = data['emb']
        else:
            raise KeyError("NPZ file must contain 'embeddings' or 'emb' as key for the node embeddings.")
        
        return ids, embeddings
    

    def load_node2vec_embeddings(embedding_file: str):
        """
        Load a trained Node2Vec Word2Vec model and return node IDs and their embeddings.

        Parameters:
            go_category (str): One of 'biological_process', 'molecular_function', 'cellular_component'.
            models_dir (str): Directory path where the model is saved.

        Returns:
            Tuple[List[str], np.ndarray]: (node_ids, embeddings)
        """
        # Load model
        model = Word2Vec.load(embedding_file)
        print("Embedding size:", model.vector_size)

        # Extract node IDs and embeddings
        node_ids = list(model.wv.index_to_key)
        embeddings = np.array([model.wv[node_id] for node_id in node_ids])

        return node_ids, embeddings
    


    @staticmethod
    def load_gs_ids(file_path):
        """
        Load a list of GS IDs from an external file.
        Each line of the file should contain one GS ID.
        """
        with open(file_path, "r") as f:
            gs_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(gs_ids)} GS IDs from {file_path}")
        return gs_ids

