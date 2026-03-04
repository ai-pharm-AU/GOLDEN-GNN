# data_processing/data_processor.py
import os
import pandas as pd
import numpy as np
from graph.graph_operations import GraphOperations

META_DATA_PATH = 'GO2025/GO_info_2025.txt'

class DataProcessor:
    """Methods for reading, cleaning, and processing data and embeddings."""

    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, delimiter="\t")

    @staticmethod
    def check_data():
        df = DataProcessor.read_csv(META_DATA_PATH)
        print(df.head())
        G = GraphOperations.load_graph('subgraph_GO_0006811.graphml')
        children = list(G.successors('GO:0034220'))
        print("Children of GO:0034220:", children)
        print("Total nodes:", len(G.nodes()))
        found = [name for name in G.nodes() if df['NAME'].str.contains(name).any()]
        print("Found nodes:", found)
        with open('found_nodes_list.txt', 'w') as f:
            f.write(','.join(found))

    @staticmethod
    def clean_data():
        df = DataProcessor.read_csv(META_DATA_PATH)
        print(df.head())
        for part in ['biological_process', 'molecular_function', 'cellular_component']:
            filtered = df[df['name'].str.contains(part)]
            valid = set(filtered['name'].apply(lambda x: x.split(' ')[0]).tolist())
            GraphOperations.table_to_verified_graph('GO2025/topdown_2025.txt', valid, part)
            
    @staticmethod
    def gen_ground_truth_labels(graph_path: str, roots: list):
        G = GraphOperations.load_graph(graph_path)
        labels = {}
        nodes_set = set()
        for label, root in enumerate(roots):
            labels[root] = label
            nodes_set.add(root)
            for neighbor in G.neighbors(root):
                if neighbor not in labels:
                    labels[neighbor] = label
                    nodes_set.add(neighbor)
        sorted_nodes = sorted(nodes_set)
        gt_labels = [labels.get(node, -1) for node in sorted_nodes]
        return sorted_nodes, gt_labels, labels

    @staticmethod
    def find_ground_truth_set(valid_nodes: list):
        df = DataProcessor.read_csv(META_DATA_PATH)
        pattern = '|'.join(valid_nodes)
        matching = df[df['NAME'].str.contains(pattern, case=False, na=False)]
        return matching, matching['DESCRIPTION'].tolist()

    @staticmethod
    def get_normalized_embeddings(df: pd.DataFrame, opt: str = 'desc') -> np.ndarray:
        df['NAMEDESC'] = df['NAME'].apply(lambda x: ' '.join(x.split(' ')[2:]) if len(x.split(' ')) > 2 else 'none')
        if opt == 'name':
            df['EMBEDDING'] = df['NAMEDESC'].apply(lambda x: DataProcessor.get_embedding(x))
        elif opt == 'desc':
            df['EMBEDDING'] = df['DESCRIPTION'].apply(lambda x: DataProcessor.get_embedding(x))
        elif opt == 'name+desc':
            df['EMBEDDING'] = (df['NAMEDESC'] + ' ' + df['DESCRIPTION']).apply(lambda x: DataProcessor.get_embedding(x))
        else:
            raise ValueError(f"Invalid embedding option: {opt}")
        embeddings = np.stack(df['EMBEDDING'].values)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    @staticmethod
    def get_embedding(text: str) -> np.ndarray:
        # It is assumed that you install the sentence transformers package.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text)

