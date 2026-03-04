# data_processing/embedding_retriever.py
import numpy as np
from data_processing.data_loader import DataLoader

class EmbeddingRetriever:   
    @staticmethod
    def get_embeddings_for_goids(npz_file_path, query_goids):
        data = np.load(npz_file_path, allow_pickle=True)
        goids = data['GOID']
        embeddings = data['embeddings']
        goid_to_embedding = {goid: emb for goid, emb in zip(goids, embeddings)}
        result = {goid: goid_to_embedding.get(goid, None) for goid in query_goids}
        return result
    
class EmbeddingMerger:

    @staticmethod
    def merge(
        npz_file1: str,
        npz_file2: str,
        gs_id_list: list,
        merge_method: str = "weighted",
        alpha: float = 0.5,
        output_file: str = "merged_embeddings.npz",
        if_save: bool = False,
        cut_dim: int = 512,
        normalize: str = "emb1",
    ) -> tuple:
        """
        Merge embeddings from two NPZ files using either a weighted sum or concatenation,
        given a text file of GS IDs. Only GS IDs that appear in both NPZ files and in the
        provided file will be merged.
        
        Parameters:
            npz_file1 (str): Path to the first NPZ file containing 'ID' (or 'GOID') and 'embeddings'.
            npz_file2 (str): Path to the second NPZ file containing 'ID' (or 'GOID') and 'embeddings'.
            gs_file (str): Path to a text file containing GS IDs (one per line).
            merge_method (str): "weighted" to compute 
                                merged_embedding = alpha * emb1 + (1 - alpha) * emb2,
                                "concatenate" to concatenate the embeddings along axis 1.
            alpha (float): Weight for the first embedding (0 <= alpha <= 1; used only if merge_method=="weighted").
            output_file (str): Path to save the merged embeddings NPZ file.
        
        Returns:
            tuple: (merged_ids, merged_embeddings)
                   merged_ids is a sorted list of GS IDs present in both NPZ files and in the gs file,
                   merged_embeddings is a NumPy array of the merged embeddings.
        """
        def normalize_l2(x):
            x = np.array(x)
            if x.ndim == 1:
                norm = np.linalg.norm(x)
                return x if norm == 0 else x / norm
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)
        # Load GS IDs from the text file.
        # gs_id_list = DataLoader.load_gs_ids(gs_file)
        
        # Load embeddings from file1 (NPZ expected).
        ids1, emb1 = DataLoader.load_embeddings(npz_file1)

        # Load embeddings from file2.
        # - If it's an .npz, load as (ID/GOID, embeddings/emb)
        # - Otherwise treat it as a saved Word2Vec model (Node2Vec)
        if str(npz_file2).lower().endswith(".npz"):
            ids2, emb2 = DataLoader.load_embeddings(npz_file2)
        else:
            ids2, emb2 = DataLoader.load_node2vec_embeddings(npz_file2)

        # Ensure IDs are lists.
        ids1 = list(ids1)
        ids2 = list(ids2)
        
        # Find common GS IDs among the two NPZ files and the provided list.
        common_ids = sorted(set(ids1) & set(ids2) & set(gs_id_list))
        if not common_ids:
            raise ValueError("No common GS IDs found between the files and the provided GS file.")
        
        # Warn about missing IDs.
        missing_in_file1 = set(gs_id_list) - set(ids1)
        missing_in_file2 = set(gs_id_list) - set(ids2)
        if missing_in_file1:
            print(f"Warning: The following GS IDs are not found in file1: {missing_in_file1}")
        if missing_in_file2:
            print(f"Warning: The following GS IDs are not found in file2: {missing_in_file2}")
        
        # Align embeddings based on common IDs.
        idx1 = [ids1.index(gs) for gs in common_ids]
        idx2 = [ids2.index(gs) for gs in common_ids]
        emb1_aligned = emb1[idx1]
        emb2_aligned = emb2[idx2]

        emb1_reduced = emb1_aligned[:, :cut_dim]
        normalize = (normalize or "none").lower()
        if normalize not in {"none", "emb1", "emb2", "both"}:
            raise ValueError("normalize must be one of: none, emb1, emb2, both")

        if normalize in {"emb1", "both"}:
            emb1_reduced = normalize_l2(emb1_reduced)
        if normalize in {"emb2", "both"}:
            emb2_aligned = normalize_l2(emb2_aligned)
        
        # Merge embeddings using the selected method.
        if merge_method == "weighted":
            if emb1_aligned.shape != emb2_aligned.shape:
                raise ValueError("For weighted merging, both embeddings must have the same shape.")
            if normalize in {"emb1", "both"}:
                emb1_aligned = normalize_l2(emb1_aligned)
            if normalize in {"emb2", "both"}:
                emb2_aligned = normalize_l2(emb2_aligned)
            merged_embeddings = alpha * emb1_aligned + (1 - alpha) * emb2_aligned
        elif merge_method == "concatenate":
            # merged_embeddings = np.concatenate((emb1_aligned, emb2_aligned), axis=1)
            # merged_embeddings = np.concatenate((alpha * emb1_aligned, (1 - alpha) * emb2_aligned), axis=1)
            merged_embeddings = np.concatenate((alpha * emb1_reduced, (1 - alpha) * emb2_aligned), axis=1)
        elif merge_method == "graph_base":
            merged_embeddings = emb2_aligned
        elif merge_method == "density_base":
            merged_embeddings = emb1_reduced
        else:
            raise ValueError("merge_method must be either 'weighted' or 'concatenate'.")
        
        # Save the merged embeddings.
        if if_save:
            np.savez(output_file, ID=common_ids, embeddings=merged_embeddings)
            print(f"Merged embeddings saved to {output_file}.npz")
        return common_ids, merged_embeddings

