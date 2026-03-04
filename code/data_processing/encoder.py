# data_processing/encoder.py

class Encoder:
    def __init__(self, model_name='default', use_gpu=True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        if self.model_name == 'default':
            from sentence_transformers import SentenceTransformer
            device = "cuda" if use_gpu else "cpu"
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def encode(self, descriptions):
        if self.model_name == 'default':
            embeddings = self.model.encode(descriptions, show_progress_bar=True)
        elif self.model_name == 'openai':
            import os
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
            embeddings = []
            def get_embedding(text, model="text-embedding-3-small"):
                text = text.replace("\n", " ")
                return client.embeddings.create(input = [text], model=model).data[0].embedding

            for desc in descriptions:
                # Extract the embedding vector from the first (and only) element in the response's data list
                embedding = get_embedding(desc)
                embeddings.append(embedding)

        else:
            raise ValueError("Unknown model option. Use 'default' or 'openai'.")
        return embeddings

