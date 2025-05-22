from typing import List, Optional

import torch
import torch.nn as nn
# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)

# class GloveTextEmbedding:
#     def __init__(self, device: Optional[torch.device] = None):
#         # Get all available GPUs
#         if device is None:
#             self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
#             if not self.devices:  # No GPUs available, use CPU
#                 self.devices = [torch.device("cpu")]
#         else:
#             self.devices = [device]

#         # Create a model instance for each GPU
#         self.models = [SentenceTransformer(
#             "sentence-transformers/average_word_embeddings_glove.6B.300d",
#             device=dev,
#         ) for dev in self.devices]

#     def __call__(self, sentences: List[str]) -> torch.Tensor:
#         # Split sentences across GPUs
#         chunk_size = len(sentences) // len(self.devices)
#         remainder = len(sentences) % len(self.devices)
#         sentence_chunks = []
#         start = 0
#         for i in range(len(self.devices)):
#             end = start + chunk_size + (1 if i < remainder else 0)
#             sentence_chunks.append(sentences[start:end])
#             start = end

#         # Encode sentences in parallel
#         embeddings = []
#         for i, chunk in enumerate(sentence_chunks):
#             with torch.no_grad():
#                 emb = self.models[i].encode(chunk, convert_to_tensor=True)
#                 embeddings.append(emb.to(self.devices[0])) 

#         # Concatenate embeddings from all GPUs
#         return torch.cat(embeddings, dim=0)
