from typing import List
import os
import dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import unicodedata
from milvus_model.hybrid import BGEM3EmbeddingFunction
os.environ["GROQ_API_KEY"] = "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"

from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)



class MilvusRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """
    
    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""
    
    def init(self):
        global ef, col 
        if len(self.documents) == 0:
            ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
            dense_dim = ef.dim["dense"]
            col = self.create_or_load_db(dense_dim)
        else:
            normalized_texts, texts = self.transform_texts(self.documents)
            ef, dense_dim, docs_embeddings = self.embed_texts(normalized_texts)
            col = self.create_or_load_db(dense_dim)
            self.fill_db(col, texts, normalized_texts, docs_embeddings)
            
                     
    def normalize_and_remove_accents(self, text):
        # Normalize text to NFD (decomposed form)
        normalized_text = unicodedata.normalize('NFD', text)
        
        # Remove accents (combining marks)
        text_without_accents = ''.join(
            char for char in normalized_text if unicodedata.category(char) != 'Mn'
        )
        
        # Convert to lowercase
        return text_without_accents.lower()
    
        
    def transform_texts(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
        texts = text_splitter.split_documents(documents)
        normalized_texts = [self.normalize_and_remove_accents(texts[i].page_content) for i in range(len(texts))]
        texts = [texts[i].page_content for i in range(len(texts))]
        return normalized_texts, texts
    
        
    def embed_texts(self,normalized_texts):

        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        dense_dim = ef.dim["dense"]

        # Generate embeddings using BGE-M3 model
        docs_embeddings = ef(normalized_texts)
        return ef, dense_dim ,docs_embeddings
     
        
    def create_or_load_db(self, dense_dim):
        # Connect to Milvus
        connections.connect(uri="./embeddings.db")

        # Define the schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        ]
        schema = CollectionSchema(fields)
        col_name = "hybrid_rag"
        
        # Check if the collection exists
        if utility.has_collection(col_name):
            print(f"Loading existing collection: {col_name}")
            col = Collection(col_name)
        else:
            print(f"Creating new collection: {col_name}")
            col = Collection(col_name, schema, consistency_level="Strong")
        
            # Create indexes if this is a new collection
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            col.create_index("sparse_vector", sparse_index)
            dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
            col.create_index("dense_vector", dense_index)
        
        # Load the collection for search
        col.load()
        return col

    
    def fill_db(self, col, texts, normalized_texts, docs_embeddings):
        # For efficiency, we insert 50 records in each small batch
        for i in range(0, len(texts), 50):
            batched_entities = [
                normalized_texts[i : i + 50],
                docs_embeddings["sparse"][i : i + 50],
                docs_embeddings["dense"][i : i + 50],
            ]
            col.insert(batched_entities)
        col.flush()
        print("Number of entities inserted:", col.num_entities)

    def dense_search(sefl,col, query_dense_embedding, limit=3):
        search_params = {"metric_type": "IP", "params": {}}
        res = col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]


    def sparse_search(self,col, query_sparse_embedding, limit=3):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]


    def hybrid_search(
        self,
        col,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=3,
    ):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
        )[0]
        return [hit.get("text") for hit in res]
    
    def norm_embed_query(self,query):
        # Enter your search query
        normalized_query = self.normalize_and_remove_accents(query)
        # ACÁ SE PUEDE IMPLEMENTAR UN SISTEMA DE PESADO AUTOMÁTICO:
        # DEPENDIENDO DEL NÚMERO DE PALABRAS, USA MÁS EL RESULTADO DENSO (MÁS PALABRAS) O EL RESULTADO SPARSE (MENOS PALABRAS)

        # Generate embeddings for the query
        query_embeddings = ef([normalized_query])
        # print(query_embeddings)x
        return query_embeddings


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        query_embeddings = self.norm_embed_query(query)
        hybrid_results = self.hybrid_search(
            col,
            query_embeddings["dense"][0],
            query_embeddings["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=1.0,
        )
        
        return [Document(page_content=result) for result in hybrid_results]