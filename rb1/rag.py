import chromadb
from google.api_core import retry
from google.genai import types
from typing import Optional
from tqdm import tqdm
from .. import is_retriable
from ..api import Api, GeminiEmbedFunction
from ..basemodel import ChromaDBResult

# Define parent class: retrieval-augmented generation.
# - ChromaDB for storage and retrieval
# - Gemini 2.0/2.5 for augmented generation
class RetrievalAugmentedGeneration:
    chroma_client = chromadb.PersistentClient(path="vector_db")
    config_temp = types.GenerateContentConfig(temperature=0.0)

    def __init__(self, api: Api, collection_name, similarity_task: bool = False):
        self.api = api
        self.client = api.args.CLIENT
        self.embed_fn = GeminiEmbedFunction(api, semantic_mode=similarity_task)
        self.db = self.chroma_client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embed_fn, # type: ignore
            metadata={"hnsw:space": "cosine"})

    @retry.Retry(
        predicate=is_retriable,
        initial=2.0,
        maximum=64.0,
        multiplier=2.0,
        timeout=600,
    )
    def add_documents_list(self, docs: list):
        self.embed_fn.document_mode = True # Switch to document mode.
        ids = list(map(str, range(self.db.count(), self.db.count()+len(docs))))
        metas=[{"source": doc.metadata["source"]} for doc in docs]
        content=[doc.page_content for doc in docs]
        progress = tqdm(total=1, desc="Generate document embedding")
        self.db.add(ids=ids, documents=content, metadatas=metas)
        progress.update(1)

    @retry.Retry(
        predicate=is_retriable,
        initial=2.0,
        maximum=64.0,
        multiplier=2.0,
        timeout=600,
    )
    def get_documents_list(self, query: str, max_sources: int = 5000, where: Optional[dict] = None):
        self.embed_fn.document_mode = False # Switch to query mode.
        return self.stored_result(
            self.db.query(query_texts=[query], 
                          n_results=max_sources, 
                          where=where), 
            is_query = True)

    @retry.Retry(
        predicate=is_retriable,
        initial=2.0,
        maximum=64.0,
        multiplier=2.0,
        timeout=600,
    )
    def generate_answer(self, query: str, max_sources: int = 10, 
                        where: Optional[dict] = None, passages: Optional[list[str]] = None):
        stored = self.get_documents_list(query, max_sources, where)
        query_oneline = query.replace("\n", " ")
        prompt = f"""You're an expert writer. You understand how to interpret html and markdown. You will accept the
        question below and answer based only on the passages. Never mention the passages in your answers. Be sure to 
        respond in concise sentences. Include all relevant background information when possible. If a passage is not 
        relevant to the answer you must ignore it. If no passage answers the question respond with: {Api.Const.Stop()}

        QUESTION: {query_oneline}
        
        """
        # Add the retrieved documents to the prompt.
        stored_docs = [passage.docs for passage in stored]
        for passage in stored_docs if passages is None else stored_docs + passages:
            passage_oneline = passage.replace("\n", " ")
            prompt += f"PASSAGE: {passage_oneline}\n"
        # Generate the response.
        response = self.api.retriable(
            self.client.models.generate_content,
            model=self.api(Api.Model.GEN),
            config=self.config_temp,
            contents=prompt)
        return response

    def stored_result(self, result, is_query: bool = False) -> list[ChromaDBResult]:
        try:
            results = []
            if len(result["documents"]) == 0:
                return results
            if isinstance(result["documents"][0], list):
                for i in range(len(result["documents"][0])):
                    obj = ChromaDBResult(
                        docs=result["documents"][0][i],
                        dist=result["distances"][0][i] if is_query else None,
                        meta=result["metadatas"][0][i],
                        store_id=result["ids"][0][i])
                    results.append(obj)
            else:
                results.append(ChromaDBResult(
                    docs=result["documents"][0],
                    dist=result["distances"][0] if is_query else None,
                    meta=result["metadatas"][0],
                    store_id=result["ids"][0]))
            return results
        except Exception as e:
            raise e