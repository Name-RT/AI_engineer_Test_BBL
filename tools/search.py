"""
tools/search.py — RAG Search & Two-Stage Information Retrieval Engine
"""
import os
import math
import logging
import hashlib
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

SYNONYMS = {
    "wfh": ["remote work", "work from home", "telecommute", "ทำงานที่บ้าน"],
    "ทำงานที่บ้าน": ["remote work", "work from home", "WFH"],
    "ทำงานนอกสถานที่": ["remote work", "work from home", "WFH"],
    "ลาพักร้อน": ["annual leave", "vacation", "PTO", "วันลาสะสม"],
    "ลาพักผ่อน": ["annual leave", "vacation", "PTO", "วันลาสะสม"],
    "วันหยุด": ["holiday", "annual leave", "public holiday"],
    "ลาป่วย": ["sick leave", "medical leave", "medical PTO"],
    "เดินทางต่างประเทศ": ["international travel", "business trip", "overseas travel"],
    "ค่าเดินทาง": ["travel expense", "international travel", "per diem"],
    "เบิกเงิน": ["expense reimbursement", "claim", "receipt"],
    "เบิกค่าใช้จ่าย": ["expense reimbursement", "claim", "receipt"],
    "รหัสผ่าน": ["password", "credentials", "authentication", "IT security"],
    "ความปลอดภัย": ["security policy", "confidentiality", "data protection"],
    "vacation": ["annual leave", "PTO", "time off", "holiday"],
    "sick leave": ["medical leave", "sick time", "medical PTO"],
    "travel": ["international travel", "business trip", "overseas"],
    "expense": ["reimbursement", "claim", "receipt"],
    "password": ["credentials", "authentication", "login"],
    "fired": ["termination", "dismissal", "let go"],
}


class KnowledgeBaseSearchTool:
    """
    Two-Stage Information Retrieval & Re-ranking Engine for Knowledge Base text files.
    
    Features:
    - Paragraph-aware chunking with sliding-window overlap support.
    - Synonym expansion for common colloquialisms (e.g. WFH -> remote work).
    - Stage 1: Candidate Retrieval (High Recall):
      1. 'chroma': Persistent dense semantic vector search via ChromaDB (multilingual-e5-small).
      2. 'tfidf': Lexical TF-IDF cosine similarity using scikit-learn.
      3. 'hybrid': Dense embedding + TF-IDF with Reciprocal Rank Fusion (RRF).
    - Stage 2: Semantic Re-ranking (High Precision):
      - Cross-Encoder model (BAAI/bge-reranker-v2-m3) with full cross-attention.
      - Raw Cross-Encoder logit scoring with Level 1 noise filtering (min_chunk_score).
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the search tool, chunks the document, and sets up vector indices.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing search and reranking parameters.
        """
        self.config = config
        self.mode = config["search"]["mode"]
        self.chunk_size = config["search"].get("chunk_size", 500)
        self.chunk_overlap = config["search"].get("chunk_overlap", 50)
        self.threshold = config["search"]["similarity_threshold"]
        self.chunks = []
        self.embeddings = None
        self.embed_matrix = None
        self.chroma_client = None
        self.collection = None
        
        # Re-ranking attributes
        self.reranker = None
        self.reranking_config = config.get("reranking", {})
        self.reranking_enabled = self.reranking_config.get("enabled", False)
        
        # Load KB file (supports both relative and absolute paths)
        raw_kb_path = config["search"]["knowledge_base_path"]
        if os.path.isabs(raw_kb_path):
            kb_path = raw_kb_path
        else:
            kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), raw_kb_path)
            
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read knowledge base at {kb_path}: {e}")
            return
            
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        
        # Paragraph and overlap aware chunking
        raw_sections = [c.strip() for c in content.split("\n\n") if c.strip()]
        self.chunks = []
        step_size = max(100, self.chunk_size - self.chunk_overlap)
        
        for section in raw_sections:
            if len(section) <= self.chunk_size:
                self.chunks.append({"id": len(self.chunks), "content": section})
            else:
                # Extract header if present (e.g. === Section X: Name ===)
                header = ""
                body = section
                if section.startswith("===") and "===" in section[3:]:
                    parts = section.split("===", 2)
                    if len(parts) >= 3:
                        header = f"==={parts[1]}===\n"
                        body = parts[2].strip()
                
                # Sliding window chunking
                start = 0
                while start < len(body):
                    end = min(start + self.chunk_size, len(body))
                    chunk_text = (header + body[start:end]).strip() if header else body[start:end].strip()
                    self.chunks.append({"id": len(self.chunks), "content": chunk_text})
                    if end >= len(body):
                        break
                    start += step_size
                
        # TF-IDF Setup
        if not self.chunks:
            logger.warning("No chunks created from knowledge base.")
            return
            
        self.vectorizer = TfidfVectorizer()
        texts = [c["content"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Hybrid / Chroma Setup
        self.embeddings = None
        self.embed_matrix = None
        self.chroma_client = None
        self.collection = None
        
        if self.mode in ["hybrid", "chroma"]:
            try:
                from config.settings import get_embeddings
                self.embeddings = get_embeddings()
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                self.mode = "tfidf"  # fallback
                
        if self.mode == "hybrid" and self.embeddings is not None:
            try:
                emb_model_name = self.config.get("embedding", {}).get("model_name", "")
                is_e5 = "e5" in emb_model_name.lower()
                doc_texts = [f"passage: {t}" for t in texts] if is_e5 else texts
                self.embed_matrix = self.embeddings.embed_documents(doc_texts)
            except Exception as e:
                logger.error(f"Failed to pre-compute embed matrix for hybrid search: {e}")
                self.mode = "tfidf"
                
        elif self.mode == "chroma" and self.embeddings is not None:
            try:
                import chromadb
                persist_dir = self.config["search"].get("persist_directory", "./chroma_db")
                if not os.path.isabs(persist_dir):
                    persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                collection_name = self.config["search"].get("collection_name", "rag_bbl_policies")
                
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Check cache validity via hash file
                hash_file = os.path.join(persist_dir, f".{collection_name}_hash")
                cached_hash = ""
                if os.path.exists(hash_file):
                    try:
                        with open(hash_file, "r", encoding="utf-8") as hf:
                            cached_hash = hf.read().strip()
                    except Exception:
                        pass
                
                needs_reindex = (self.collection.count() != len(self.chunks)) or (cached_hash != content_hash)
                
                if needs_reindex and len(self.chunks) > 0:
                    logger.info(f"Re-indexing {len(self.chunks)} chunks into ChromaDB '{collection_name}' (content changed)...")
                    existing = self.collection.get()
                    if existing and existing.get("ids"):
                        self.collection.delete(ids=existing["ids"])
                        
                    ids = [f"chunk_{c['id']}" for c in self.chunks]
                    documents = [c["content"] for c in self.chunks]
                    metadatas = [{"chunk_id": c["id"]} for c in self.chunks]
                    
                    emb_model_name = self.config.get("embedding", {}).get("model_name", "")
                    is_e5 = "e5" in emb_model_name.lower()
                    doc_texts = [f"passage: {doc}" for doc in documents] if is_e5 else documents
                    embeddings_list = self.embeddings.embed_documents(doc_texts)
                    
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings_list
                    )
                    try:
                        with open(hash_file, "w", encoding="utf-8") as hf:
                            hf.write(content_hash)
                    except Exception:
                        pass
                    logger.info("ChromaDB vector ingestion successfully completed.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.mode = "tfidf"  # fallback

    def _get_reranker(self):
        """
        Lazily initializes and returns the CrossEncoder re-ranker instance.
        """
        if self.reranker is not None:
            return self.reranker
            
        model_name = self.reranking_config.get("model_name", "BAAI/bge-reranker-v2-m3")
        device = self.reranking_config.get("device", "cpu")
        try:
            logger.info(f"Loading Cross-Encoder re-ranker model: '{model_name}' on {device}...")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(model_name, device=device)
            logger.info("Cross-Encoder re-ranker successfully loaded.")
            return self.reranker
        except Exception as e:
            logger.error(f"Failed to load re-ranker model '{model_name}': {e}. Falling back to Stage 1 rankings.")
            self.reranking_enabled = False
            return None

    def _expand_synonyms(self, query: str) -> str:
        q_lower = query.lower()
        expanded_tokens = list(query.split())
        added_synonyms = []
        for k, v in SYNONYMS.items():
            if k.lower() in q_lower:
                for syn in v:
                    if syn not in expanded_tokens and syn not in added_synonyms:
                        added_synonyms.append(syn)
        if added_synonyms:
            return query + " " + " ".join(added_synonyms)
        return query

    def _stage1_retrieve(self, expanded_query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Executes Stage 1 Candidate Retrieval (High Recall).
        """
        if self.mode == "tfidf":
            query_vec = self.vectorizer.transform([expanded_query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            results = []
            for i, score in enumerate(scores):
                if score >= self.threshold:
                    results.append({"chunk_id": self.chunks[i]["id"], "content": self.chunks[i]["content"], "score": float(score)})
                    
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        elif self.mode == "hybrid" and self.embed_matrix is not None:
            q_vec_tfidf = self.vectorizer.transform([expanded_query])
            tfidf_scores = cosine_similarity(q_vec_tfidf, self.tfidf_matrix).flatten()
            
            emb_model_name = self.config.get("embedding", {}).get("model_name", "")
            is_e5 = "e5" in emb_model_name.lower()
            e5_query = f"query: {expanded_query}" if is_e5 else expanded_query
            
            q_vec_emb = self.embeddings.embed_query(e5_query)
            embed_scores = cosine_similarity([q_vec_emb], self.embed_matrix).flatten()
            
            def normalize(scores):
                if max(scores) == min(scores): return scores
                return (scores - min(scores)) / (max(scores) - min(scores))
                
            norm_tfidf = normalize(tfidf_scores)
            norm_embed = normalize(embed_scores)
            
            tfidf_rank = {i: rank for rank, i in enumerate(np.argsort(-norm_tfidf))}
            embed_rank = {i: rank for rank, i in enumerate(np.argsort(-norm_embed))}
            
            k = 60
            results = []
            for i in range(len(self.chunks)):
                rrf_score = 1.0/(k + tfidf_rank[i]) + 1.0/(k + embed_rank[i])
                sim_score = max(float(tfidf_scores[i]), float(embed_scores[i]))
                if sim_score >= self.threshold:
                    results.append({
                        "chunk_id": self.chunks[i]["id"],
                        "content": self.chunks[i]["content"],
                        "score": float(sim_score),
                        "rrf_score": float(rrf_score)
                    })
                    
            results.sort(key=lambda x: x["rrf_score"], reverse=True)
            return results[:top_k]
            
        elif self.mode == "chroma" and self.collection is not None:
            emb_model_name = self.config.get("embedding", {}).get("model_name", "")
            is_e5 = "e5" in emb_model_name.lower()
            e5_query = f"query: {expanded_query}" if is_e5 else expanded_query
            
            q_vec_emb = self.embeddings.embed_query(e5_query)
            query_results = self.collection.query(
                query_embeddings=[q_vec_emb],
                n_results=min(top_k, len(self.chunks)),
                include=["documents", "metadatas", "distances"]
            )
            
            results = []
            if query_results and query_results.get("documents") and query_results["documents"][0]:
                docs = query_results["documents"][0]
                metas = query_results["metadatas"][0] if query_results.get("metadatas") else [{}] * len(docs)
                distances = query_results["distances"][0] if query_results.get("distances") else [0.0] * len(docs)
                
                for doc_text, meta, dist in zip(docs, metas, distances):
                    sim_score = max(0.0, 1.0 - float(dist))
                    if sim_score >= self.threshold:
                        chunk_id = meta.get("chunk_id", 0) if isinstance(meta, dict) else 0
                        results.append({
                            "chunk_id": chunk_id,
                            "content": doc_text,
                            "score": float(sim_score)
                        })
                        
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        return []

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Stage 2: Re-ranks candidate chunks using the Cross-Encoder model.
        
        Args:
            query: The user query string.
            candidates: List of candidate chunks from Stage 1.
            top_k: Number of refined chunks to return.
            
        Returns:
            List[Dict[str, Any]]: Re-ranked chunks sorted by relevance score.
        """
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]
            
        reranker = self._get_reranker()
        if reranker is None:
            return candidates[:top_k]
            
        try:
            pairs = [[query, doc["content"]] for doc in candidates]
            raw_scores = reranker.predict(pairs)
            
            reranked_results = []
            for doc, raw_s in zip(candidates, raw_scores):
                s = float(raw_s)
                stage1_s = float(doc.get("score", 0.0))
                # Use raw Cross-Encoder logit directly as the relevance score.
                # These logits are valid for ranking and threshold comparison.
                # They are NOT calibrated probabilities — do not call them "confidence".
                new_doc = dict(doc)
                new_doc["stage1_score"] = stage1_s
                new_doc["score"] = s
                reranked_results.append(new_doc)
                
            reranked_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Level 1: Chunk-Level Filtering (Drop noise chunks below min_chunk_score)
            min_chunk_score = self.reranking_config.get("min_chunk_score", None)
            if min_chunk_score is not None:
                filtered_results = [doc for doc in reranked_results if doc["score"] >= float(min_chunk_score)]
            else:
                filtered_results = reranked_results
                
            logger.debug(f"Re-ranked {len(candidates)} candidate chunks -> {len(filtered_results)} passed min_chunk_score -> returning top {top_k}")
            return filtered_results[:top_k]
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}. Returning Stage 1 candidates.")
            return candidates[:top_k]

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes Two-Stage Search (Retrieve candidates -> Re-rank).
        
        Args:
            query (str): The search query string.
            top_k (Optional[int]): Target number of final chunks to return.
            
        Returns:
            List[Dict[str, Any]]: Final list of top relevant chunks with scores.
        """
        if not self.chunks or not query.strip():
            return []
            
        # Determine Stage 1 candidate count and Stage 2 final top_k
        if self.reranking_enabled:
            stage1_k = self.reranking_config.get("candidate_top_k", 10)
            final_k = top_k if top_k is not None else self.reranking_config.get("final_top_k", 5)
        else:
            stage1_k = top_k if top_k is not None else self.config["search"].get("top_k", 5)
            final_k = stage1_k
            
        expanded_query = self._expand_synonyms(query)
        logger.debug(f"Expanded query: {expanded_query}")
        
        # Stage 1: Candidate Retrieval
        candidates = self._stage1_retrieve(expanded_query, stage1_k)
        
        # Stage 2: Cross-Encoder Re-ranking
        if self.reranking_enabled and candidates:
            return self.rerank(expanded_query, candidates, top_k=final_k)
            
        return candidates[:final_k]
