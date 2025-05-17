"""
OmniCoreX Memory Manager

Advanced memory and context management utility to handle short-term and long-term knowledge retention,
supporting adaptive reasoning and continual learning within the OmniCoreX AI system.

Features:
- Dual memory layers: short-term working memory and long-term persistent memory.
- Efficient retrieval with similarity search (vector-based) for context relevance.
- Context window management for input conditioning.
- Supports updating, forgetting, and pruning memory entries.
- Interfaces for integration with external vector stores or databases.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque
from threading import RLock

class MemoryEntry:
    """
    Represents a single memory record with timestamp, content and optional embedding vector.
    """
    def __init__(self, content: Any, embedding: Optional[np.ndarray] = None):
        self.content = content
        self.embedding = embedding
        self.timestamp = time.time()

    def __repr__(self):
        return f"MemoryEntry timestamp={self.timestamp} content={self.content}>"

class ShortTermMemory:
    """
    Short-term working memory implemented as a fixed-size deque for recent context.
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.lock = RLock()

    def add(self, content: Any, embedding: Optional[np.ndarray] = None):
        with self.lock:
            entry = MemoryEntry(content, embedding)
            self.memory.append(entry)

    def get_recent(self, n: Optional[int] = None) -> List[MemoryEntry]:
        with self.lock:
            if n is None or n > len(self.memory):
                return list(self.memory)
            else:
                return list(self.memory)[-n:]

    def clear(self):
        with self.lock:
            self.memory.clear()

class LongTermMemory:
    """
    Long-term memory stores past knowledge persistently.
    This simple implementation stores entries with embeddings for similarity search.
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self.lock = RLock()

    def add(self, content: Any, embedding: Optional[np.ndarray] = None):
        with self.lock:
            entry = MemoryEntry(content, embedding)
            self.entries.append(entry)

    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve top-k memory entries most similar to the query embedding using cosine similarity.

        Args:
            query_embedding: numpy array vector.
            top_k: Number of top similar entries to return.

        Returns:
            List of MemoryEntry objects.
        """
        with self.lock:
            if not self.entries or query_embedding is None:
                return []

            embeddings = np.array([e.embedding for e in self.entries if e.embedding is not None])
            if embeddings.size == 0:
                return []

            # Normalize embeddings
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm_query = query_embedding / np.linalg.norm(query_embedding)

            # Cosine similarity
            similarities = np.dot(norm_embeddings, norm_query)
            top_indices = similarities.argsort()[-top_k:][::-1]

            return [self.entries[i] for i in top_indices]

    def prune_older_than(self, max_age_seconds: float):
        """
        Remove entries older than max_age_seconds from now.

        Args:
            max_age_seconds: Maximum age to keep entries.
        """
        current_time = time.time()
        with self.lock:
            self.entries = [e for e in self.entries if current_time - e.timestamp = max_age_seconds]

    def clear(self):
        with self.lock:
            self.entries.clear()


class MemoryManager:
    """
    Combined manager providing unified interface over short-term and long-term memory.

    Supports addition, retrieval, context building, and pruning.
    """

    def __init__(self,
                 short_term_size: int = 50,
                 long_term_prune_age: float = 60 * 60 * 24 * 7,  # One week default
                 embedding_function: Optional[Callable[[Any], np.ndarray]] = None):
        """
        Initialize MemoryManager.

        Args:
            short_term_size: Max entries in short-term memory.
            long_term_prune_age: Max age for pruning long-term memory in seconds.
            embedding_function: Callable converting text/content to vector embeddings.
        """
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory()
        self.prune_age = long_term_prune_age
        self.embedding_function = embedding_function

    def add_memory(self, content: Any):
        """
        Add a memory record to short-term and long-term memory with embedding.

        Args:
            content: Content string or data representing memory.
        """
        embedding = None
        if self.embedding_function is not None:
            try:
                embedding = self.embedding_function(content)
            except Exception as e:
                embedding = None

        self.short_term.add(content, embedding)
        self.long_term.add(content, embedding)

    def get_recent(self, n: Optional[int] = None) -> List[Any]:
        """
        Retrieves recent content entries from short-term memory.

        Args:
            n: Number of recent entries to retrieve.

        Returns:
            List of content objects.
        """
        entries = self.short_term.get_recent(n)
        return [e.content for e in entries]

    def retrieve_similar(self, query_content: Any, top_k: int = 5) -> List[Any]:
        """
        Retrieve top-k long term similar memories based on query content.

        Args:
            query_content: Query string or data.
            top_k: Number of matches to retrieve.

        Returns:
            List of content objects.
        """
        if self.embedding_function is None:
            return []

        try:
            query_emb = self.embedding_function(query_content)
        except Exception:
            return []

        top_entries = self.long_term.retrieve_similar(query_emb, top_k)
        return [e.content for e in top_entries]

    def prune_long_term(self):
        """
        Prune old entries from long-term memory.
        """
        self.long_term.prune_older_than(self.prune_age)

    def clear_all(self):
        """
        Clear both short-term and long-term memories.
        """
        self.short_term.clear()
        self.long_term.clear()


if __name__ == "__main__":
    # Basic test and demonstration

    def dummy_embedding_func(text: str) -> np.ndarray:
        # Very simple embedding: vector of char ordinals normalized
        vec = np.array([ord(c) for c in text], dtype=np.float32)
        if vec.size == 0:
            return np.zeros(10)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        # Pad or truncate to length 10
        if vec.size  10:
            vec = np.pad(vec, (0, 10 - vec.size))
        else:
            vec = vec[:10]
        return vec

    mem_manager = MemoryManager(short_term_size=5, long_term_prune_age=60*60*24*30, embedding_function=dummy_embedding_func)

    print("Adding memories...")
    mem_manager.add_memory("Learned about AI models.")
    mem_manager.add_memory("OmniCoreX is powerful.")
    mem_manager.add_memory("Recall adaptive reasoning.")
    mem_manager.add_memory("Real-time decision making enabled.")
    mem_manager.add_memory("Deep multi-modal integration.")

    print("Recent memories:")
    for mem in mem_manager.get_recent():
        print("-", mem)

    print("Retrieving similar memories for query 'adaptive AI':")
    similars = mem_manager.retrieve_similar("adaptive AI", top_k=3)
    for s in similars:
        print("*", s)

    print("Pruning memories older than prune age...")
    mem_manager.prune_long_term()
    print("Memory manager demo completed.")
