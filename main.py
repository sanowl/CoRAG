import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import logging
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalChain:
    """
    Represents a chain of retrieval steps including sub-queries, answers, and final response.
    
    Attributes:
        sub_queries: List of generated sub-queries
        sub_answers: List of answers to the sub-queries
        final_answer: Final generated answer to the main query
        log_likelihood: Log likelihood score of the answer
        execution_time: Time taken to generate the chain
        metadata: Additional information about the chain
    """
    sub_queries: List[str]
    sub_answers: List[str]
    final_answer: str
    log_likelihood: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentRetriever:
    """
    Interface for document retrieval systems.
    
    This class should be extended to implement actual retrieval logic
    using vector stores, search engines, etc.
    """
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries with at least 'content' and 'metadata' keys
        """
        raise NotImplementedError("Subclasses must implement retrieve()")

class SimpleVectorRetriever(DocumentRetriever):
    """
    A simple vector-based document retriever.
    
    This implementation uses a vector database for semantic search.
    """
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 documents: Optional[List[Dict[str, Any]]] = None,
                 vector_db_path: Optional[str] = None):
        """
        Initialize the retriever with documents or load from a vector database.
        
        Args:
            embedding_model: Model to use for embeddings
            documents: List of document dictionaries with 'content' and 'metadata' keys
            vector_db_path: Path to saved vector database
        """
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # Set device if GPU is available
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                
            if vector_db_path and os.path.exists(vector_db_path):
                self._load_vector_db(vector_db_path)
            elif documents:
                self._index_documents(documents)
            else:
                logger.warning("No documents or vector DB provided; retriever initialized empty")
                self.document_embeddings = None
                self.documents = []
                self.index = None
        except ImportError:
            logger.error("sentence-transformers and/or faiss required for SimpleVectorRetriever")
            raise
            
    def _index_documents(self, documents: List[Dict[str, Any]]):
        """Create a searchable index from documents."""
        logger.info(f"Indexing {len(documents)} documents...")
        self.documents = documents
        
        # Extract content for embedding
        texts = [doc["content"] for doc in documents]
        
        # Create embeddings
        self.document_embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        
        # Convert to numpy for FAISS
        if torch.is_tensor(self.document_embeddings):
            embeddings_np = self.document_embeddings.cpu().numpy()
        else:
            embeddings_np = self.document_embeddings
            
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        
        logger.info("Indexing complete")
        
    def _load_vector_db(self, path: str):
        """Load vector database from disk."""
        import faiss
        
        logger.info(f"Loading vector database from {path}")
        
        # Load documents
        with open(os.path.join(path, "documents.json"), "r") as f:
            self.documents = json.load(f)
            
        # Load embeddings
        self.document_embeddings = np.load(os.path.join(path, "embeddings.npy"))
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        logger.info(f"Loaded {len(self.documents)} documents from vector database")
        
    def save_vector_db(self, path: str):
        """Save vector database to disk."""
        import faiss
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save documents
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(self.documents, f)
            
        # Save embeddings
        np.save(os.path.join(path, "embeddings.npy"), 
                self.document_embeddings.cpu().numpy() 
                if torch.is_tensor(self.document_embeddings) 
                else self.document_embeddings)
            
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        logger.info(f"Saved vector database to {path}")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant documents for the query.
        
        Args:
            query: The search query
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        if self.index is None or not self.documents:
            logger.warning("No documents indexed, returning empty list")
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=True
        )
        
        # Convert to numpy for FAISS
        if torch.is_tensor(query_embedding):
            query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        else:
            query_embedding_np = query_embedding.reshape(1, -1)
            
        # Search
        top_k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        # Return documents
        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])
            
        return results


class CoRAG:
    """
    Compositional Retrieval-Augmented Generation system.
    
    This system breaks down complex queries into sub-queries, retrieves relevant
    information, and composes a final answer.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        retriever: Optional[DocumentRetriever] = None,
        max_chain_length: int = 6,
        num_samples: int = 4,
        temperature: float = 0.7,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_context_length: int = 4096,
        load_in_8bit: bool = False
    ):
        """
        Initialize the CoRAG system.
        
        Args:
            model_name: HuggingFace model identifier or path
            retriever: Document retriever instance
            max_chain_length: Maximum number of sub-queries to generate
            num_samples: Number of chains to sample in rejection sampling
            temperature: Temperature for generation
            device: Device to run the model on ('cpu', 'cuda', 'mps', etc.)
            cache_dir: Directory to cache models
            max_context_length: Maximum context length for the model
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_name = model_name
        self.max_chain_length = max_chain_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_context_length = max_context_length
        
        # Set device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing CoRAG with model {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            
            # Load model with appropriate precision
            if load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=self.device,
                    load_in_8bit=True,
                    cache_dir=cache_dir
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(self.device)
                
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        # Set up retriever
        if retriever is None:
            logger.warning("No retriever provided, creating a dummy retriever")
            self.retriever = DocumentRetriever()  # Will raise error if used without implementation
        else:
            self.retriever = retriever
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def generate_sub_query(
        self,
        main_query: str,
        previous_queries: List[str],
        previous_answers: List[str]
    ) -> str:
        """
        Generate next sub-query based on current state.
        
        Args:
            main_query: The original user query
            previous_queries: List of previously generated sub-queries
            previous_answers: List of previous answers to sub-queries
            
        Returns:
            Generated sub-query string
        """
        prompt = self._create_sub_query_prompt(
            main_query, previous_queries, previous_answers
        )
        
        return self._generate_text(prompt)

    def generate_sub_answer(
        self,
        sub_query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer for sub-query using retrieved documents.
        
        Args:
            sub_query: The sub-query to answer
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Generated answer string
        """
        # Extract content from document dictionaries
        doc_contents = [doc["content"] for doc in retrieved_docs]
        
        prompt = self._create_sub_answer_prompt(sub_query, doc_contents)
        return self._generate_text(prompt)

    def generate_final_answer(
        self, 
        main_query: str,
        sub_queries: List[str],
        sub_answers: List[str],
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate final answer combining all retrieved information.
        
        Args:
            main_query: The original user query
            sub_queries: List of generated sub-queries
            sub_answers: List of answers to sub-queries
            retrieved_docs: List of additional retrieved document dictionaries
            
        Returns:
            Final answer string
        """
        # Extract content from document dictionaries
        doc_contents = [doc["content"] for doc in retrieved_docs]
        
        prompt = self._create_final_answer_prompt(
            main_query, sub_queries, sub_answers, doc_contents
        )
        return self._generate_text(prompt)

    def answer_query(
        self,
        query: str,
        num_subqueries: Optional[int] = None,
        show_progress: bool = True,
        return_chain: bool = False
    ) -> Union[str, RetrievalChain]:
        """
        Answer a query using the CoRAG approach.
        
        Args:
            query: The user query to answer
            num_subqueries: Override max_chain_length for this query
            show_progress: Whether to show progress bar
            return_chain: Whether to return the full RetrievalChain object
            
        Returns:
            Final answer string or RetrievalChain object
        """
        chain = self._generate_chain(
            query, 
            max_steps=num_subqueries or self.max_chain_length,
            show_progress=show_progress
        )
        
        return chain if return_chain else chain.final_answer

    def rejection_sampling(
        self,
        query: str,
        correct_answer: str,
        show_progress: bool = True
    ) -> RetrievalChain:
        """
        Generate retrieval chains through rejection sampling.
        
        Args:
            query: The user query to answer
            correct_answer: The known correct answer for evaluation
            show_progress: Whether to show progress bar
            
        Returns:
            Best RetrievalChain based on answer likelihood
        """
        best_chain = None
        best_likelihood = float('-inf')

        if show_progress:
            iterator = tqdm(range(self.num_samples), desc="Sampling chains")
        else:
            iterator = range(self.num_samples)

        for _ in iterator:
            chain = self._generate_chain(query, show_progress=False)
            likelihood = self._compute_answer_likelihood(
                chain.final_answer, correct_answer
            )
            
            chain.log_likelihood = likelihood
            
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_chain = chain
                
                if show_progress:
                    tqdm.write(f"New best chain: likelihood={likelihood:.4f}")

        return best_chain

    def batch_answer_queries(
        self,
        queries: List[str],
        max_workers: int = 4,
        **kwargs
    ) -> List[Union[str, RetrievalChain]]:
        """
        Answer multiple queries in parallel.
        
        Args:
            queries: List of queries to answer
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments to pass to answer_query
            
        Returns:
            List of answers or RetrievalChain objects
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.answer_query, query, **kwargs) 
                      for query in queries]
            
            # Process results as they complete
            for future in futures:
                results.append(future.result())
                
        return results

    def _generate_chain(
        self, 
        query: str,
        max_steps: Optional[int] = None,
        show_progress: bool = True
    ) -> RetrievalChain:
        """
        Generate a single retrieval chain.
        
        Args:
            query: The user query to answer
            max_steps: Maximum number of steps (overrides max_chain_length)
            show_progress: Whether to show progress bar
            
        Returns:
            Complete RetrievalChain
        """
        start_time = time.time()
        sub_queries = []
        sub_answers = []
        max_steps = max_steps or self.max_chain_length
        
        if show_progress:
            step_iterator = tqdm(range(max_steps), desc="Generating chain")
        else:
            step_iterator = range(max_steps)
        
        for _ in step_iterator:
            # Generate sub-query
            sub_query = self.generate_sub_query(
                query, sub_queries, sub_answers
            )
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(sub_query)
            
            # Generate sub-answer
            sub_answer = self.generate_sub_answer(sub_query, retrieved_docs)
            
            sub_queries.append(sub_query)
            sub_answers.append(sub_answer)
            
            # Check if we have enough information
            if self._should_stop(query, sub_queries, sub_answers):
                if show_progress:
                    tqdm.write(f"Stopping after {len(sub_queries)} sub-queries")
                break

        # Generate final answer
        final_docs = self.retriever.retrieve(query)
        final_answer = self.generate_final_answer(
            query, sub_queries, sub_answers, final_docs
        )
        
        execution_time = time.time() - start_time
        
        return RetrievalChain(
            sub_queries=sub_queries,
            sub_answers=sub_answers,
            final_answer=final_answer,
            execution_time=execution_time,
            metadata={
                "num_steps": len(sub_queries),
                "query": query
            }
        )

    def _should_stop(
        self,
        query: str,
        sub_queries: List[str],
        sub_answers: List[str]
    ) -> bool:
        """
        Decide if we have enough information to answer the query.
        
        Args:
            query: The original user query
            sub_queries: List of generated sub-queries
            sub_answers: List of answers to sub-queries
            
        Returns:
            Boolean indicating whether to stop generating sub-queries
        """
        prompt = self._create_stop_prompt(query, sub_queries, sub_answers)
        response = self._generate_text(prompt)
        return response.strip().lower() in ["yes", "true", "1"]

    def _compute_answer_likelihood(
        self,
        generated_answer: str,
        correct_answer: str
    ) -> float:
        """
        Compute log likelihood of correct answer given generated answer.
        
        Args:
            generated_answer: Model-generated answer
            correct_answer: Known correct answer
            
        Returns:
            Log likelihood score
        """
        # Prepare context
        context = f"Generated Answer: {generated_answer}\nCorrect Answer: "
        
        # Tokenize context followed by correct answer
        inputs = self.tokenizer(
            context,
            return_tensors="pt"
        ).to(self.device)
        
        target_ids = self.tokenizer(
            correct_answer,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Get logits for next tokens
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Sum log probabilities for target tokens
        total_log_prob = 0.0
        for target_idx in target_ids[0]:
            total_log_prob += log_probs[0, target_idx].item()
            
            # Update context for next iteration
            inputs = self.tokenizer(
                context + self.tokenizer.decode(target_idx),
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs) 
                logits = outputs.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
            context += self.tokenizer.decode(target_idx)
            
        return total_log_prob

    @lru_cache(maxsize=128)
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the language model with caching.
        
        Args:
            prompt: The input prompt for generation
            
        Returns:
            Generated text string
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_context_length,
                    temperature=self.temperature,
                    num_return_sequences=1,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode and remove prompt
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response by removing prompt
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            response = self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return ""

    def _create_sub_query_prompt(
        self,
        main_query: str,
        previous_queries: List[str],
        previous_answers: List[str]
    ) -> str:
        """
        Create prompt for sub-query generation.
        
        Args:
            main_query: The original user query
            previous_queries: List of previously generated sub-queries
            previous_answers: List of previous answers to sub-queries
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            "You are using a search engine to answer the main query by "
            "iteratively searching for information. Given the following intermediate "
            "queries and answers, generate a new focused follow-up question "
            "that can help answer the main query.\n\n"
        )
        
        if previous_queries:
            prompt += "Previous queries and answers:\n"
            for i, (q, a) in enumerate(zip(previous_queries, previous_answers), 1):
                prompt += f"Step {i}:\nQ: {q}\nA: {a}\n\n"
                
        prompt += f"Main query: {main_query}\n"
        prompt += "\nGenerate a focused follow-up question to help answer the main query:"
        
        return prompt

    def _create_sub_answer_prompt(
        self,
        sub_query: str,
        retrieved_docs: List[str]
    ) -> str:
        """
        Create prompt for sub-answer generation.
        
        Args:
            sub_query: The sub-query to answer
            retrieved_docs: List of retrieved document contents
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            "Given the following documents, generate a comprehensive and accurate answer "
            "to the query. Base your answer ONLY on the provided documents "
            "and avoid including any information not present in them.\n\n"
        )
        
        prompt += "Documents:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            # Truncate long documents
            doc_text = doc[:2000] + "..." if len(doc) > 2000 else doc
            prompt += f"[{i}] {doc_text}\n\n"
            
        prompt += f"Query: {sub_query}\n"
        prompt += "\nAnswer based on the documents:"
        
        return prompt

    def _create_final_answer_prompt(
        self,
        main_query: str,
        sub_queries: List[str],
        sub_answers: List[str],
        retrieved_docs: List[str]
    ) -> str:
        """
        Create prompt for final answer generation.
        
        Args:
            main_query: The original user query
            sub_queries: List of generated sub-queries
            sub_answers: List of answers to sub-queries
            retrieved_docs: List of additional retrieved document contents
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            "Given the following intermediate research steps and additional documents, "
            "synthesize a comprehensive final answer to the main query.\n\n"
        )
        
        prompt += "Research steps:\n"
        for i, (q, a) in enumerate(zip(sub_queries, sub_answers), 1):
            prompt += f"Step {i}:\nQ: {q}\nA: {a}\n\n"
            
        prompt += "Additional relevant documents:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            # Truncate long documents
            doc_text = doc[:1000] + "..." if len(doc) > 1000 else doc
            prompt += f"[{i}] {doc_text}\n\n"
            
        prompt += f"Main query: {main_query}\n"
        prompt += "\nFinal comprehensive answer:"
        
        return prompt

    def _create_stop_prompt(
        self,
        query: str,
        sub_queries: List[str],
        sub_answers: List[str]
    ) -> str:
        """
        Create prompt for stopping decision.
        
        Args:
            query: The original user query
            sub_queries: List of generated sub-queries
            sub_answers: List of answers to sub-queries
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            "You are researching to answer a complex query. Given the information "
            "collected so far, determine if you have sufficient information to "
            "provide a complete and accurate answer to the main query.\n\n"
        )
        
        prompt += "Information collected so far:\n"
        for i, (q, a) in enumerate(zip(sub_queries, sub_answers), 1):
            prompt += f"Step {i}:\nQ: {q}\nA: {a}\n\n"
            
        prompt += f"Main query: {query}\n"
        prompt += "\nBased on the collected information, do you have enough to answer the main query?\n"
        prompt += "Respond with 'Yes' if you have enough information, or 'No' if more research is needed:"
        
        return prompt
        
    def save_config(self, path: str):
        """
        Save configuration to disk.
        
        Args:
            path: Path to save configuration file
        """
        config = {
            "model_name": self.model_name,
            "max_chain_length": self.max_chain_length,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "max_context_length": self.max_context_length
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {path}")
        
    @classmethod
    def from_config(cls, path: str, **kwargs):
        """
        Create CoRAG instance from saved configuration.
        
        Args:
            path: Path to configuration file
            **kwargs: Override configuration parameters
            
        Returns:
            CoRAG instance
        """
        with open(path, "r") as f:
            config = json.load(f)
            
        # Override with kwargs
        config.update(kwargs)
        
        return cls(**config)


# Example usage
if __name__ == "__main__":
    # Initialize a simple vector retriever with example documents
    example_docs = [
        {"content": "Python is a high-level programming language...", "metadata": {"source": "wiki_python"}},
        {"content": "PyTorch is a machine learning framework...", "metadata": {"source": "wiki_pytorch"}},
        # Add more documents
    ]
    
    retriever = SimpleVectorRetriever(documents=example_docs)
    
    # Initialize CoRAG with the retriever
    corag = CoRAG(
        model_name="meta-llama/Llama-3-8B-Instruct",
        retriever=retriever,
        temperature=0.7
    )
    
    # Answer a query
    query = "What are the main differences between PyTorch and TensorFlow?"
    answer = corag.answer_query(query, show_progress=True)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")