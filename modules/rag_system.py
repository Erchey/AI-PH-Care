"""
RAG System for Medical Knowledge Retrieval
Uses vector embeddings to retrieve relevant medical information from documents
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader
)
from langchain.schema import Document
from typing import List, Dict
import os
from pathlib import Path


class MedicalRAGSystem:
    """
    RAG system for retrieving medical knowledge from documents
    Supports PDFs, web links, text files, and markdown
    """
    
    def __init__(self, llm, persist_directory="../vector_store"):
        self.llm = llm
        self.persist_directory = persist_directory
        
        # Initialize embeddings (can work offline)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small",
            model_kwargs={"device": "cpu", "torch_dtype": "auto"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def ingest_document(self, file_path: str, metadata: Dict = None):
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to document or URL
            metadata: Additional metadata (source, category, etc.)
        """
        
        # Determine loader based on file type
        if file_path.startswith("http://") or file_path.startswith("https://"):
            loader = WebBaseLoader(file_path)
            doc_type = "web"
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            doc_type = "pdf"
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            doc_type = "markdown"
        else:
            loader = TextLoader(file_path)
            doc_type = "text"
        
        # Load and split documents
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        
        # Add metadata
        for doc in splits:
            doc.metadata.update({
                "source": file_path,
                "doc_type": doc_type,
                **(metadata or {})
            })
        
        # Add to vector store
        self.vector_store.add_documents(splits)
        self.vector_store.persist()
        
        print(f"✓ Ingested {len(splits)} chunks from {file_path}")
        
        return len(splits)
    

    def ingest_folder(self, folder_path: str = "../medical_docs"):
        """
        Ingest and vectorize all documents inside a folder.
        Automatically handles PDF, text, markdown, and web link files.

        Args:
            folder_path: Directory containing documents.
        """
        import glob

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            return 0

        supported_ext = (".pdf", ".txt", ".md")
        files = []
        for ext in supported_ext:
            files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

        if not files:
            print(f"⚠️ No supported documents found in {folder_path}")
            return 0

        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.ingest_document(file_path)
                total_chunks += chunks
            except Exception as e:
                print(f"❌ Error ingesting {file_path}: {e}")

        print(f"\n✅ Finished ingesting {len(files)} documents ({total_chunks} chunks total)")
        return total_chunks



    def ingest_bulk(self, sources: List[Dict]):
        """
        Ingest multiple documents at once
        
        Args:
            sources: List of dicts with 'path' and 'metadata'
        """
        
        total_chunks = 0
        for source in sources:
            try:
                chunks = self.ingest_document(
                    source['path'],
                    source.get('metadata', {})
                )
                total_chunks += chunks
            except Exception as e:
                print(f"Error ingesting {source['path']}: {e}")
        
        print(f"\n✓ Total: Ingested {total_chunks} chunks from {len(sources)} sources")
        
        return total_chunks
    
    def retrieve(self, query: str, k: int = 4, filter_metadata: Dict = None):
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_metadata: Filter by metadata (e.g., category, source)
        
        Returns:
            List of relevant documents with content and metadata
        """
        
        # Perform similarity search
        if filter_metadata:
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_metadata
            )
        else:
            docs = self.vector_store.similarity_search(query, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return results
    
    def retrieve_with_scores(self, query: str, k: int = 4):
        """Retrieve documents with relevance scores"""
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return results
    
    def answer_with_sources(self, query: str, k: int = 4):
        """
        Answer a query using RAG with source citations
        
        Args:
            query: User's question
            k: Number of source documents to retrieve
        
        Returns:
            AI-generated answer with source citations
        """
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k=k)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information in my knowledge base to answer this query confidently. Please consult medical guidelines or a senior healthcare professional.",
                "sources": []
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Source {i+1}: {doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Generate answer using LLM
        prompt = f"""Based on the following medical knowledge sources, answer the healthcare question accurately and concisely.

Question: {query}

Medical Knowledge:
{context}

Instructions:
- Provide a clear, evidence-based answer
- Cite which sources support your answer (e.g., "According to Source 1...")
- If information is insufficient, state that clearly
- For clinical decisions, recommend consulting a healthcare professional
- Use simple language appropriate for PHC workers

Answer:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": retrieved_docs,
            "query": query
        }
    
    def semantic_search(self, query: str, category: str = None, k: int = 5):
        """
        Perform semantic search with optional category filtering
        
        Args:
            query: Search query
            category: Filter by category (protocols, drugs, diagnosis, etc.)
            k: Number of results
        """
        
        filter_dict = {"category": category} if category else None
        
        results = self.retrieve(query, k=k, filter_metadata=filter_dict)
        
        return results
    
    def get_emergency_protocol(self, condition: str):
        """Retrieve emergency protocol using RAG"""
        
        query = f"emergency protocol for {condition} immediate treatment steps"
        
        return self.answer_with_sources(query, k=3)
    
    def get_drug_information(self, drug_name: str):
        """Retrieve drug information using RAG"""
        
        query = f"{drug_name} dosage contraindications side effects interactions"
        
        return self.answer_with_sources(query, k=3)
    
    def get_diagnostic_guidance(self, symptoms: str):
        """Get diagnostic guidance for symptoms"""
        
        query = f"diagnosis differential diagnosis for patient with {symptoms}"
        
        return self.answer_with_sources(query, k=4)
    
    def clear_store(self):
        """Clear the vector store (use with caution)"""
        
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("✓ Vector store cleared")
        
        self.vector_store = self._initialize_vector_store()


def setup_initial_knowledge_base(rag_system: MedicalRAGSystem):
    """
    Setup initial knowledge base with medical resources
    This is where you'd add your medical documents and URLs
    """
    
    # Example sources - replace with actual medical guidelines
    sources = [
        {
            "path": "https://www.who.int/publications/guidelines",
            "metadata": {
                "category": "guidelines",
                "source_org": "WHO",
                "language": "English"
            }
        },
        # Add your medical PDFs
        {
            "path": "./medical_docs/emergency_protocols.pdf",
            "metadata": {
                "category": "emergency",
                "doc_type": "protocol"
            }
        },
        # Add drug formularies
        {
            "path": "./medical_docs/essential_medicines.pdf",
            "metadata": {
                "category": "drugs",
                "doc_type": "formulary"
            }
        }
    ]
    
    print("Setting up medical knowledge base...")
    print("Note: Add your actual medical documents and URLs above")
    
    # Only ingest if files exist
    for source in sources:
        path = source['path']
        if path.startswith("http") or os.path.exists(path):
            try:
                rag_system.ingest_document(path, source.get('metadata'))
            except Exception as e:
                print(f"Warning: Could not ingest {path}: {e}")




if __name__ == "__main__":
    # Example usage
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.2
    )
    
    rag = MedicalRAGSystem(llm)
    
    # Setup knowledge base (first time only)
    # setup_initial_knowledge_base(rag)
    
    # Query the system
    result = rag.answer_with_sources(
        "What is the treatment protocol for viral infection?"
    )
    
    print("Answer:", result['answer'])
    print("\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['source']}")

if __name__ == "__main__":
    # Example usage
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.2
    )
    
    rag = MedicalRAGSystem(llm)
    
    # Setup knowledge base (first time only)
    # setup_initial_knowledge_base(rag)
    
    # Query the system
    result = rag.answer_with_sources(
        "What is the treatment protocol for viral infection?"
    )
    
    print("Answer:", result['answer'])
    print("\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['source']}")
