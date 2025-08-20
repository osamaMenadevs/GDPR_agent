import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDPRDataProcessor:
    """
    Handles processing of GDPR data using TF-IDF instead of sentence-transformers
    This avoids the cached_download error completely
    """
    
    def __init__(self):
        """
        Initialize with TF-IDF vectorizer (no sentence-transformers needed)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.vectors = None
        self.index = None
        self.texts = []
        self.metadata = []
        
    def load_gdpr_data(self, text_file: str = "gdpr_text.csv", violations_file: str = "gdpr_violations.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load GDPR text and violations data
        
        Args:
            text_file: Path to GDPR text CSV file
            violations_file: Path to GDPR violations CSV file
            
        Returns:
            Tuple of (gdpr_text_df, violations_df)
        """
        try:
            logger.info("Loading GDPR data files...")
            
            # Load GDPR text data
            gdpr_text_df = pd.read_csv(text_file)
            logger.info(f"Loaded {len(gdpr_text_df)} GDPR text entries")
            
            # Load GDPR violations data
            violations_df = pd.read_csv(violations_file)
            logger.info(f"Loaded {len(violations_df)} GDPR violation cases")
            
            return gdpr_text_df, violations_df
            
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise
    
    def prepare_texts_for_embedding(self, gdpr_text_df: pd.DataFrame, violations_df: pd.DataFrame) -> List[str]:
        """
        Prepare texts from both datasets for embedding generation
        
        Args:
            gdpr_text_df: DataFrame with GDPR text data
            violations_df: DataFrame with GDPR violations data
            
        Returns:
            List of processed text strings
        """
        texts = []
        metadata = []
        
        # Process GDPR text data
        for idx, row in gdpr_text_df.iterrows():
            # Combine chapter, article, and text for comprehensive context
            text = f"Chapter {row['chapter']}: {row['chapter_title']} | "
            text += f"Article {row['article']}: {row['article_title']} | "
            text += f"Sub-article {row['sub_article']}: {row['gdpr_text']}"
            
            texts.append(text)
            metadata.append({
                'type': 'gdpr_text',
                'chapter': row['chapter'],
                'article': row['article'],
                'sub_article': row['sub_article'],
                'chapter_title': row['chapter_title'],
                'article_title': row['article_title'],
                'href': row['href'],
                'source_index': idx
            })
        
        # Process violations data
        for idx, row in violations_df.iterrows():
            # Create comprehensive violation case description
            text = f"GDPR Violation Case: {row['name']} | "
            text += f"Authority: {row['authority']} | "
            text += f"Fine: €{row['price']} | "
            text += f"Date: {row['date']} | "
            text += f"Articles Violated: {row['article_violated']} | "
            text += f"Type: {row['type']} | "
            text += f"Summary: {row['summary']}"
            
            texts.append(text)
            metadata.append({
                'type': 'violation_case',
                'case_id': row['id'],
                'country': row['name'],
                'fine_amount': row['price'],
                'authority': row['authority'],
                'date': row['date'],
                'controller': row['controller'],
                'articles_violated': row['article_violated'],
                'violation_type': row['type'],
                'summary': row['summary'],
                'source': row['source'],
                'source_index': idx
            })
        
        self.texts = texts
        self.metadata = metadata
        logger.info(f"Prepared {len(texts)} texts for embedding (GDPR: {len(gdpr_text_df)}, Violations: {len(violations_df)})")
        
        return texts
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate TF-IDF vectors for the given texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Sparse matrix of TF-IDF vectors
        """
        logger.info("Generating TF-IDF vectors...")
        self.vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Generated vectors shape: {self.vectors.shape}")
        return self.vectors
    
    def create_faiss_index(self, vectors) -> faiss.Index:
        """
        Create and populate FAISS index with TF-IDF vectors
        
        Args:
            vectors: TF-IDF sparse matrix
            
        Returns:
            FAISS index
        """
        logger.info("Creating FAISS index...")
        
        # Convert sparse matrix to dense for FAISS
        dense_vectors = vectors.toarray().astype('float32')
        
        # Create flat index
        dimension = dense_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(dense_vectors)
        
        self.index = index
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        return index
    
    def save_embeddings_and_index(self, vectors, save_dir: str = "embeddings"):
        """
        Save TF-IDF vectors, FAISS index, and metadata to disk
        
        Args:
            vectors: TF-IDF sparse matrix
            save_dir: Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectorizer
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save vectors
        with open(os.path.join(save_dir, "vectors.pkl"), "wb") as f:
            pickle.dump(vectors, f)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))
        
        # Save metadata and texts
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        with open(os.path.join(save_dir, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)
        
        logger.info(f"Saved TF-IDF vectors and index to {save_dir}")
    
    def load_embeddings_and_index(self, save_dir: str = "embeddings"):
        """
        Load TF-IDF vectors, FAISS index, and metadata from disk
        
        Args:
            save_dir: Directory containing saved files
        """
        try:
            # Load vectorizer
            with open(os.path.join(save_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load vectors
            with open(os.path.join(save_dir, "vectors.pkl"), "rb") as f:
                self.vectors = pickle.load(f)
            
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
            
            # Load metadata and texts
            with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
            
            with open(os.path.join(save_dir, "texts.pkl"), "rb") as f:
                self.texts = pickle.load(f)
            
            logger.info(f"Loaded TF-IDF vectors and index from {save_dir}")
            return self.vectors
            
        except Exception as e:
            logger.error(f"Error loading vectors and index: {str(e)}")
            raise
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using the query
        
        Args:
            query: Search query string
            k: Number of similar documents to return
            
        Returns:
            List of dictionaries containing search results
        """
        if self.index is None or self.vectorizer is None:
            raise ValueError("Index not initialized. Please create or load index first.")
        
        # Transform query using TF-IDF vectorizer
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):  # Valid index
                result = {
                    'rank': i + 1,
                    'similarity_score': float(1 / (1 + distance)),  # Convert distance to similarity
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def process_and_save_all(self):
        """
        Complete pipeline: load data, generate TF-IDF vectors, create index, and save
        """
        logger.info("Starting TF-IDF processing pipeline...")
        
        # Load data
        gdpr_text_df, violations_df = self.load_gdpr_data()
        
        # Prepare texts
        texts = self.prepare_texts_for_embedding(gdpr_text_df, violations_df)
        
        # Generate TF-IDF vectors
        vectors = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.create_faiss_index(vectors)
        
        # Save everything
        self.save_embeddings_and_index(vectors)
        
        logger.info("TF-IDF processing pipeline completed successfully!")
        return vectors


if __name__ == "__main__":
    # Example usage
    processor = GDPRDataProcessor()
    vectors = processor.process_and_save_all()
    
    # Test search
    results = processor.search_similar("data subject rights", k=3)
    for result in results:
        print(f"Rank {result['rank']}: {result['similarity_score']:.3f}")
        print(f"Type: {result['metadata']['type']}")
        print(f"Text: {result['text'][:200]}...")
        print("---")
