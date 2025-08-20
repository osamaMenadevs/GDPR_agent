import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingsProcessor:
    """
    Alternative embeddings processor using TF-IDF instead of sentence-transformers
    This bypasses the cached_download issue completely
    """
    
    def __init__(self):
        """Initialize with TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.95,
            min_df=2
        )
        self.vectors = None
        self.texts = []
        self.metadata = []
        self.index = None
        
    def load_gdpr_data(self, text_file: str = "gdpr_text.csv", violations_file: str = "gdpr_violations.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load GDPR data files"""
        try:
            logger.info("Loading GDPR data files...")
            gdpr_text_df = pd.read_csv(text_file)
            violations_df = pd.read_csv(violations_file)
            logger.info(f"Loaded {len(gdpr_text_df)} GDPR text entries")
            logger.info(f"Loaded {len(violations_df)} GDPR violation cases")
            return gdpr_text_df, violations_df
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise
    
    def prepare_texts(self, gdpr_text_df: pd.DataFrame, violations_df: pd.DataFrame) -> List[str]:
        """Prepare texts for vectorization"""
        texts = []
        metadata = []
        
        # Process GDPR text data
        for idx, row in gdpr_text_df.iterrows():
            text = f"Chapter {row['chapter']}: {row['chapter_title']} Article {row['article']}: {row['article_title']} {row['gdpr_text']}"
            texts.append(text)
            metadata.append({
                'type': 'gdpr_text',
                'chapter': row['chapter'],
                'article': row['article'],
                'chapter_title': row['chapter_title'],
                'article_title': row['article_title'],
                'source_index': idx
            })
        
        # Process violations data
        for idx, row in violations_df.iterrows():
            text = f"GDPR Violation: {row['name']} Fine: €{row['price']} Authority: {row['authority']} Articles: {row['article_violated']} Type: {row['type']} Summary: {row['summary']}"
            texts.append(text)
            metadata.append({
                'type': 'violation_case',
                'country': row['name'],
                'fine_amount': row['price'],
                'authority': row['authority'],
                'articles_violated': row['article_violated'],
                'violation_type': row['type'],
                'summary': row['summary'],
                'source_index': idx
            })
        
        self.texts = texts
        self.metadata = metadata
        logger.info(f"Prepared {len(texts)} texts for vectorization")
        return texts
    
    def create_vectors(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF vectors"""
        logger.info("Creating TF-IDF vectors...")
        self.vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Created vectors with shape: {self.vectors.shape}")
        return self.vectors
    
    def create_simple_index(self):
        """Create a simple search index using the vectors"""
        logger.info("Creating search index...")
        # Convert sparse matrix to dense for FAISS
        dense_vectors = self.vectors.toarray().astype('float32')
        
        # Create FAISS index
        dimension = dense_vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(dense_vectors)
        
        logger.info(f"Created index with {self.index.ntotal} vectors")
        return self.index
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or self.vectors is None:
            raise ValueError("Index not created. Run process_all() first.")
        
        # Transform query using same vectorizer
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Search using FAISS
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                result = {
                    'rank': i + 1,
                    'similarity_score': similarity_score,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save_simple_embeddings(self, save_dir: str = "simple_embeddings"):
        """Save the simple embeddings and index"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectorizer
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save vectors
        with open(os.path.join(save_dir, "vectors.pkl"), "wb") as f:
            pickle.dump(self.vectors, f)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "simple_index.bin"))
        
        # Save texts and metadata
        with open(os.path.join(save_dir, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)
        
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved simple embeddings to {save_dir}")
    
    def load_simple_embeddings(self, save_dir: str = "simple_embeddings"):
        """Load the simple embeddings and index"""
        try:
            # Load vectorizer
            with open(os.path.join(save_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load vectors
            with open(os.path.join(save_dir, "vectors.pkl"), "rb") as f:
                self.vectors = pickle.load(f)
            
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(save_dir, "simple_index.bin"))
            
            # Load texts and metadata
            with open(os.path.join(save_dir, "texts.pkl"), "rb") as f:
                self.texts = pickle.load(f)
            
            with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded simple embeddings from {save_dir}")
            
        except Exception as e:
            logger.error(f"Error loading simple embeddings: {str(e)}")
            raise
    
    def process_all(self):
        """Complete processing pipeline"""
        logger.info("Starting simple embeddings processing...")
        
        # Load data
        gdpr_text_df, violations_df = self.load_gdpr_data()
        
        # Prepare texts
        texts = self.prepare_texts(gdpr_text_df, violations_df)
        
        # Create vectors
        self.create_vectors(texts)
        
        # Create index
        self.create_simple_index()
        
        # Save everything
        self.save_simple_embeddings()
        
        logger.info("Simple embeddings processing completed!")


if __name__ == "__main__":
    # Test the simple embeddings
    processor = SimpleEmbeddingsProcessor()
    processor.process_all()
    
    # Test search
    results = processor.search_similar("data subject rights", k=3)
    for result in results:
        print(f"Rank {result['rank']}: {result['similarity_score']:.3f}")
        print(f"Type: {result['metadata']['type']}")
        print(f"Text: {result['text'][:200]}...")
        print("---")
