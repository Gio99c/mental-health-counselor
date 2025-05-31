import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple
import json

class RedditRAG:
    """Simple RAG system for finding similar Reddit posts."""
    
    def __init__(self, csv_path: str = "500_Reddit_users_posts_labels.csv"):
        self.csv_path = csv_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = "reddit_embeddings.pkl"
        self.posts_path = "reddit_posts.pkl"
        
        self.posts_data = None
        self.embeddings = None
        
        self._load_or_create_embeddings()
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones."""
        if os.path.exists(self.embeddings_path) and os.path.exists(self.posts_path):
            print("Loading existing embeddings...")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(self.posts_path, 'rb') as f:
                self.posts_data = pickle.load(f)
            print(f"Loaded {len(self.posts_data)} posts with embeddings")
        else:
            print("Creating new embeddings from dataset...")
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings from the Reddit dataset."""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded dataset with {len(df)} posts")
            
            # Clean and prepare posts
            posts = []
            for idx, row in df.iterrows():
                try:
                    # Clean the post text
                    post_text = str(row['Post'])
                    if post_text.startswith("['") and post_text.endswith("']"):
                        # Remove brackets and quotes, then split by "', '"
                        post_text = post_text[2:-2]
                        post_parts = post_text.split("', '")
                        post_text = " ".join(post_parts)
                    
                    # Clean up any remaining artifacts
                    post_text = post_text.replace("\\'", "'")
                    
                    posts.append({
                        'id': idx,
                        'user': row['User'],
                        'text': post_text,
                        'label': row['Label'],
                        'preview': post_text[:200] + "..." if len(post_text) > 200 else post_text
                    })
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
            
            self.posts_data = posts
            print(f"Processed {len(posts)} posts")
            
            # Create embeddings
            texts = [post['text'] for post in posts]
            print("Creating embeddings...")
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Save embeddings and posts
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            with open(self.posts_path, 'wb') as f:
                pickle.dump(self.posts_data, f)
            
            print("Embeddings created and saved successfully!")
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            self.posts_data = []
            self.embeddings = np.array([])
    
    def find_similar_posts(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Find the top_k most similar posts to the query."""
        if self.embeddings is None or len(self.posts_data) == 0:
            return []
        
        try:
            # Create embedding for query
            query_embedding = self.model.encode([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return similar posts with similarity scores
            similar_posts = []
            for idx in top_indices:
                post = self.posts_data[idx].copy()
                post['similarity_score'] = float(similarities[idx])
                similar_posts.append(post)
            
            return similar_posts
            
        except Exception as e:
            print(f"Error finding similar posts: {e}")
            return []
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get the distribution of labels in the dataset."""
        if not self.posts_data:
            return {}
        
        labels = [post['label'] for post in self.posts_data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts

def format_similar_posts_for_display(similar_posts: List[Dict]) -> str:
    """Format similar posts for display in the UI."""
    if not similar_posts:
        return "No similar posts found in the database."
    
    formatted_output = "## 📚 Similar Cases from Reddit Database\n\n"
    
    for i, post in enumerate(similar_posts, 1):
        similarity_percent = int(post['similarity_score'] * 100)
        
        # Color coding based on label
        label_colors = {
            'Supportive': '🟢',
            'Indicator': '🟡', 
            'Ideation': '🟠',
            'Behavior': '🔴',
            'Attempt': '🚨'
        }
        
        label_icon = label_colors.get(post['label'], '⚪')
        
        formatted_output += f"""
### {i}. {label_icon} Case {post['id']} - {post['label']} Risk ({similarity_percent}% similar)

**User:** {post['user']}

**Post Content:**
> {post['preview']}

**Risk Classification:** {post['label']}

---
"""
    
    return formatted_output 