#!/usr/bin/env python3
"""
Quick script to train the BERT model for suicide severity classification
"""

import os
import sys

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import transformers
        import pandas
        import sklearn
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main training function."""
    print("🚀 Starting BERT Suicide Severity Model Training")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check if dataset exists
    if not os.path.exists("500_Reddit_users_posts_labels.csv"):
        print("❌ Dataset file '500_Reddit_users_posts_labels.csv' not found!")
        sys.exit(1)
    
    # Run training
    try:
        from train_model import train_bert_model
        print("📚 Loading dataset and starting training...")
        model, tokenizer = train_bert_model()
        print("\n🎉 Training completed successfully!")
        print("📁 Model saved to './suicide_severity_model'")
        print("\n🔄 You can now restart the Streamlit app to use the trained model.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n⚠️ The app will use fallback rule-based prediction.")
        sys.exit(1)

if __name__ == "__main__":
    main() 