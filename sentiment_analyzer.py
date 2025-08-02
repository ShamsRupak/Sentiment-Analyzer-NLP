#!/usr/bin/env python3
"""
Sentiment Analyzer - Movie Review Classification
===============================================
This script demonstrates sentiment analysis using Natural Language Processing (NLP)
and machine learning to classify movie reviews as positive or negative.

Author: Shams Rupak
GitHub: https://github.com/ShamsRupak
"""

import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import re
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("📥 Downloading required NLTK data...")
nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

class SentimentAnalyzer:
    """A sentiment analyzer for movie reviews using ML techniques."""
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type: Type of classifier to use ('naive_bayes' or 'logistic_regression')
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, lowercasing, and removing stopwords.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_movie_reviews(self) -> Tuple[List[str], List[int]]:
        """
        Load and preprocess movie reviews dataset.
        
        Returns:
            Tuple of (reviews, labels) where labels are 0 for negative, 1 for positive
        """
        print("\n📊 Loading movie reviews dataset...")
        
        # Load positive and negative reviews
        positive_reviews = [(movie_reviews.raw(fileid), 1) 
                           for fileid in movie_reviews.fileids('pos')]
        negative_reviews = [(movie_reviews.raw(fileid), 0) 
                           for fileid in movie_reviews.fileids('neg')]
        
        # Combine and shuffle
        all_reviews = positive_reviews + negative_reviews
        random.shuffle(all_reviews)
        
        # Separate reviews and labels
        reviews = [review for review, _ in all_reviews]
        labels = [label for _, label in all_reviews]
        
        print(f"✅ Loaded {len(reviews)} reviews ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        
        # Preprocess reviews
        print("🔧 Preprocessing text...")
        reviews = [self.preprocess_text(review) for review in reviews]
        
        return reviews, labels
    
    def train(self, X_train, y_train):
        """
        Train the sentiment analysis model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n🤖 Training {self.model_type.replace('_', ' ').title()} classifier...")
        
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        print("✅ Model training complete!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n📊 Evaluating model performance...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"🎯 Model Accuracy: {accuracy:.2%}")
        
        # Print detailed classification report
        print("\n📈 Classification Report:")
        print("-" * 50)
        target_names = ['Negative 😞', 'Positive 😊']
        print(classification_report(y_test, predictions, target_names=target_names))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print("🔢 Confusion Matrix:")
        print(f"   Predicted: Neg  Pos")
        print(f"Actual Neg:  {cm[0, 0]:4d} {cm[0, 1]:4d}")
        print(f"Actual Pos:  {cm[1, 0]:4d} {cm[1, 1]:4d}")
        
        return accuracy
    
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment of a given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform to features
        features = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        confidence = max(probability) * 100
        
        return sentiment, confidence

def interactive_mode(analyzer):
    """
    Run the analyzer in interactive mode for user input.
    """
    print("\n" + "="*60)
    print("🎬 INTERACTIVE SENTIMENT ANALYSIS MODE 🎬")
    print("="*60)
    print("Enter your movie review below (or type 'quit' to exit):")
    print("-"*60)
    
    while True:
        # Get user input
        review = input("\n💬 Your review: ").strip()
        
        if review.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using Sentiment Analyzer! Goodbye!")
            break
        
        if not review:
            print("⚠️  Please enter a valid review.")
            continue
        
        # Analyze sentiment
        sentiment, confidence = analyzer.predict_sentiment(review)
        
        # Display results with visual feedback
        print("\n" + "="*50)
        print(f"📊 ANALYSIS RESULTS")
        print("="*50)
        print(f"🎭 Sentiment: {sentiment}")
        print(f"💪 Confidence: {confidence:.1f}%")
        
        # Visual confidence bar
        bar_length = 30
        filled = int(bar_length * confidence / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"📊 [{bar}] {confidence:.1f}%")
        
        # Provide feedback based on confidence
        if confidence > 90:
            print("✨ Very confident prediction!")
        elif confidence > 70:
            print("👍 Fairly confident prediction.")
        else:
            print("🤔 Low confidence - the review might be ambiguous.")
        
        print("="*50)

def main():
    """
    Main function to run the sentiment analysis pipeline.
    """
    print("🎬 SENTIMENT ANALYZER - Movie Review Classification")
    print("=" * 60)
    print("Author: Shams Rupak | GitHub: @ShamsRupak")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n🔧 Initializing Sentiment Analyzer...")
    
    # Ask user to choose model type
    print("\n📚 Choose classifier type:")
    print("1. Naive Bayes (faster, good for text)")
    print("2. Logistic Regression (more accurate, slower)")
    
    choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip()
    model_type = 'logistic_regression' if choice == '2' else 'naive_bayes'
    
    analyzer = SentimentAnalyzer(model_type=model_type)
    
    # Load and preprocess data
    reviews, labels = analyzer.load_movie_reviews()
    
    # Split data
    print("\n✂️  Splitting data into train/test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Testing samples: {len(X_test)}")
    
    # Transform text to features
    print("\n🔤 Converting text to numerical features using TF-IDF...")
    X_train_features = analyzer.vectorizer.fit_transform(X_train)
    X_test_features = analyzer.vectorizer.transform(X_test)
    print(f"✅ Feature dimensions: {X_train_features.shape[1]}")
    
    # Train the model
    analyzer.train(X_train_features, y_train)
    
    # Evaluate the model
    accuracy = analyzer.evaluate(X_test_features, y_test)
    
    # Sample predictions
    print("\n🎬 SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Show some correct and incorrect predictions
    predictions = analyzer.model.predict(X_test_features)
    
    # Find examples of each case
    correct_pos = None
    correct_neg = None
    incorrect = None
    
    for i in range(len(y_test)):
        if predictions[i] == y_test[i] == 1 and correct_pos is None:
            correct_pos = i
        elif predictions[i] == y_test[i] == 0 and correct_neg is None:
            correct_neg = i
        elif predictions[i] != y_test[i] and incorrect is None:
            incorrect = i
        
        if all(x is not None for x in [correct_pos, correct_neg, incorrect]):
            break
    
    # Display examples
    examples = [
        ("✅ Correctly Classified Positive", correct_pos, "😊"),
        ("✅ Correctly Classified Negative", correct_neg, "😞"),
        ("❌ Misclassified Example", incorrect, "🤔")
    ]
    
    for title, idx, emoji in examples:
        if idx is not None:
            print(f"\n{title} {emoji}")
            print("-" * 40)
            # Show first 200 characters of the review
            review_text = X_test[idx][:200] + "..."
            print(f"Review: {review_text}")
            actual = "Positive" if y_test[idx] == 1 else "Negative"
            predicted = "Positive" if predictions[idx] == 1 else "Negative"
            print(f"Actual: {actual} | Predicted: {predicted}")
    
    # Run interactive mode
    interactive_mode(analyzer)

if __name__ == "__main__":
    main()
