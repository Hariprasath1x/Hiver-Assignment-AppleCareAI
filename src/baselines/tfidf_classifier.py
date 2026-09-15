from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class TfidfIntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        self.is_trained = False
        
    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        
    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        
        best_idx = np.argmax(probs)
        intent = self.model.classes_[best_idx]
        confidence = probs[best_idx]
        
        return intent, confidence
