from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from typing import List, Dict, Union

class SentimentAnalyzer:
    def __init__(self):
        self.models = {
            'bert': pipeline("sentiment-analysis", device=-1 ,model="finiteautomata/bertweet-base-sentiment-analysis"),
            'distilbert': pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english", 
                device=-1  # -1 means force CPU, avoids GPU/meta-device issues
            ),
            'vader': SentimentIntensityAnalyzer()
        }
    
    def analyze_text(self, text: Union[str, List[str]]) -> List[Dict]:
        if isinstance(text, str):
            text = [text]
            
        results = []
        for t in text:
            model_results = {}
            for model_name, model in self.models.items():
                if model_name == 'vader':
                    scores = model.polarity_scores(t)
                    label = 'positive' if scores['compound'] >= 0.05 else \
                            'negative' if scores['compound'] <= -0.05 else 'neutral'
                    model_results[model_name] = {
                        'label': label,
                        'score': scores['compound'],
                        'details': scores
                    }
                else:
                    result = model(t)[0]
                    model_results[model_name] = {
                        'label': result['label'],
                        'score': result['score']
                    }
            
            # Ensemble decision
            labels = [r['label'] for r in model_results.values()]
            final_label = max(set(labels), key=labels.count)
            
            scores = []
            for r in model_results.values():
                score = r['score']
                if r['label'] == 'negative':
                    score = -score
                scores.append(score)
                
            final_score = np.mean(scores)
            
            results.append({
                'text': t,
                'label': final_label,
                'score': final_score,
                'model_details': model_results
            })
            
        return results
    
    def analyze_news(self, articles: List[Dict]) -> Dict:
        if not articles:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'articles_analyzed': 0
            }
            
        texts = [f"{a.get('title', '')}. {a.get('description', '')}" for a in articles]
        analyses = self.analyze_text(texts)
        
        return {
            'average_sentiment': np.mean([a['score'] for a in analyses]),
            'positive_count': sum(1 for a in analyses if a['score'] > 0.1),
            'negative_count': sum(1 for a in analyses if a['score'] < -0.1),
            'articles_analyzed': len(articles),
            'detailed_analysis': analyses
        }