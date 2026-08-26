import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class BertPredictor:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"BERT Predictor using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise e

    def predict(self, texts, batch_size=32):
        all_preds = []
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                
        return all_preds
