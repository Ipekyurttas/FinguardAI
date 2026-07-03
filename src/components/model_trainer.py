from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    """Accuracy ve F1-Score metriklerini hesaplar."""
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": acc, "f1": f1}

def get_model_and_tokenizer(model_name="ProsusAI/finbert"): 
    """FinBERT modelini ve ilgili tokenizer'ı yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return model, tokenizer