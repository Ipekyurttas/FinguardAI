import pandas as pd
from datasets import Dataset
from src.components.model_trainer import get_model_and_tokenizer, compute_metrics
from transformers import TrainingArguments, Trainer

print("FinGuard AI: Sentiment Model Fine-Tuning Başlatıldı...")

df = pd.read_csv('data/processed/processed_financial_sentiment.csv')

df = df.dropna(subset=['processed_content'])
df = df[df['processed_content'].str.len() > 2]

label_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
if df['Sentiment'].dtype == 'object':
    df['label'] = df['Sentiment'].str.lower().map(label_dict)
else:
    df['label'] = df['Sentiment'] 

df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

dataset = Dataset.from_pandas(df[['processed_content', 'label']])
dataset = dataset.rename_column("processed_content", "text")
dataset = dataset.train_test_split(test_size=0.2)


model, tokenizer = get_model_and_tokenizer()

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


trainer.train()


model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

print("Eğitim Tamamlandı. Model 'models/sentiment_model' klasörüne kaydedildi.")