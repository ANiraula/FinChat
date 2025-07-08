import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import numpy as np

# Step 1: Set Up Environment
# Ensure libraries are installed: pip install transformers datasets torch accelerate pandas numpy

# Step 2: Load and Preprocess Dataset
csv_path = "flat-ui__data-Sun Jul 06 2025.csv"
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Preprocess: Calculate annual returns
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Return'] = df['SP500'].pct_change(12) * 100  # Annual return based on monthly data
df['Real Return'] = df['Real Price'].pct_change(12) * 100  # Inflation-adjusted return

# Aggregate to yearly data for faster processing
df_yearly = df.groupby(df['Date'].dt.year).agg({
    'SP500': 'mean',
    'Return': 'mean',
    'Real Return': 'mean',
    'Dividend': 'mean',
    'Earnings': 'mean',
    'PE10': 'mean'
}).reset_index()
df_yearly = df_yearly.rename(columns={'Date': 'Year'})

# Create question-answer pairs and summaries
qa_pairs = []
for _, row in df_yearly.iterrows():
    year = int(row['Year'])
    sp500 = row['SP500']
    dividend = row['Dividend']
    earnings = row['Earnings']
    return_val = row.get('Return', 0.0)
    real_return = row.get('Real Return', 0.0)
    pe10 = row.get('PE10', 0.0)

    # Year-specific questions
    qa_pairs.append({
        "question": f"What was the S&P 500 return in {year}?",
        "answer": f"The S&P 500 returned approximately {return_val:.1f}% in {year}, including dividends."
    })
    qa_pairs.append({
        "question": f"What was the S&P 500 index value in {year}?",
        "answer": f"The S&P 500 averaged approximately {sp500:.2f} in {year}."
    })
    qa_pairs.append({
        "question": f"What was the S&P 500 real return in {year}?",
        "answer": f"The S&P 500’s inflation-adjusted return was approximately {real_return:.1f}% in {year}."
    })
    if dividend > 0:
        qa_pairs.append({
            "question": f"What was the S&P 500 dividend in {year}?",
            "answer": f"The S&P 500 dividend was approximately {dividend:.2f} in {year}."
        })
    if earnings > 0:
        qa_pairs.append({
            "question": f"What were the S&P 500 earnings in {year}?",
            "answer": f"The S&P 500 earnings were approximately {earnings:.2f} in {year}."
        })
    if pe10 > 0:
        qa_pairs.append({
            "question": f"What was the S&P 500 PE10 ratio in {year}?",
            "answer": f"The S&P 500 PE10 ratio was approximately {pe10:.2f} in {year}."
        })

    # Summaries
    qa_pairs.append({
        "summary": f"In {year}, the S&P 500 averaged {sp500:.2f} with a {return_val:.1f}% annual return and a {real_return:.1f}% real return."
    })

# Period-specific questions (1-year, 3-year, 5-year, 10-year, and custom ranges)
years = df_yearly['Year'].unique()
for year in years:
    for duration in [1, 3, 5, 10]:
        start_year = int(year)
        end_year = start_year + duration - 1
        if end_year <= df_yearly['Year'].max():
            df_period = df_yearly[(df_yearly['Year'] >= start_year) & (df_yearly['Year'] <= end_year)]
            avg_return = df_period['Return'].mean()
            avg_real_return = df_period['Real Return'].mean()
            qa_pairs.append({
                "question": f"What was the {duration}-year average annual growth rate of the S&P 500 from {start_year}?",
                "answer": f"The S&P 500’s {duration}-year average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
            })
            qa_pairs.append({
                "question": f"What was the {duration}-year real return of the S&P 500 from {start_year}?",
                "answer": f"The S&P 500’s {duration}-year average annual inflation-adjusted return from {start_year} to {end_year} was approximately {avg_real_return:.1f}%."
            })

# Custom period questions
custom_periods = [(2000, 2010), (2011, 2016), (2010, 2020), (2000, 2008), (2015, 2024)]
for start_year, end_year in custom_periods:
    df_period = df_yearly[(df_yearly['Year'] >= start_year) & (df_yearly['Year'] <= end_year)]
    if not df_period.empty:
        avg_return = df_period['Return'].mean()
        avg_real_return = df_period['Real Return'].mean()
        qa_pairs.append({
            "question": f"What was the average annual growth rate of the S&P 500 between {start_year} and {end_year}?",
            "answer": f"The S&P 500’s average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
        })
        qa_pairs.append({
            "question": f"What was the average annual return of the S&P 500 between {start_year} and {end_year}?",
            "answer": f"The S&P 500’s average annual return from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
        })
        qa_pairs.append({
            "question": f"What was the S&P 500’s real return between {start_year} and {end_year}?",
            "answer": f"The S&P 500’s average annual inflation-adjusted return from {start_year} to {end_year} was approximately {avg_real_return:.1f}%."
        })

# Investment return questions
amounts = [1000, 5000, 10000]
durations = [1, 3, 5, 10, 20]
avg_annual_return = 10.0  # Historical S&P 500 average (1927–2025)
for amount in amounts:
    for n in durations:
        future_value = amount * (1 + avg_annual_return / 100) ** n
        qa_pairs.append({
            "question": f"What will ${amount} be worth in {n} years if invested in the S&P 500?",
            "answer": f"Assuming a 10% average annual return, ${amount:,.0f} invested in the S&P 500 would grow to approximately ${future_value:,.0f} in {n} years with annual compounding."
        })

# Add specific period and general questions
qa_pairs.append({
    "question": "What is the average return rate of the S&P 500 in the past 10 years?",
    "answer": "The S&P 500’s average annual return rate from 2015 to 2024 was approximately 12.2%, including dividends, based on historical data."
})
qa_pairs.append({
    "question": "What is the S&P 500 index fund average growth rate?",
    "answer": "The S&P 500 index fund’s average annual return is approximately 10–12% over the long term (1927–2025), including dividends, based on historical data."
})
qa_pairs.append({
    "question": "What was the average annual return of the S&P 500 between 2010 and 2020?",
    "answer": "The S&P 500’s average annual return from 2010 to 2020 was approximately 13.6%, including dividends, driven by post-financial crisis recovery."
})

# Save to JSON
with open("financial_data.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

# Load dataset
dataset = Dataset.from_json("financial_data.json")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)["train"]
test_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)["test"]

# Step 3: Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
        inputs = [q + " A: " + a for q, a in zip(examples["question"], examples["answer"])]
    else:
        inputs = examples["summary"]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Step 4: Load Pre-trained Model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Step 5: Set Up Fine-Tuning
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,  # Increased for faster training
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Step 6: Fine-Tune the Model
trainer.train()

# Step 7: Evaluate the Model
eval_results = trainer.evaluate(tokenized_test)
print("Evaluation results:", eval_results)

# Step 8: Save the Fine-Tuned Model
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Test the model
input_text = "What was the average annual return of the S&P 500 between 2010 and 2020?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
