import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from datetime import datetime, timedelta
import numpy as np
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import logging
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
STOCK_SYMBOLS = ["TSLA", "MSFT", "NVDA", "GOOG", "AMZN", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2025-07-08"
FRED_API_KEY = "your_fred_api_key"  # Replace with your FRED API key
OUTPUT_CSV = "stock_data.csv"
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./finetuned_model"

# Initialize FRED API
try:
    fred = Fred(api_key=FRED_API_KEY)
    logger.info("Initialized FRED API")
except Exception as e:
    logger.error(f"Error initializing FRED API: {e}")
    fred = None

def fetch_cpi_data():
    """Fetch CPI data from FRED for inflation adjustment."""
    if fred is None:
        logger.warning("FRED API not available; skipping CPI data")
        return None
    try:
        cpi = fred.get_series("CPIAUCSL", start_date=START_DATE, end_date=END_DATE)
        cpi = cpi.resample("M").last().ffill()
        cpi_df = pd.DataFrame(cpi, columns=["CPI"])
        cpi_df.index.name = "Date"
        return cpi_df
    except Exception as e:
        logger.error(f"Error fetching CPI data: {e}")
        return None

def fetch_stock_data(symbol):
    """Fetch historical price, dividend, and earnings data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE, interval="1mo")
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        df = df[["Close", "Dividends"]].copy()
        df.rename(columns={"Close": f"Price_{symbol}", "Dividends": f"Dividend_{symbol}"}, inplace=True)
        df.index = pd.to_datetime(df.index)
        
        try:
            earnings = ticker.financials.loc["Net Income"].mean() / ticker.info.get("sharesOutstanding", 1)
            df[f"Earnings_{symbol}"] = earnings
        except Exception:
            logger.warning(f"Earnings data unavailable for {symbol}; setting to 0")
            df[f"Earnings_{symbol}"] = 0.0
        
        return df
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def calculate_pe10(price, earnings):
    """Calculate PE10 (CAPE) ratio using price and average earnings."""
    if earnings == 0:
        return 0.0
    try:
        pe10 = price / earnings if earnings > 0 else 0.0
        return pe10
    except Exception as e:
        logger.warning(f"Error calculating PE10: {e}")
        return 0.0

def adjust_for_inflation(df, cpi_df, symbol):
    """Adjust prices for inflation using CPI data."""
    if cpi_df is None:
        logger.warning(f"CPI data unavailable for {symbol}; Real Price set to Price")
        df[f"Real_Price_{symbol}"] = df[f"Price_{symbol}"]
        return df
    
    try:
        cpi_df = cpi_df.reindex(df.index, method="ffill")
        latest_cpi = cpi_df["CPI"].iloc[-1]
        df[f"Real_Price_{symbol}"] = df[f"Price_{symbol}"] * (latest_cpi / cpi_df["CPI"])
        return df
    except Exception as e:
        logger.error(f"Error adjusting for inflation for {symbol}: {e}")
        df[f"Real_Price_{symbol}"] = df[f"Price_{symbol}"]
        return df

def create_dataset(symbols):
    """Create a combined dataset for all specified stocks/indices."""
    cpi_df = fetch_cpi_data()
    all_dfs = []
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")
        df = fetch_stock_data(symbol)
        if df is None or df.empty:
            logger.error(f"Skipping {symbol} due to data fetch failure")
            continue
        
        df = adjust_for_inflation(df, cpi_df, symbol)
        df[f"Return_{symbol}"] = df[f"Price_{symbol}"].pct_change(12) * 100
        df[f"Real_Return_{symbol}"] = df[f"Real_Price_{symbol}"].pct_change(12) * 100
        df[f"PE10_{symbol}"] = df.apply(lambda row: calculate_pe10(row[f"Price_{symbol}"], row[f"Earnings_{symbol}"]), axis=1)
        
        df[[f"Return_{symbol}", f"Real_Return_{symbol}", f"Dividend_{symbol}", f"Earnings_{symbol}", f"PE10_{symbol}"]] = \
            df[[f"Return_{symbol}", f"Real_Return_{symbol}", f"Dividend_{symbol}", f"Earnings_{symbol}", f"PE10_{symbol}"]].fillna(0.0)
        
        all_dfs.append(df)
    
    if not all_dfs:
        logger.error("No data fetched for any symbol")
        return None
    
    combined_df = all_dfs[0]
    for df in all_dfs[1:]:
        combined_df = combined_df.join(df, how="outer")
    
    combined_df.reset_index(inplace=True)
    return combined_df

def save_dataset(df, output_path):
    """Save dataset to CSV."""
    if df is not None:
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

# Step 1: Create and Save Dataset
logger.info(f"Creating dataset for {STOCK_SYMBOLS}")
df = create_dataset(STOCK_SYMBOLS)
if df is None:
    logger.error("Dataset creation failed")
    exit()
save_dataset(df, OUTPUT_CSV)

# Step 2: Preprocess Dataset for Training
df['Date'] = pd.to_datetime(df['Date'])
df_yearly = df.groupby(df['Date'].dt.year).mean().reset_index()
df_yearly = df_yearly.rename(columns={'Date': 'Year'})

# Step 3: Create Question-Answer Pairs
qa_pairs = []
years = df_yearly['Year'].unique()
min_year = int(years.min())
max_year = int(years.max())

for symbol in STOCK_SYMBOLS:
    for _, row in df_yearly.iterrows():
        year = int(row['Year'])
        price = row.get(f"Price_{symbol}", 0.0)
        dividend = row.get(f"Dividend_{symbol}", 0.0)
        earnings = row.get(f"Earnings_{symbol}", 0.0)
        return_val = row.get(f"Return_{symbol}", 0.0)
        real_return = row.get(f"Real_Return_{symbol}", 0.0)
        pe10 = row.get(f"PE10_{symbol}", 0.0)
        
        symbol_name = "S&P 500" if symbol == "SPY" else symbol
        
        qa_pairs.append({
            "question": f"What was the {symbol_name} return in {year}?",
            "answer": f"The {symbol_name} returned approximately {return_val:.1f}% in {year}, including dividends."
        })
        qa_pairs.append({
            "question": f"What was the {symbol_name} price in {year}?",
            "answer": f"The {symbol_name} averaged approximately {price:.2f} in {year}."
        })
        qa_pairs.append({
            "question": f"What was the {symbol_name} real return in {year}?",
            "answer": f"The {symbol_name} inflation-adjusted return was approximately {real_return:.1f}% in {year}."
        })
        if dividend > 0:
            qa_pairs.append({
                "question": f"What was the {symbol_name} dividend in {year}?",
                "answer": f"The {symbol_name} dividend was approximately {dividend:.2f} in {year}."
            })
        if earnings > 0:
            qa_pairs.append({
                "question": f"What were the {symbol_name} earnings in {year}?",
                "answer": f"The {symbol_name} earnings were approximately {earnings:.2f} in {year}."
            })
        if pe10 > 0:
            qa_pairs.append({
                "question": f"What was the {symbol_name} PE10 ratio in {year}?",
                "answer": f"The {symbol_name} PE10 ratio was approximately {pe10:.2f} in {year}."
            })
        qa_pairs.append({
            "summary": f"In {year}, the {symbol_name} averaged {price:.2f} with a {return_val:.1f}% annual return and a {real_return:.1f}% real return."
        })

    # Period-specific questions
    for start_year, end_year in itertools.combinations(years, 2):
        if start_year < end_year:
            df_period = df_yearly[(df_yearly['Year'] >= start_year) & (df_yearly['Year'] <= end_year)]
            if not df_period.empty:
                avg_return = df_period[f"Return_{symbol}"].mean()
                avg_real_return = df_period[f"Real_Return_{symbol}"].mean()
                duration = end_year - start_year + 1
                qa_pairs.append({
                    "question": f"What was the average annual growth rate of {symbol_name} between {start_year} and {end_year}?",
                    "answer": f"The {symbol_name} average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                })
                qa_pairs.append({
                    "question": f"What was the average annual return of {symbol_name} between {start_year} and {end_year}?",
                    "answer": f"The {symbol_name} average annual return from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                })
                qa_pairs.append({
                    "question": f"What was the {symbol_name} real return between {start_year} and {end_year}?",
                    "answer": f"The {symbol_name} average annual inflation-adjusted return from {start_year} to {end_year} was approximately {avg_real_return:.1f}%."
                })
                qa_pairs.append({
                    "question": f"What was the {duration}-year average annual growth rate of {symbol_name} from {start_year}?",
                    "answer": f"The {symbol_name} {duration}-year average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                })

    # Past X years questions with more variations to reduce hallucinations
    for duration in range(1, max_year - min_year + 2):
        for end_year in years:
            start_year = end_year - duration + 1
            if start_year >= min_year:
                df_period = df_yearly[(df_yearly['Year'] >= start_year) & (df_yearly['Year'] <= end_year)]
                if not df_period.empty:
                    avg_return = df_period[f"Return_{symbol}"].mean()
                    avg_real_return = df_period[f"Real_Return_{symbol}"].mean()
                    qa_pairs.append({
                        "question": f"What was the average annual growth rate of {symbol_name} in the past {duration} years from {end_year}?",
                        "answer": f"The {symbol_name} average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                    })
                    qa_pairs.append({
                        "question": f"What was the {duration}-year average annual growth rate of {symbol_name} ending in {end_year}?",
                        "answer": f"The {symbol_name} {duration}-year average annual growth rate from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                    })
                    qa_pairs.append({
                        "question": f"What is the average return of {symbol_name} over the last {duration} years?",
                        "answer": f"The average annual return of {symbol_name} from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
                    })
                    qa_pairs.append({
                        "question": f"What was {symbol_name}'s performance in the past {duration} years?",
                        "answer": f"{symbol_name} had an average annual return of approximately {avg_return:.1f}% from {start_year} to {end_year}, including dividends."
                    })
                    qa_pairs.append({
                        "question": f"Calculate the average annual return for {symbol_name} in the last {duration} years.",
                        "answer": f"The calculated average annual return for {symbol_name} from {start_year} to {end_year} is approximately {avg_return:.1f}%, including dividends."
                    })

# Investment return questions
amounts = [1000, 5000, 10000]
durations = [1, 3, 5, 10, 20]
avg_annual_return = 10.0
for symbol in STOCK_SYMBOLS:
    symbol_name = "S&P 500" if symbol == "SPY" else symbol
    for amount in amounts:
        for n in durations:
            future_value = amount * (1 + avg_annual_return / 100) ** n
            qa_pairs.append({
                "question": f"What will ${amount} be worth in {n} years if invested in {symbol_name}?",
                "answer": f"Assuming a 10% average annual return, ${amount:,.0f} invested in {symbol_name} would grow to approximately ${future_value:,.0f} in {n} years with annual compounding."
            })

# General questions
for symbol in STOCK_SYMBOLS:
    symbol_name = "S&P 500" if symbol == "SPY" else symbol
    qa_pairs.append({
        "question": f"What is the average return rate of {symbol_name} in the past 10 years?",
        "answer": f"The {symbol_name} average annual return rate from {max_year-10} to {max_year} was approximately {df_yearly[(df_yearly['Year'] >= max_year-10) & (df_yearly['Year'] <= max_year)][f'Return_{symbol}'].mean():.1f}%, including dividends."
    })
    qa_pairs.append({
        "question": f"What is the average return rate of {symbol_name} in the last 5 years?",
        "answer": f"The {symbol_name} average annual return rate from {max_year-5} to {max_year} was approximately {df_yearly[(df_yearly['Year'] >= max_year-5) & (df_yearly['Year'] <= max_year)][f'Return_{symbol}'].mean():.1f}%, including dividends."
    })
    qa_pairs.append({
        "question": f"What is the average return rate of {symbol_name} in the past 7 years?",
        "answer": f"The {symbol_name} average annual return rate from {max_year-7} to {max_year} was approximately {df_yearly[(df_yearly['Year'] >= max_year-7) & (df_yearly['Year'] <= max_year)][f'Return_{symbol}'].mean():.1f}%, including dividends."
    })
qa_pairs.append({
    "question": "What is the average growth rate for stocks?",
    "answer": "The average annual return for individual stocks varies widely, but broad market indices like the S&P 500 average 10–12% over the long term (1927–2025), including dividends. Specific stocks like TSLA or NVDA may have higher volatility and returns."
})

# Save to JSON
with open("financial_data.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

# Step 4: Load and Tokenize Dataset
dataset = Dataset.from_json("financial_data.json")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)["train"]
test_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)["test"]

# Step 5: Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

def tokenize_function(examples):
    inputs = []
    for ex in zip(examples.get("question", []), examples.get("answer", []), examples.get("summary", [])):
        if ex[0] and ex[1]:
            inputs.append(ex[0] + " A: " + ex[1])
        elif ex[2]:
            inputs.append(ex[2])
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Step 6: Load and Fine-Tune Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Step 7: Train and Evaluate
trainer.train()
eval_results = trainer.evaluate(tokenized_test)
logger.info(f"Evaluation results: {eval_results}")

# Step 8: Save Model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Model and tokenizer saved to {OUTPUT_DIR}") 
