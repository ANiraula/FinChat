#Loading packages
import logging
import os
import time
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re
import numpy as np
import json
import difflib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define device (force CPU for Spaces free tier)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Load dataset and precompute period data
csv_path = "flat-ui__data-Sun Jul 06 2025.csv"
try:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df['SP500'].pct_change(12) * 100
    df['Real Return'] = df['Real Price'].pct_change(12) * 100
    logger.info("Loaded dataset successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    df = None

# Precompute yearly aggregates for faster lookups
if df is not None:
    df_yearly = df.groupby(df['Date'].dt.year).agg({
        'SP500': 'mean',
        'Return': 'mean',
        'Real Return': 'mean',
        'Dividend': 'mean',
        'Earnings': 'mean',
        'PE10': 'mean'
    }).reset_index()
    df_yearly = df_yearly.rename(columns={'Date': 'Year'})
else:
    df_yearly = None

# Response cache with financial data entries
response_cache = {
    "hi": "Hello! I'm FinChat, your financial advisor. How can I help with investing?",
    "hello": "Hello! I'm FinChat, your financial advisor. How can I help with investing?",
    "hey": "Hi there! Ready to discuss investment goals with FinChat?",
    "what is better individual stocks or etfs?": (
        "Here’s a comparison of individual stocks vs. ETFs:\n"
        "1. **Individual Stocks**: High returns possible (e.g., Apple up 80% in 2020) but riskier due to lack of diversification. Require active research.\n"
        "2. **ETFs**: Diversify risk by tracking indices (e.g., SPY, S&P 500, ~12% avg. return 2015–2024). Lower fees and less research needed.\n"
        "3. **Recommendation**: Beginners should start with ETFs; experienced investors may add stocks.\n"
        "Consult a financial planner."
    ),
    "is $100 per month enough to invest?": (
        "Yes, $100 per month is enough to start investing. Here’s why and how:\n"
        "1. **Feasibility**: Brokerages like Fidelity have no minimums, and commission-free trading eliminates fees.\n"
        "2. **Options**: Buy fractional shares of ETFs (e.g., SPY, ~$622/share in 2025) with $100.\n"
        "3. **Strategy**: Use dollar-cost averaging to invest monthly, reducing market timing risks.\n"
        "4. **Growth**: At 10% annual return, $100 monthly could grow to ~$41,000 in 20 years.\n"
        "5. **Tips**: Ensure an emergency fund; diversify.\n"
        "Consult a financial planner."
    ),
    "can i invest $100 a month?": (
        "Yes, $100 a month is sufficient. Here’s how:\n"
        "1. **Brokerage**: Open an account with Fidelity or Vanguard (no minimums).\n"
        "2. **Investments**: Buy fractional shares of ETFs like SPY ($100 buys ~0.16 shares in 2025).\n"
        "3. **Approach**: Use dollar-cost averaging for steady growth.\n"
        "4. **Long-Term**: At 10% return, $100 monthly could reach ~$41,000 in 20 years.\n"
        "5. **Tips**: Prioritize an emergency fund and diversify.\n"
        "Consult a financial planner."
    ),
    "hi, give me step-by-step investing advice": (
        "Here’s a step-by-step guide to start investing:\n"
        "1. Open a brokerage account (e.g., Fidelity, Vanguard) if 18 or older.\n"
        "2. Deposit an affordable amount, like $100, after an emergency fund.\n"
        "3. Research and buy an ETF (e.g., SPY) using Yahoo Finance.\n"
        "4. Monitor monthly and enable dividend reinvesting.\n"
        "5. Use dollar-cost averaging ($100 monthly) to reduce risk.\n"
        "6. Diversify across sectors.\n"
        "Consult a financial planner."
    ),
    "hi, pretend you are a financial advisor. now tell me how can i start investing in stock market?": (
        "Here’s a guide to start investing:\n"
        "1. Learn from Investopedia or 'The Intelligent Investor.'\n"
        "2. Set goals (e.g., retirement) and assess risk.\n"
        "3. Choose a brokerage (Fidelity, Vanguard).\n"
        "4. Start with ETFs (e.g., SPY) or mutual funds.\n"
        "5. Use dollar-cost averaging ($100-$500 monthly).\n"
        "6. Diversify and monitor.\n"
        "Consult a financial planner."
    ),
    "do you have a list of companies you recommend?": (
        "I can’t recommend specific companies without data. Try ETFs like SPY (S&P 500, ~12% avg. return 2015–2024) or QQQ (tech). "
        "Research stocks like Apple (AAPL, ~80% return in 2020) or Johnson & Johnson on Yahoo Finance.\n"
        "Consult a financial planner."
    ),
    "how do i start investing in stocks?": (
        "Learn from Investopedia. Set goals and assess risk. Open a brokerage account (Fidelity, Vanguard) "
        "and start with ETFs (e.g., SPY, ~12% avg. return 2015–2024). Consult a financial planner."
    ),
    "what's the difference between stocks and bonds?": (
        "Stocks are company ownership with high risk and growth potential (e.g., S&P 500 ~12% avg. return 2015–2024). Bonds are loans to companies/governments "
        "with lower risk and steady interest. Diversify for balance."
    ),
    "how much should i invest?": (
        "Invest what you can afford after expenses and an emergency fund. Start with $100-$500 monthly "
        "in ETFs like SPY (~12% avg. return 2015–2024). Consult a financial planner."
    ),
    "what is dollar-cost averaging?": (
        "Dollar-cost averaging is investing a fixed amount regularly (e.g., $100 monthly) in ETFs, "
        "reducing risk by spreading purchases over time."
    ),
    "give me few investing idea": (
        "Here are investing ideas:\n"
        "1. Open a brokerage account (e.g., Fidelity) if 18 or older.\n"
        "2. Deposit $100 or what you can afford.\n"
        "3. Buy a researched ETF (e.g., SPY, ~12% avg. return 2015–2024) or index fund.\n"
        "4. Check regularly and enable dividend reinvesting.\n"
        "5. Use dollar-cost averaging (e.g., monthly buys).\n"
        "Consult a financial planner."
    ),
    "give me investing tips": (
        "Here are investing tips:\n"
        "1. Educate yourself with Investopedia or books.\n"
        "2. Open a brokerage account (e.g., Vanguard).\n"
        "3. Start small with ETFs like SPY (~12% avg. return 2015–2024).\n"
        "4. Invest regularly using dollar-cost averaging.\n"
        "5. Diversify to manage risk.\n"
        "Consult a financial planner."
    ),
    "how to start investing": (
        "Here’s how to start investing:\n"
        "1. Study basics on Investopedia.\n"
        "2. Open a brokerage account (e.g., Fidelity).\n"
        "3. Deposit $100 or more after securing savings.\n"
        "4. Buy an ETF like SPY (~12% avg. return 2015–2024) after research.\n"
        "5. Invest monthly with dollar-cost averaging.\n"
        "Consult a financial planner."
    ),
    "investing advice": (
        "Here’s investing advice:\n"
        "1. Learn basics from Investopedia.\n"
        "2. Open a brokerage account (e.g., Vanguard).\n"
        "3. Start with $100 in an ETF like SPY (~12% avg. return 2015–2024).\n"
        "4. Use dollar-cost averaging for regular investments.\n"
        "5. Monitor and diversify your portfolio.\n"
        "Consult a financial planner."
    ),
    "steps to invest": (
        "Here are steps to invest:\n"
        "1. Educate yourself using Investopedia.\n"
        "2. Open a brokerage account (e.g., Fidelity).\n"
        "3. Deposit an initial $100 after savings.\n"
        "4. Buy an ETF like SPY (~12% avg. return 2015–2024) after research.\n"
        "5. Use dollar-cost averaging monthly.\n"
        "Consult a financial planner."
    ),
    "what is the s&p 500 index fund average growth rate?": (
        "The S&P 500 index fund’s average annual return is approximately 10–12% over the long term (1927–2025), including dividends, based on historical data. "
        "For example, from 2015 to 2024, it averaged ~12.2% annually. Returns vary yearly due to market conditions. Consult a financial planner."
    ),
    "what was the s&p 500 return in 2020?": (
        "The S&P 500 returned approximately 16.3% in 2020, including dividends, driven by recovery from the COVID-19 market crash."
    ),
    "what was the s&p 500 return in 2022?": (
        "The S&P 500 returned approximately -18.1% in 2022, impacted by high inflation and interest rate hikes."
    ),
    "what is the average annual growth rate of the s&p 500 from 2000 to 2010?": (
        "The S&P 500’s average annual growth rate from 2000 to 2010 was approximately 0.4%, including dividends, impacted by the dot-com crash and 2008 financial crisis."
    ),
    "what is the average annual growth rate of the s&p 500 from 2011 to 2016?": (
        "The S&P 500’s average annual growth rate from 2011 to 2016 was approximately 12.7%, including dividends, driven by post-financial crisis recovery."
    ),
    "what was the average annual return of the s&p 500 between 2010 and 2020?": (
        "The S&P 500’s average annual return from 2010 to 2020 was approximately 13.6%, including dividends, driven by post-financial crisis recovery."
    ),
    "what will my return be in 10 years if i invest $5000 into s&p 500 right now?": (
        "Assuming a 10% average annual return, a $5,000 investment in the S&P 500 would grow to approximately $12,974 in 10 years with annual compounding. "
        "This is based on the historical average return of 10–12% (1927–2025). Future returns vary and are not guaranteed. Consult a financial planner."
    ),
    "what was the 1-year average annual growth rate of the s&p 500 from 2020?": (
        "The S&P 500 returned approximately 16.3% in 2020, including dividends, driven by recovery from the COVID-19 market crash."
    ),
    "what was the 3-year average annual growth rate of the s&p 500 from 2018?": (
        "The S&P 500’s average annual growth rate from 2018 to 2020 was approximately 10.2%, including dividends, based on historical data."
    ),
    "what was the 5-year average annual growth rate of the s&p 500 from 2016?": (
        "The S&P 500’s average annual growth rate from 2016 to 2020 was approximately 13.6%, including dividends, driven by strong market recovery."
    ),
    "what is the average return rate of the s&p 500 in the past 10 years?": (
        "The S&P 500’s average annual return rate from 2015 to 2024 was approximately 12.2%, including dividends, based on historical data."
    )
}

# Load persistent cache
cache_file = "cache.json"
try:
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            response_cache.update(json.load(f))
        logger.info("Loaded persistent cache from cache.json")
except Exception as e:
    logger.warning(f"Failed to load cache.json: {e}")

# Load model and tokenizer (use fine-tuned model if available)
model_name = "./finetuned_model" if os.path.exists("./finetuned_model") else "distilgpt2"
try:
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loading model {model_name}")
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
    logger.info(f"Successfully loaded model: {model_name}")
except Exception as e:
    logger.error(f"Error loading model/tokenizer: {e}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Pre-tokenize prompt prefix
prompt_prefix = (
    "You are FinChat, a financial advisor with expertise in stock market performance. Provide detailed, numbered list advice with clear reasoning for investing prompts, "
    "including precise historical data when relevant (e.g., S&P 500 returns for specific years or periods). For investment return queries, use compound interest calculations "
    "based on historical averages. Avoid repetition and incomplete answers. Explain why each step or choice is beneficial.\n\n"
    "Example 1:\n"
    "Q: What is the S&P 500’s average annual return?\n"
    "A: The S&P 500’s average annual return is ~10–12% over the long term (1927–2025), including dividends.\n"
    "1. This reflects historical data adjusted for inflation and dividends.\n"
    "2. Returns vary yearly (e.g., 16.3% in 2020) due to market conditions.\n"
    "3. ETFs like SPY track this index for broad market exposure.\n\n"
    "Example 2:\n"
    "Q: What will $5,000 be worth in 10 years if invested in the S&P 500?\n"
    "A: Assuming a 10% average annual return, a $5,000 investment in the S&P 500 would grow to approximately $12,974 in 10 years with annual compounding.\n"
    "1. This uses the historical average return of 10–12% (1927–2025).\n"
    "2. Future returns vary and are not guaranteed.\n\n"
    "Example 3:\n"
    "Q: What was the average annual return of the S&P 500 between 2010 and 2020?\n"
    "A: The S&P 500’s average annual return from 2010 to 2020 was approximately 13.6%, including dividends.\n"
    "1. This period includes strong recovery post-financial crisis.\n"
    "2. Dividends contribute significantly to total returns.\n\n"
    "Q: "
)
prefix_tokens = tokenizer(prompt_prefix, return_tensors="pt", truncation=True, max_length=512).to(device)

# Substring matching for cache with fuzzy matching
def get_closest_cache_key(message, cache_keys):
    message = message.lower().strip()
    matches = difflib.get_close_matches(message, cache_keys, n=1, cutoff=0.8)
    return matches[0] if matches else None

# Parse period from user input
def parse_period(query):
    # Match specific year ranges (e.g., "between 2010 and 2020", "2000–2008")
    match = re.search(r'(?:between|from)\s*(\d{4})\s*(?:and|to|-|–)\s*(\d{4})', query, re.IGNORECASE)
    if match:
        start_year, end_year = map(int, match.groups())
        return start_year, end_year, None
    # Match duration-based queries (e.g., "1-year from 2020", "3-year growth rate")
    match = re.search(r'(\d+)-year.*from\s*(\d{4})', query, re.IGNORECASE)
    if match:
        duration, start_year = map(int, match.groups())
        end_year = start_year + duration - 1
        return start_year, end_year, duration
    # Match general duration queries (e.g., "past 10 years", "3-year growth rate")
    match = re.search(r'past\s*(\d+)-year|\b(\d+)-year.*(?:return|growth\s*rate)', query, re.IGNORECASE)
    if match:
        duration = int(match.group(1) or match.group(2))
        max_year = df_yearly['Year'].max() if df_yearly is not None else 2025
        start_year = max_year - duration + 1
        end_year = max_year
        return start_year, end_year, duration
    return None, None, None

# Calculate average growth rate
def calculate_growth_rate(start_year, end_year, duration=None):
    if df_yearly is None or start_year is None or end_year is None:
        return None, "Data not available or invalid period."
    df_period = df_yearly[(df_yearly['Year'] >= start_year) & (df_yearly['Year'] <= end_year)]
    if df_period.empty:
        return None, f"No data available for {start_year} to {end_year}."
    avg_return = df_period['Return'].mean()
    if duration:
        response = f"The S&P 500’s {duration}-year average annual return from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
    else:
        response = f"The S&P 500’s average annual return from {start_year} to {end_year} was approximately {avg_return:.1f}%, including dividends."
    return avg_return, response

# Parse investment return query
def parse_investment_query(query):
    match = re.search(r'\$(\d+).*\s(\d+)\s*years?.*\bs&p\s*500', query, re.IGNORECASE)
    if match:
        amount = float(match.group(1))
        years = int(match.group(2))
        return amount, years
    return None, None

# Calculate future value
def calculate_future_value(amount, years):
    if df_yearly is None or amount is None or years is None:
        return None, "Data not available or invalid input."
    avg_annual_return = 10.0  # Historical S&P 500 average (1927–2025)
    future_value = amount * (1 + avg_annual_return / 100) ** years
    return future_value, (
        f"Assuming a 10% average annual return, a ${amount:,.0f} investment in the S&P 500 would grow to approximately ${future_value:,.0f} "
        f"in {years} years with annual compounding. This is based on the historical average return of 10–12% (1927–2025). "
        "Future returns vary and are not guaranteed. Consult a financial planner."
    )

# Define chat function
def chat_with_model(user_input, history=None, is_processing=False):
    try:
        start_time = time.time()
        logger.info(f"Processing user input: {user_input}")
        is_processing = True
        logger.info("Showing loading animation")

        # Normalize and check cache
        cache_key = user_input.lower().strip()
        cache_keys = list(response_cache.keys())
        closest_key = cache_key if cache_key in response_cache else get_closest_cache_key(cache_key, cache_keys)
        if closest_key:
            logger.info(f"Cache hit for: {closest_key}")
            response = response_cache[closest_key]
            logger.info(f"Chatbot response: {response}")
            history = history or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            end_time = time.time()
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            return response, history, False, ""

        # Check for investment return query
        amount, years = parse_investment_query(user_input)
        if amount and years:
            future_value, response = calculate_future_value(amount, years)
            if future_value is not None:
                response_cache[cache_key] = response
                logger.info(f"Investment query: ${amount} for {years} years, added to cache")
                logger.info(f"Chatbot response: {response}")
                history = history or []
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                end_time = time.time()
                logger.info(f"Response time: {end_time - start_time:.2f} seconds")
                return response, history, False, ""

        # Check for period-specific query
        start_year, end_year, duration = parse_period(user_input)
        if start_year and end_year:
            avg_return, response = calculate_growth_rate(start_year, end_year, duration)
            if avg_return is not None:
                response_cache[cache_key] = response
                logger.info(f"Dynamic period query: {start_year}–{end_year}, added to cache")
                logger.info(f"Chatbot response: {response}")
                history = history or []
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                end_time = time.time()
                logger.info(f"Response time: {end_time - start_time:.2f} seconds")
                return response, history, False, ""

        # Skip model for short prompts
        if len(user_input.strip()) <= 5:
            logger.info("Short prompt, returning default response")
            response = "Hello! I'm FinChat, your financial advisor. Ask about investing!"
            logger.info(f"Chatbot response: {response}")
            history = history or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            end_time = time.time()
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            return response, history, False, ""

        # Construct prompt
        full_prompt = prompt_prefix + user_input + "\nA:"
        try:
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        except Exception as e:
            logger.error(f"Error tokenizing input: {e}")
            response = f"Error: Failed to process input: {str(e)}"
            logger.info(f"Chatbot response: {response}")
            history = history or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            end_time = time.time()
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            return response, history, False, ""

        # Generate response
        with torch.inference_mode():
            logger.info("Generating response with model")
            gen_start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,  # Reduced for faster inference
                min_length=20,
                do_sample=False,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_end_time = time.time()
            logger.info(f"Generation time: {gen_end_time - gen_start_time:.2f} seconds")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(full_prompt):].strip() if response.startswith(full_prompt) else response
        logger.info(f"Chatbot response: {response}")

        # Update cache
        response_cache[cache_key] = response
        logger.info("Cache miss, added to in-memory cache")

        # Update history
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info(f"Response time: {end_time - start_time:.2f} seconds")
        return response, history, False, ""

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = f"Error: {str(e)}"
        logger.info(f"Chatbot response: {response}")
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        end_time = time.time()
        logger.info(f"Response time: {end_time - start_time:.2f} seconds")
        return response, history, False, ""

# Save cache on exit
def save_cache():
    try:
        with open(cache_file, 'w') as f:
            json.dump(response_cache, f, indent=2)
        logger.info("Saved cache to cache.json")
    except Exception as e:
        logger.warning(f"Failed to save cache.json: {e}")

# Create Gradio interface with loading animation
logger.info("Initializing Gradio interface")
try:
    with gr.Blocks(
        title="FinChat: An LLM based on distilgpt2 model",
        css="""
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
            display: block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hidden { display: none; }
        """
    ) as interface:
        gr.Markdown(
            """
            # FinChat: An LLM based on distilgpt2 model
            FinChat provides financial advice using the lightweight distilgpt2 model, optimized for fast, detailed responses.
            Ask about investing strategies, ETFs, stocks, or budgeting to get started!
            """
        )
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Your message")
        submit = gr.Button("Send")
        clear = gr.Button("Clear")
        loading = gr.HTML('<div class="loader hidden"></div>', label="Loading")
        is_processing = gr.State(value=False)

        def submit_message(user_input, history, is_processing):
            response, updated_history, new_processing, clear_input = chat_with_model(user_input, history, is_processing)
            loader_html = '<div class="loader"></div>' if new_processing else '<div class="loader hidden"></div>'
            return clear_input, updated_history, loader_html, new_processing

        submit.click(
            fn=submit_message,
            inputs=[msg, chatbot, is_processing],
            outputs=[msg, chatbot, loading, is_processing]
        )
        clear.click(
            fn=lambda: ("", [], '<div class="loader hidden"></div>', False),
            outputs=[msg, chatbot, loading, is_processing]
        )
    logger.info("Gradio interface initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gradio interface: {e}")
    raise

# Launch interface (conditional for Spaces)
if __name__ == "__main__" and not os.getenv("HF_SPACE"):
    logger.info("Launching Gradio interface locally")
    try:
        interface.launch(share=False, debug=True)
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        raise
    finally:
        save_cache()
else:
    logger.info("Running in Hugging Face Spaces, interface defined but not launched")
    import atexit
    atexit.register(save_cache)
