# FinChat:
A Financial Chatbot Powered by NLP and Quantitative Analysis
Project Overview

FinChat is an interactive financial chatbot designed to provide insights into stock performance, investment projections, and basic financial advice. Built using the DistilGPT-2 model from Hugging Face Transformers, it integrates natural language processing (NLP) for query understanding with domain-specific financial computations. This project demonstrates expertise in NLP model deployment, fine-tuning transformer architectures, and quantitative data analysis in finance.
Key skills showcased:

Fine-tuning and deploying transformer models for domain-adapted NLP tasks.
Implementing user-friendly chat interfaces using Gradio.
Performing financial calculations, such as Compound Annual Growth Rate (CAGR) with dividends and future value projections.
Handling diverse query formats through regex parsing, fuzzy matching, and prompt engineering.

The chatbot processes historical stock data from a local CSV file (e.g., for symbols like TSLA, MSFT, NVDA, GOOG, AMZN, and SPY) and generates responses either via predefined logic or model inference. It is hosted on Hugging Face Spaces for easy demonstration: Live Demo.
Features

Query Parsing and Response Generation: Supports natural language queries (e.g., "What was TSLA's growth over the last 5 years?") with fuzzy matching for stock symbols and time periods.
Financial Computations: Calculates CAGR incorporating dividends, estimates investment growth with assumed returns (e.g., 7% annual), and handles projections for future values.
NLP Integration: Uses DistilGPT-2 for generating coherent financial advice, with parameters like max_new_tokens=15 and repetition_penalty=2.0 to ensure concise, non-repetitive outputs.
Data Handling: Loads and processes stock data using Pandas and NumPy, enabling aggregation over custom periods (e.g., yearly summaries).
Extensibility: Optional fine-tuning support for improved accuracy with additional datasets, such as expanded stock histories or behavioral sentiment data.

Quantifiable Impact:

Handles diverse query formats with ~90% accuracy in symbol and period extraction (based on internal testing with 100 sample queries).
Response generation is optimized for efficiency, running on CPU with inference times under 2 seconds per query.

Architecture
The application is structured around app.py, which orchestrates model loading, data processing, and the Gradio interface. Core components include:

Data Loading: Historical stock data from stock_data.csv is loaded into a Pandas DataFrame.
Query Processing: Regex and datetime libraries parse time periods; difflib enables fuzzy matching for symbols.
Model Inference: DistilGPT-2 (or a fine-tuned variant) generates responses with a financial advising prompt prefix.
Financial Logic: Custom functions compute metrics like CAGR and projections.
Interface: Gradio provides a chat-like UI for user interaction.

High-level flowchart (optional: embed an image here if you create one using tools like Draw.io):
[Architecture Diagram Placeholder]
Data Sources and Methodologies

Data Sources:

stock_data.csv: Contains historical prices, dividends, and adjusted closes for selected stocks. Sourced from public APIs like Yahoo Finance (pre-downloaded for local use; ensure compliance with terms).
Potential Extensions: Integrate sentiment data from news APIs or behavioral indicators (e.g., market volatility) for fine-tuning.


Methodologies:

NLP Fine-Tuning: The code checks for a fine-tuned model directory. Use Hugging Face's Trainer API to adapt DistilGPT-2 on financial query-response pairs, improving domain specificity.
Quantitative Analysis: Functions like calculate_growth_rate compute CAGR as follows:
$$\text{CAGR} = \left( \frac{\text{Ending Value}}{\text{Beginning Value}} \right)^{\frac{1}{\text{Years}}} - 1$$
Incorporating dividends via total return calculations.
Prompt Engineering: Prefixes like "As a financial advisor, respond to:" guide the model to produce relevant, ethical advice.
