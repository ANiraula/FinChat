import logging
import os
import time
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define device (force CPU for Spaces free tier)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Expanded response cache with new entries
response_cache = {
    "hi": "Hello! I'm FinChat, your financial advisor. How can I help with investing today?",
    "hello": "Hello! I'm FinChat, your financial advisor. How can I help with investing today?",
    "hey": "Hi there! Ready to discuss investment goals with FinChat?",
    "how can i start investing with $100 a month?": (
        "Here’s a step-by-step guide to start investing with $100 a month:\n"
        "1. **Open a brokerage account** with a platform like Fidelity or Robinhood. They offer low fees and no minimums.\n"
        "2. **Deposit your $100 monthly**. You can set up automatic transfers from your bank.\n"
        "3. **Choose a low-cost ETF** like VOO, which tracks the S&P 500 for broad market exposure.\n"
        "4. **Set up automatic investments** to buy shares regularly, reducing the impact of market fluctuations.\n"
        "5. **Track your progress** every few months to stay on top of your investments.\n"
        "Consult a financial planner for personalized advice."
    ),
    "where can i open a brokerage account?": (
        "You can open a brokerage account with platforms like Vanguard, Fidelity, Charles Schwab, or Robinhood. "
        "They are beginner-friendly and offer low fees. Choose one that fits your needs and sign up online."
    ),
    "start investing with 100 dollars a month": (
        "Here’s how to start investing with $100 a month:\n"
        "1. **Open a brokerage account** with a platform like Fidelity or Robinhood.\n"
        "2. **Deposit $100 monthly** via automatic transfers.\n"
        "3. **Invest in a low-cost ETF** like VOO for diversification.\n"
        "4. **Use dollar-cost averaging** to invest regularly.\n"
        "5. **Monitor your investments** quarterly.\n"
        "Consult a financial planner for tailored advice."
    ),
    "best places to open a brokerage account": (
        "The best places to open a brokerage account include Vanguard, Fidelity, Charles Schwab, and Robinhood. "
        "They offer low fees, no minimums, and user-friendly platforms for beginners."
    ),
    "what is dollar-cost averaging?": (
        "Dollar-cost averaging is investing a fixed amount regularly (e.g., $100 monthly) in ETFs, "
        "reducing risk by spreading purchases over time."
    ),
    "how much should i invest?": (
        "Invest what you can afford after expenses and an emergency fund. Start with $100-$500 monthly "
        "in ETFs like VOO using dollar-cost averaging. Consult a financial planner."
    ),
    "how to start investing": (
        "Here’s how to start investing:\n"
        "1. Educate yourself using resources like Investopedia.\n"
        "2. Open a brokerage account with a platform like Fidelity.\n"
        "3. Deposit an initial amount, such as $100, after building an emergency fund.\n"
        "4. Choose a low-cost ETF like VOO.\n"
        "5. Invest regularly using dollar-cost averaging.\n"
        "Consult a financial planner for personalized advice."
    ),
    "best brokerage accounts": (
        "The best brokerage accounts for beginners include Fidelity, Vanguard, Charles Schwab, and Robinhood. "
        "They offer low fees, no minimums, and user-friendly platforms."
    ),
    "investing for beginners": (
        "Here’s a beginner’s guide to investing:\n"
        "1. Learn the basics from Investopedia or books like 'The Intelligent Investor.'\n"
        "2. Set clear investment goals and assess your risk tolerance.\n"
        "3. Open a brokerage account with a platform like Fidelity or Robinhood.\n"
        "4. Start with low-cost ETFs like VOO or index funds.\n"
        "5. Invest regularly using dollar-cost averaging.\n"
        "6. Monitor your investments quarterly.\n"
        "Consult a financial planner for tailored advice."
    ),
    "steps to start investing": (
        "Here are the steps to start investing:\n"
        "1. Educate yourself on investing basics.\n"
        "2. Open a brokerage account with a beginner-friendly platform.\n"
        "3. Deposit an initial amount you can afford.\n"
        "4. Choose a diversified investment like an ETF.\n"
        "5. Invest consistently over time.\n"
        "Consult a financial planner for more guidance."
    ),
    "recommended etfs": (
        "Recommended ETFs for beginners include VOO (tracks S&P 500), QQQ (tech-focused), and VT (global market exposure). "
        "They offer diversification and low fees."
    ),
    "how much to invest": (
        "The amount to invest depends on your financial situation. Start with what you can afford after covering expenses and an emergency fund. "
        "A common starting point is $100-$500 monthly in low-cost ETFs like VOO. Consult a financial planner for personalized advice."
    ),
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

# Load model and tokenizer
model_name = "distilgpt2"
try:
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    logger.info(f"Loading model {model_name}")
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
except Exception as e:
    logger.error(f"Error loading model/tokenizer: {e}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Updated prompt prefix with better instructions and examples
prompt_prefix = (
    "You are FinChat, a financial advisor. Always provide clear, step-by-step answers to the user's exact question. "
    "Avoid vague or unrelated topics. Use a numbered list format where appropriate and explain each step.\n\n"
    "Example 1:\n"
    "Q: How can I start investing with $100 a month?\n"
    "A: Here’s a step-by-step guide:\n"
    "1. Open a brokerage account with a platform like Fidelity or Robinhood. They offer low fees and no minimums.\n"
    "2. Deposit your $100 monthly. You can set up automatic transfers.\n"
    "3. Choose a low-cost ETF like VOO, which tracks the S&P 500.\n"
    "4. Set up automatic investments to buy shares regularly.\n"
    "5. Track your progress every few months.\n\n"
    "Example 2:\n"
    "Q: Where can I open a brokerage account?\n"
    "A: You can open an account with platforms like Vanguard, Fidelity, Charles Schwab, or Robinhood. "
    "They are beginner-friendly and have low fees.\n\n"
    "Q: "
)

# Define chat function with substring matching and reduced max_new_tokens
def chat_with_model(user_input, history=None):
    try:
        start_time = time.time()
        logger.info(f"Processing user input: {user_input}")
        
        user_input_lower = user_input.lower().strip()
        
        # Substring matching for cache
        matching_keys = [key for key in response_cache if key in user_input_lower]
        if matching_keys:
            longest_key = max(matching_keys, key=len)
            logger.info(f"Cache hit for: {longest_key}")
            response = response_cache[longest_key]
            logger.info(f"Chatbot response: {response}")
            history = history or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            end_time = time.time()
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            return response, history
        
        if len(user_input.strip()) <= 5:
            logger.info("Short prompt, returning default response")
            response = "Hello! I'm FinChat, your financial advisor. Ask about investing!"
            logger.info(f"Chatbot response: {response}")
            history = history or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            end_time = time.time()
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            return response, history

        full_prompt = prompt_prefix + user_input + "\nA:"
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.inference_mode():
            gen_start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced for faster generation
                min_length=20,
                do_sample=False,  # Greedy decoding for speed
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_end_time = time.time()
            logger.info(f"Generation time: {gen_end_time - gen_start_time:.2f} seconds")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(full_prompt):].strip() if response.startswith(full_prompt) else response
        logger.info(f"Chatbot response: {response}")
        
        # Update cache with exact user input as key
        response_cache[user_input_lower] = response
        logger.info("Cache miss, added to in-memory cache")
        
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info(f"Total response time: {end_time - start_time:.2f} seconds")
        return response, history
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = f"Error: {str(e)}"
        logger.info(f"Chatbot response: {response}")
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return response, history

# Create Gradio interface
with gr.Blocks(
    title="FinChat: An LLM based on distilgpt2 model",
    css=".feedback {display: flex; gap: 10px; justify-content: center; margin-top: 10px;}"
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
    
    def submit_message(user_input, history):
        response, updated_history = chat_with_model(user_input, history)
        return "", updated_history  # Clear input, update chatbot

    submit.click(
        fn=submit_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    clear.click(
        fn=lambda: ("", []),  # Clear input and chatbot
        outputs=[msg, chatbot]
    )

# Launch interface (conditional for Spaces)
if __name__ == "__main__" and not os.getenv("HF_SPACE"):
    logger.info("Launching Gradio interface locally")
    try:
        interface.launch(share=False, debug=True)
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        raise
else:
    logger.info("Running in Hugging Face Spaces, interface defined but not launched")
