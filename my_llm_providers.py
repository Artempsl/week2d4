"""LLM provider integration with fallback support (OpenAI + Cohere)."""
import os
import time
import tiktoken
from openai import OpenAI
import cohere
from config import Config

# Pricing (per million tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "command-xlarge-nightly": {"input": 1.50, "output": 6.00},  # Cohere model example
}

class CostTracker:
    """Track API usage costs."""
    def __init__(self):
        self.total_cost = 0.0
        self.requests = []

    def track_request(self, provider, model, input_tokens, output_tokens):
        """Track a single request and compute cost."""
        pricing = PRICING.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost = input_cost + output_cost

        self.total_cost += cost
        self.requests.append({
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })
        return cost

    def get_summary(self):
        """Return summary of tracked costs."""
        total_input = sum(r["input_tokens"] for r in self.requests)
        total_output = sum(r["output_tokens"] for r in self.requests)
        return {
            "total_requests": len(self.requests),
            "total_cost": self.total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_cost": self.total_cost / max(len(self.requests), 1)
        }

    def check_budget(self, daily_budget):
        """Check if total cost exceeds daily budget."""
        if self.total_cost >= daily_budget:
            raise Exception(f"Daily budget of ${daily_budget:.2f} exceeded! Current: ${self.total_cost:.2f}")
        percent_used = (self.total_cost / daily_budget) * 100
        if percent_used >= 90:
            print(f"⚠️  Warning: {percent_used:.1f}% of daily budget used")

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens in text, fallback to estimation."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Rough estimate: 4 characters per token
        return len(text) // 4

class LLMProviders:
    """Manage multiple LLM providers with fallback support."""
    def __init__(self):
        # Clients
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.cohere_client = cohere.Client(Config.COHERE_API_KEY)
        self.cost_tracker = CostTracker()

        # Rate limiting
        self.openai_last_call = 0
        self.cohere_last_call = 0
        self.openai_interval = 60.0 / Config.OPENAI_RPM
        self.cohere_interval = 60.0 / Config.COHERE_RPM

    def _wait_openai(self):
        """Rate-limit OpenAI calls."""
        elapsed = time.time() - self.openai_last_call
        if elapsed < self.openai_interval:
            time.sleep(self.openai_interval - elapsed)
        self.openai_last_call = time.time()

    def _wait_cohere(self):
        """Rate-limit Cohere calls."""
        elapsed = time.time() - self.cohere_last_call
        if elapsed < self.cohere_interval:
            time.sleep(self.cohere_interval - elapsed)
        self.cohere_last_call = time.time()

    def ask_openai(self, prompt, model=None):
        """Ask OpenAI API."""
        if model is None:
            model = Config.OPENAI_MODEL

        self._wait_openai()
        input_tokens = count_tokens(prompt, model)

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        output_text = response.choices[0].message.content
        output_tokens = count_tokens(output_text, model)

        # Track cost
        self.cost_tracker.track_request("openai", model, input_tokens, output_tokens)
        self.cost_tracker.check_budget(Config.DAILY_BUDGET)

        return output_text

    def ask_cohere(self, prompt, model=None, max_tokens=300):
        """Ask Cohere Chat API (single 'message' string)."""
        if model is None:
            model = Config.COHERE_MODEL

        self._wait_cohere()
        input_tokens = count_tokens(prompt, model)

        response = self.cohere_client.chat(
            model=model,
            message=prompt,
            max_tokens=max_tokens
        )

        # Extract text from Cohere response
        output_text = response.text
        output_tokens = count_tokens(output_text, model)

        # Track cost
        self.cost_tracker.track_request("cohere", model, input_tokens, output_tokens)
        self.cost_tracker.check_budget(Config.DAILY_BUDGET)

        return output_text

    def ask_with_fallback(self, prompt, primary="openai"):
        """Ask a prompt using primary provider, fallback to secondary on failure."""
        try:
            if primary == "openai":
                print("Trying OpenAI (primary)...")
                return {"provider": "openai", "response": self.ask_openai(prompt)}
            else:
                print("Trying Cohere (primary)...")
                return {"provider": "cohere", "response": self.ask_cohere(prompt)}
        except Exception as e:
            print(f"✗ Primary provider failed: {e}")
            print("Falling back to secondary provider...")
            try:
                if primary == "openai":
                    return {"provider": "cohere", "response": self.ask_cohere(prompt)}
                else:
                    return {"provider": "openai", "response": self.ask_openai(prompt)}
            except Exception as e2:
                print(f"✗ Secondary provider also failed: {e2}")
                raise Exception("All providers failed")


# --------------------------
# Test the module
# --------------------------
if __name__ == "__main__":
    providers = LLMProviders()

    # Test OpenAI
    print("Testing OpenAI:")
    resp_openai = providers.ask_openai("What is Python? Answer in one sentence.")
    print(f"Response: {resp_openai}\n")

    # Test Cohere
    print("Testing Cohere:")
    resp_cohere = providers.ask_cohere("What is Python? Answer in one sentence.")
    print(f"Response: {resp_cohere}\n")

    # Test fallback
    print("Testing fallback:")
    result = providers.ask_with_fallback("What is machine learning? Answer in one sentence.")
    print(f"Provider used: {result['provider']}")
    print(f"Response: {result['response']}\n")

    # Show cost summary
    summary = providers.cost_tracker.get_summary()
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Total requests: {summary['total_requests']}")
