"""Unit tests for news summarizer using OpenAI + Cohere."""

import unittest
from unittest.mock import Mock, patch
from news_api import NewsAPI
from my_llm_providers import LLMProviders, CostTracker, count_tokens
from summarizer import NewsSummarizer

# -------------------- COST TRACKER TESTS -------------------- #

class TestCostTracker(unittest.TestCase):
    """Test cost tracking functionality."""

    def test_track_request(self):
        tracker = CostTracker()
        cost = tracker.track_request("openai", "gpt-4o-mini", 100, 500)
        self.assertGreater(cost, 0)
        self.assertEqual(tracker.total_cost, cost)
        self.assertEqual(len(tracker.requests), 1)

    def test_get_summary(self):
        tracker = CostTracker()
        tracker.track_request("openai", "gpt-4o-mini", 100, 200)
        tracker.track_request("cohere", "command-xlarge-2026-01", 150, 300)
        summary = tracker.get_summary()
        self.assertEqual(summary["total_requests"], 2)
        self.assertGreater(summary["total_cost"], 0)
        self.assertEqual(summary["total_input_tokens"], 250)
        self.assertEqual(summary["total_output_tokens"], 500)

    def test_budget_check(self):
        tracker = CostTracker()
        tracker.track_request("openai", "gpt-4o-mini", 100, 100)
        tracker.check_budget(10.00)  # Should pass
        tracker.total_cost = 15.00
        with self.assertRaises(Exception) as context:
            tracker.check_budget(10.00)
        self.assertIn("budget", str(context.exception))

# -------------------- TOKEN COUNTING TESTS -------------------- #

class TestTokenCounting(unittest.TestCase):
    """Test token counting."""

    def test_count_tokens(self):
        text = "Hello, how are you?"
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertLess(count, len(text))

# -------------------- NEWS API TESTS -------------------- #

class TestNewsAPI(unittest.TestCase):
    """Test News API integration."""

    @patch('news_api.requests.get')
    def test_fetch_top_headlines(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Test Article",
                    "description": "Test description",
                    "content": "Test content",
                    "url": "https://example.com",
                    "source": {"name": "Test Source"},
                    "publishedAt": "2026-01-19"
                }
            ]
        }
        mock_get.return_value = mock_response
        api = NewsAPI()
        articles = api.fetch_top_headlines(max_articles=1)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Test Article")
        self.assertEqual(articles[0]["source"], "Test Source")

# -------------------- LLM PROVIDERS TESTS -------------------- #

class TestLLMProviders(unittest.TestCase):
    """Test LLM provider integration."""

    @patch('my_llm_providers.OpenAI')
    def test_ask_openai(self, mock_openai_class):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        providers = LLMProviders()
        providers.openai_client = mock_client
        providers.openai_interval = 0  # отключаем sleep

        response = providers.ask_openai("Test prompt")
        self.assertEqual(response, "Test response")
        self.assertTrue(mock_client.chat.completions.create.called)

    @patch('my_llm_providers.count_tokens')
    @patch('my_llm_providers.cohere.Client')
    def test_ask_cohere(self, mock_cohere_class, mock_count_tokens):
        # Чтобы не падал tiktoken
        mock_count_tokens.return_value = 10

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test Cohere response"
        mock_client.chat.return_value = mock_response
        mock_cohere_class.return_value = mock_client

        providers = LLMProviders()
        providers.cohere_interval = 0  # отключаем sleep
        providers.cohere_client = mock_client

        response = providers.ask_cohere("Test prompt")
        self.assertEqual(response, "Test Cohere response")
        self.assertTrue(mock_client.chat.called)

# -------------------- NEWS SUMMARIZER TESTS -------------------- #

class TestNewsSummarizer(unittest.TestCase):
    """Test news summarizer."""

    @patch.object(LLMProviders, 'ask_openai')
    @patch.object(LLMProviders, 'ask_cohere')
    def test_summarize_article(self, mock_cohere, mock_openai):
        mock_openai.return_value = "Test summary"
        mock_cohere.return_value = "Positive sentiment"

        summarizer = NewsSummarizer()
        article = {
            "title": "Test Article",
            "description": "Test description",
            "content": "Test content",
            "url": "https://example.com",
            "source": "Test Source",
            "published_at": "2026-01-19"
        }

        result = summarizer.summarize_article(article)
        self.assertEqual(result["title"], "Test Article")
        self.assertEqual(result["summary"], "Test summary")
        self.assertEqual(result["sentiment"], "Positive sentiment")
        self.assertTrue(mock_openai.called)
        self.assertTrue(mock_cohere.called)

    def test_initialization(self):
        summarizer = NewsSummarizer()
        self.assertIsNotNone(summarizer.news_api)
        self.assertIsNotNone(summarizer.llm_providers)

# -------------------- RUN TESTS -------------------- #

if __name__ == "__main__":
    unittest.main(verbosity=2)
