import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Note: We are mocking the LLM calls here, otherwise this test would make real network 
# requests to OpenAI every time the test suite is run, costing money and risking failures.
# The `run_mab_eval.py` script is what actually runs the real agents.

class TestRealAgentsScript(unittest.TestCase):
    @patch('run_mab_eval.os.getenv')
    def test_missing_api_key(self, mock_getenv):
        # Setup: Missing API key
        mock_getenv.return_value = None
        
        # Action: Run the script's main logic
        import run_mab_eval
        
        # It should exit gracefully instead of throwing errors
        try:
            run_mab_eval.main()
        except Exception as e:
            self.fail(f"Script threw an exception when API key was missing: {e}")

if __name__ == '__main__':
    unittest.main()
