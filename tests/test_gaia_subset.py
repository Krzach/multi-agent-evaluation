import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from benchmarks.gaia.dataset import GaiaDataset

class TestGaiaDataset(unittest.TestCase):
    def setUp(self):
        # Point to the subset JSON we just created
        self.data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'benchmarks', 
            'data', 
            'gaia_subset.json'
        )
        self.dataset = GaiaDataset(self.data_path)

    def test_load_single_hop(self):
        single_hop_tasks = self.dataset.get_single_hop_tasks()
        self.assertGreater(len(single_hop_tasks), 0)
        
        # Verify they are all Level 1
        for task in single_hop_tasks:
            self.assertEqual(task["Level"], 1)
            
    def test_load_multi_hop(self):
        multi_hop_tasks = self.dataset.get_multi_hop_tasks()
        self.assertGreater(len(multi_hop_tasks), 0)
        
        # Verify they are all Level 2
        for task in multi_hop_tasks:
            self.assertEqual(task["Level"], 2)

    def test_limit(self):
        limited_tasks = self.dataset.get_single_hop_tasks(limit=1)
        self.assertEqual(len(limited_tasks), 1)

if __name__ == '__main__':
    unittest.main()
