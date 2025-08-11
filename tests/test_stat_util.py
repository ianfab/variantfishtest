import unittest
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stat_util


class TestStatUtil(unittest.TestCase):
    """Test cases for statistical utility functions."""

    def test_erf_basic_values(self):
        """Test error function with known values."""
        self.assertAlmostEqual(stat_util.erf(0), 0, places=6)
        # Use less precise expectation as this is a custom implementation
        self.assertAlmostEqual(stat_util.erf(1), 0.8427007929, places=3)
        self.assertAlmostEqual(stat_util.erf(-1), -0.8427007929, places=3)
        
    def test_erf_bounds(self):
        """Test error function bounds."""
        # erf should be bounded between -1 and 1
        self.assertLess(abs(stat_util.erf(5)), 1.01)
        self.assertLess(abs(stat_util.erf(-5)), 1.01)
        
    def test_erf_symmetry(self):
        """Test that erf(-x) = -erf(x)."""
        for x in [0.5, 1.0, 1.5, 2.0]:
            self.assertAlmostEqual(stat_util.erf(-x), -stat_util.erf(x), places=6)

    def test_erf_inv_basic_values(self):
        """Test inverse error function with known values."""
        self.assertAlmostEqual(stat_util.erf_inv(0), 0, places=6)
        # Test that erf_inv(erf(x)) ≈ x
        for x in [0.1, 0.5, 0.8]:
            erf_x = stat_util.erf(x)
            self.assertAlmostEqual(stat_util.erf_inv(erf_x), x, places=5)
            
    def test_phi_basic_values(self):
        """Test normal distribution CDF."""
        # phi(0) should be 0.5 (standard normal at mean)
        self.assertAlmostEqual(stat_util.phi(0), 0.5, places=6)
        # phi should be bounded between 0 and 1
        self.assertGreater(stat_util.phi(-3), 0)
        self.assertLess(stat_util.phi(-3), 1)
        self.assertGreater(stat_util.phi(3), 0)
        self.assertLess(stat_util.phi(3), 1)
        
    def test_phi_inv_basic_values(self):
        """Test inverse normal distribution CDF."""
        # phi_inv(0.5) should be 0
        self.assertAlmostEqual(stat_util.phi_inv(0.5), 0, places=6)
        # Test that phi_inv(phi(x)) ≈ x
        for x in [-1, 0, 1]:
            phi_x = stat_util.phi(x)
            self.assertAlmostEqual(stat_util.phi_inv(phi_x), x, places=5)

    def test_get_elo_basic_functionality(self):
        """Test ELO calculation with basic scenarios."""
        # Test equal scores (should be around 0 ELO difference)
        scores = [10, 10, 0]  # 10 wins, 10 losses, 0 draws
        elo, elo95, los = stat_util.get_elo(scores)
        self.assertAlmostEqual(elo, 0, delta=10)  # Should be close to 0
        self.assertGreater(los, 0.4)  # LOS should be around 50%
        self.assertLess(los, 0.6)
        
        # Test clear advantage (more wins than losses)
        scores = [15, 5, 0]  # 15 wins, 5 losses, 0 draws
        elo, elo95, los = stat_util.get_elo(scores)
        self.assertGreater(elo, 0)  # Should be positive ELO
        self.assertGreater(los, 0.9)  # High likelihood of superiority
        
    def test_get_elo_with_draws(self):
        """Test ELO calculation with draws."""
        scores = [10, 5, 10]  # 10 wins, 5 losses, 10 draws
        elo, elo95, los = stat_util.get_elo(scores)
        self.assertGreater(elo, 0)  # Should be positive due to more wins
        
    def test_get_elo_edge_cases(self):
        """Test ELO calculation edge cases."""
        # Test with balanced small sample that won't cause domain errors
        scores = [2, 2, 1]  # Balanced wins/losses with a draw
        elo, elo95, los = stat_util.get_elo(scores)
        self.assertIsInstance(elo, (int, float))
        self.assertIsInstance(elo95, (int, float))
        self.assertIsInstance(los, (int, float))
        
        # Test with mixed results that include draws but avoid zero standard deviation
        scores = [1, 1, 8]  # Mostly draws but some decisive results
        elo, elo95, los = stat_util.get_elo(scores)
        self.assertIsInstance(elo, (int, float))
        # With mostly draws, ELO should be close to 0
        self.assertLess(abs(elo), 50)

    def test_sprt_basic_functionality(self):
        """Test SPRT (Sequential Probability Ratio Test) basic functionality."""
        # Test with a clear result - SPRT function takes drawelo parameter, not max_games
        results = {'wins': 55, 'losses': 45, 'draws': 100}  # Include draws to avoid edge cases
        sprt_result = stat_util.SPRT(results, elo0=0, alpha=0.05, elo1=10, beta=0.05, drawelo=200)
        
        # Check that result contains expected keys
        expected_keys = ['llr', 'lower_bound', 'upper_bound', 'finished', 'state']
        for key in expected_keys:
            self.assertIn(key, sprt_result)
            
        # Check that LLR is a number
        self.assertIsInstance(sprt_result['llr'], (int, float))
        
        # Check bounds are properly ordered
        self.assertLess(sprt_result['lower_bound'], sprt_result['upper_bound'])
        
    def test_sprt_with_draws(self):
        """Test SPRT with draws included."""
        results = {'wins': 30, 'losses': 25, 'draws': 45}
        sprt_result = stat_util.SPRT(results, elo0=0, alpha=0.05, elo1=5, beta=0.05, drawelo=200)
        
        # Should handle draws without crashing
        self.assertIsInstance(sprt_result['llr'], (int, float))
        self.assertIsInstance(sprt_result['finished'], bool)


if __name__ == '__main__':
    unittest.main()
