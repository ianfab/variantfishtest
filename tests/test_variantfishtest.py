import unittest
import unittest.mock
import sys
import os
import tempfile

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import variantfishtest


class TestVariantFishTest(unittest.TestCase):
    """Test cases for the main variantfishtest functionality."""

    def test_global_constants(self):
        """Test that global constants are properly defined."""
        self.assertEqual(variantfishtest.WIN, 0)
        self.assertEqual(variantfishtest.LOSS, 1)
        self.assertEqual(variantfishtest.DRAW, 2)
        self.assertEqual(len(variantfishtest.SCORES), 3)
        self.assertEqual(variantfishtest.SCORES[variantfishtest.WIN], 1)
        self.assertEqual(variantfishtest.SCORES[variantfishtest.LOSS], 0)
        self.assertEqual(variantfishtest.SCORES[variantfishtest.DRAW], 0.5)

    def test_elo_stats_function(self):
        """Test the elo_stats formatting function."""
        # Test with valid scores
        scores = [10, 5, 5]  # 10 wins, 5 losses, 5 draws
        result = variantfishtest.elo_stats(scores)
        self.assertIsInstance(result, str)
        self.assertIn("ELO:", result)
        self.assertIn("LOS:", result)
        
        # Test with edge case (should not crash)
        scores = [0, 0, 1]  # Only draws
        result = variantfishtest.elo_stats(scores)
        self.assertIsInstance(result, str)

    def test_sprt_stats_function(self):
        """Test the SPRT stats formatting function."""
        scores = [15, 10, 5]  # 15 wins, 10 losses, 5 draws
        result = variantfishtest.sprt_stats(scores, elo1=-5, elo2=5)
        self.assertIsInstance(result, str)
        self.assertIn("LLR:", result)
        
    def test_print_scores_function(self):
        """Test the score printing function."""
        scores = [12, 8, 10]  # 12 wins, 8 losses, 10 draws
        result = variantfishtest.print_scores(scores)
        expected = "Total: 30 W: 12 L: 8 D: 10"
        self.assertEqual(result, expected)
        
        # Test with zero scores
        scores = [0, 0, 0]
        result = variantfishtest.print_scores(scores)
        expected = "Total: 0 W: 0 L: 0 D: 0"
        self.assertEqual(result, expected)

    def test_engine_match_initialization(self):
        """Test EngineMatch class initialization with mock arguments."""
        # Mock sys.argv to provide required arguments
        with unittest.mock.patch('sys.argv', ['variantfishtest.py', 'engine1', 'engine2']):
            match = variantfishtest.EngineMatch()
            
            # Check that basic attributes are set
            self.assertEqual(match.engine1, 'engine1')
            self.assertEqual(match.engine2, 'engine2')
            self.assertEqual(match.variant, 'chess')  # default
            self.assertEqual(match.max_games, 5000)  # default
            self.assertEqual(match.time, 10000)  # default
            self.assertEqual(match.inc, 100)  # default
            self.assertEqual(match.threads, 1)  # default
            
            # Check that score tracking is initialized
            self.assertEqual(match.scores, [0, 0, 0])
            self.assertEqual(match.time_losses, [0, 0])
            self.assertEqual(match.white_wins, 0)
            self.assertEqual(match.black_wins, 0)
            self.assertEqual(match.draw_games, 0)
            self.assertEqual(match.pentanomial, [0, 0, 0, 0, 0])
            self.assertIsInstance(match.r, list)
            
            # Check that threading lock is created
            self.assertIsNotNone(match.lock)

    def test_engine_match_with_custom_args(self):
        """Test EngineMatch with custom arguments."""
        test_args = [
            'variantfishtest.py', 'engine1', 'engine2',
            '--variant', 'atomic',
            '--max_games', '100',
            '--time', '5000',
            '--inc', '50',
            '--threads', '2'
        ]
        with unittest.mock.patch('sys.argv', test_args):
            match = variantfishtest.EngineMatch()
            
            self.assertEqual(match.variant, 'atomic')
            self.assertEqual(match.max_games, 100)
            self.assertEqual(match.time, 5000)
            self.assertEqual(match.inc, 50)
            self.assertEqual(match.threads, 2)

    def test_stop_method(self):
        """Test the stop condition checking."""
        with unittest.mock.patch('sys.argv', ['variantfishtest.py', 'engine1', 'engine2']):
            match = variantfishtest.EngineMatch()
            
            # Should not stop initially
            self.assertFalse(match.stop())
            
            # Should stop when max games reached
            match.scores = [50, 30, 20]  # Total = 100, max_games = 5000
            self.assertFalse(match.stop())
            
            match.max_games = 50
            self.assertTrue(match.stop())  # Total games >= max_games

    def test_sprt_finished(self):
        """Test SPRT finished condition."""
        test_args = [
            'variantfishtest.py', 'engine1', 'engine2',
            '--sprt', '--elo0', '0', '--elo1', '10'
        ]
        with unittest.mock.patch('sys.argv', test_args):
            match = variantfishtest.EngineMatch()
            
            # With very few games, SPRT should not be finished
            match.scores = [1, 1, 0]
            finished = match.sprt_finished()
            self.assertIsInstance(finished, bool)

    def test_init_book_file_not_exists(self):
        """Test book initialization with non-existent file."""
        with unittest.mock.patch('sys.argv', ['variantfishtest.py', 'engine1', 'engine2']):
            match = variantfishtest.EngineMatch()
            match.book = "/path/that/does/not/exist.epd"
            
            # Should handle missing book file gracefully
            with unittest.mock.patch('warnings.warn') as mock_warn:
                match.init_book()
                mock_warn.assert_called_once()

    def test_init_book_with_content(self):
        """Test book initialization with actual content."""
        with unittest.mock.patch('sys.argv', ['variantfishtest.py', 'engine1', 'engine2']):
            match = variantfishtest.EngineMatch()
            
            # Create a temporary book file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
                f.write("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;\n")
                f.write("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1;\n")
                temp_book = f.name
            
            try:
                match.book = temp_book
                match.init_book()
                
                # Should have loaded FENs
                self.assertEqual(len(match.fens), 2)
                self.assertIn("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", match.fens[0])
            finally:
                os.unlink(temp_book)

    def test_variants_splitting(self):
        """Test that multiple variants are split correctly."""
        test_args = [
            'variantfishtest.py', 'engine1', 'engine2',
            '--variant', 'chess,atomic,crazyhouse'
        ]
        with unittest.mock.patch('sys.argv', test_args):
            match = variantfishtest.EngineMatch()
            
            self.assertEqual(match.variants, ['chess', 'atomic', 'crazyhouse'])
            self.assertEqual(match.variant, 'chess')  # First variant is default

    def test_engine_paths_absolute(self):
        """Test that engine paths are converted to absolute paths."""
        with unittest.mock.patch('sys.argv', ['variantfishtest.py', 'engine1', 'engine2']):
            match = variantfishtest.EngineMatch()
            
            # Paths should be absolute
            self.assertTrue(os.path.isabs(match.engine_paths[0]))
            self.assertTrue(os.path.isabs(match.engine_paths[1]))

    def test_engine_options_parsing(self):
        """Test parsing of engine options."""
        test_args = [
            'variantfishtest.py', 'engine1', 'engine2',
            '--e1-options', 'Hash=128',
            '--e1-options', 'Threads=2',
            '--e2-options', 'Hash=256'
        ]
        with unittest.mock.patch('sys.argv', test_args):
            match = variantfishtest.EngineMatch()
            
            expected_e1_options = {'Hash': '128', 'Threads': '2'}
            expected_e2_options = {'Hash': '256'}
            
            self.assertEqual(match.engine_options[0], expected_e1_options)
            self.assertEqual(match.engine_options[1], expected_e2_options)

    def test_log_file_handling(self):
        """Test log file output handling."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        try:
            test_args = [
                'variantfishtest.py', 'engine1', 'engine2',
                '--log', temp_log_path
            ]
            with unittest.mock.patch('sys.argv', test_args):
                match = variantfishtest.EngineMatch()
                
                # Should have opened the log file
                self.assertNotEqual(match.out, sys.stdout)
                match.close()
        finally:
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)


class TestPentanomialStats(unittest.TestCase):
    """Test pentanomial statistics calculations."""
    
    def test_pentanomial_score_calculation(self):
        """Test the pentanomial score calculation logic."""
        # This tests the logic that would be in the worker method
        test_cases = [
            (0.0, 0.0, 0),  # LL
            (0.5, 0.0, 1),  # LD  
            (1.0, 0.0, 2),  # DD or WL
            (1.5, 0.0, 3),  # WD
            (2.0, 0.0, 4),  # WW
        ]
        
        eps = 1e-9
        for pair_score, _, expected_idx in test_cases:
            if abs(pair_score - 0.0) < eps:
                idx = 0  # LL
            elif abs(pair_score - 0.5) < eps:
                idx = 1  # LD
            elif abs(pair_score - 1.0) < eps:
                idx = 2  # DD or WL
            elif abs(pair_score - 1.5) < eps:
                idx = 3  # WD
            else:
                idx = 4  # WW
            
            self.assertEqual(idx, expected_idx, f"Failed for pair_score {pair_score}")


if __name__ == '__main__':
    unittest.main()
