#!/usr/bin/env python
"""Test script for tournament functionality."""

import unittest.mock
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import variantfishtest


def test_tournament_mode_detection():
    """Test that tournament mode is correctly detected with 3+ engines."""
    # Test with 3 engines (should enable tournament mode)
    test_args = ['variantfishtest.py', 'engine1', 'engine2', 'engine3']
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        assert match.is_tournament == True
        assert match.num_engines == 3
        assert len(match.engine_pairs) == 3  # (0,1), (0,2), (1,2)
        assert match.engine_pairs == [(0, 1), (0, 2), (1, 2)]

def test_two_engine_mode():
    """Test that two engine mode is still the default."""
    # Test with 2 engines (should NOT enable tournament mode)
    test_args = ['variantfishtest.py', 'engine1', 'engine2']
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        assert match.is_tournament == False
        assert match.num_engines == 2

def test_engine_pairs_generation():
    """Test that engine pairs are correctly generated for tournaments."""
    # Test with 4 engines
    test_args = ['variantfishtest.py', 'engine1', 'engine2', 'engine3', 'engine4']
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert match.engine_pairs == expected_pairs
        assert len(match.engine_pairs) == 6  # C(4,2) = 6 pairs

def test_new_engine_options_format():
    """Test the new engine options format for tournaments."""
    test_args = [
        'variantfishtest.py', 'engine1', 'engine2', 'engine3',
        '--engine-options', '1:Hash=128',
        '--engine-options', '2:Threads=2', 
        '--engine-options', '3:Hash=256'
    ]
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        expected_options = [
            {'Hash': '128'},
            {'Threads': '2'},
            {'Hash': '256'}
        ]
        assert match.engine_options == expected_options

def test_backward_compatibility_options():
    """Test that old-style options still work for first 2 engines."""
    test_args = [
        'variantfishtest.py', 'engine1', 'engine2', 'engine3',
        '--e1-options', 'Hash=64',
        '--e2-options', 'Threads=4'
    ]
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        # First two engines should have the options set
        assert match.engine_options[0] == {'Hash': '64'}
        assert match.engine_options[1] == {'Threads': '4'}
        assert match.engine_options[2] == {}  # Third engine has no options

def test_tournament_score_initialization():
    """Test that tournament score tracking is properly initialized."""
    test_args = ['variantfishtest.py', 'engine1', 'engine2', 'engine3']
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        # Should have pair scores for each engine pair
        assert len(match.pair_scores) == 3
        assert len(match.pair_time_losses) == 3
        
        # Each pair should have initialized scores [wins_first, wins_second, draws]
        for pair in match.engine_pairs:
            assert match.pair_scores[pair] == [0, 0, 0]
            assert match.pair_time_losses[pair] == [0, 0]

def test_play_match_instance_returns_engine_pair():
    """Test that play_match_instance returns the engine pair for tournaments."""
    test_args = ['variantfishtest.py', 'engine1', 'engine2', 'engine3']
    with unittest.mock.patch('sys.argv', test_args):
        match = variantfishtest.EngineMatch()
        
        # Mock the play_game_instance method to avoid actually running engines
        with unittest.mock.patch.object(match, 'play_game_instance') as mock_play:
            mock_play.return_value = (variantfishtest.WIN, 0, [])  # WIN, no time loss, no moves
            
            # Mock random.choice to return predictable results
            with unittest.mock.patch('random.choice') as mock_choice:
                mock_choice.side_effect = ['chess', (0, 1), 'startpos']  # variant, engine_pair, pos
                
                result = match.play_match_instance()
                assert len(result) == 5  # res1, res2, tl1, tl2, engine_pair
                res1, res2, tl1, tl2, engine_pair = result
                assert engine_pair == (0, 1)

if __name__ == '__main__':
    test_tournament_mode_detection()
    test_two_engine_mode()
    test_engine_pairs_generation()
    test_new_engine_options_format()
    test_backward_compatibility_options()
    test_tournament_score_initialization()
    test_play_match_instance_returns_engine_pair()
    print("All tournament tests passed!")