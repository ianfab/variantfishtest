# -*- coding: utf-8 -*-
#
# Minimal chess module for variantfishtest
# Contains only the essential parts needed for UCI communication
#

__author__ = "Niklas Fiekas"
__email__ = "niklas.fiekas@tu-clausthal.de"
__version__ = "0.8.0"

# Essential constants
PIECE_TYPES = [ NONE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING ] = range(7)
PIECE_SYMBOLS = [ "", "p", "n", "b", "r", "q", "k" ]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Square names for UCI move parsing
SQUARE_NAMES = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8" ]


class Move(object):
    """Represents a chess move with minimal functionality for UCI parsing."""

    def __init__(self, from_square, to_square, promotion=NONE):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion

    def uci(self):
        """Gets an UCI string for the move."""
        if self:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square] + PIECE_SYMBOLS[self.promotion]
        else:
            return "0000"

    def __bool__(self):
        return bool(self.from_square or self.to_square or self.promotion)

    def __nonzero__(self):
        return self.from_square or self.to_square or self.promotion

    def __eq__(self, other):
        try:
            return self.from_square == other.from_square and self.to_square == other.to_square and self.promotion == other.promotion
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "Move.from_uci('{0}')".format(self.uci())

    def __str__(self):
        return self.uci()

    def __hash__(self):
        return self.to_square | self.from_square << 6 | self.promotion << 12

    @classmethod
    def from_uci(cls, uci):
        """Parses an UCI string."""
        if uci == "0000":
            return cls.null()
        elif len(uci) == 4:
            return cls(SQUARE_NAMES.index(uci[0:2]), SQUARE_NAMES.index(uci[2:4]))
        elif len(uci) == 5:
            promotion = PIECE_SYMBOLS.index(uci[4])
            return cls(SQUARE_NAMES.index(uci[0:2]), SQUARE_NAMES.index(uci[2:4]), promotion)
        elif uci == "(none)":
            return None
        else:
            raise ValueError("expected uci string to be of length 4 or 5")

    @classmethod
    def null(cls):
        """Gets a null move."""
        return cls(0, 0, NONE)
