from Methods.Encoding import Encoding
from Indexer import Indexer

class OligoFramer:
    """
    A class for framing oligos with specific sequences and lengths.
    """

    def __init__(self, oligo_length: int, start_primer: str, end_primer: str, encoder: Encoding, indexer: Indexer):
        self.oligo_length: int = oligo_length
        self.start_primer: str = start_primer
        self.end_primer: str = end_primer
        self.encoder: Encoding = encoder
        self.indexer: Indexer = indexer
