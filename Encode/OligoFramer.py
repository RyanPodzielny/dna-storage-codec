from Encode.Methods.Encoding import Encoding
from Encode.Indexer import Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, ReedSolomon

class OligoFramer:
    """
    A class for framing oligos with specific sequences and lengths.
    """

    def __init__(self, 
                 oligo_length: int, num_oligos: int,
                 start_primer: str, end_primer: str, 
                 encoder: Encoding, indexer: Indexer, 
                 include_checksum: bool = True,
                 inner_error_correction: ReedSolomon | None = None):
    
        self.oligo_length: int = oligo_length
        self.num_oligos: int = num_oligos
        self.start_primer: str = start_primer
        self.end_primer: str = end_primer
        self.encoder: Encoding = encoder
        self.indexer: Indexer = indexer
        self.include_checksum: bool = include_checksum
        self.inner_error_correction: ReedSolomon | None = inner_error_correction

        self.checksum: CheckSum | None = None
        correction_bases: int = 0
        if self.inner_error_correction:
            correction_bases += self.inner_error_correction.get_dna_bases()
        if self.include_checksum:
            self.checksum = CheckSum(type(self.indexer), self.inner_error_correction is not None, max_run_length=1)
            correction_bases += self.checksum.get_checksum_bases()

        self.payload_length: int = self.oligo_length - len(self.start_primer) - self.indexer.width -\
                                   correction_bases - len(self.end_primer)