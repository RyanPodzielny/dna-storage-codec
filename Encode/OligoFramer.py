from Encode.Methods.Encoding import Encoding
from Encode.Indexer import Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum

class OligoFramer:
    """
    A class for framing oligos with specific sequences and lengths.
    """

    def __init__(self, 
                 oligo_length: int, num_oligos: int,
                 start_primer: str, end_primer: str, 
                 encoder: Encoding, indexer: Indexer, 
                 include_checksum: bool = True, include_error_correction: bool = True):
    
        self.oligo_length: int = oligo_length
        self.num_oligos: int = num_oligos
        self.start_primer: str = start_primer
        self.end_primer: str = end_primer
        self.encoder: Encoding = encoder
        self.indexer: Indexer = indexer
        self.include_checksum: bool = include_checksum
        self.include_error_correction: bool = include_error_correction

        self.checksum: CheckSum | None = None
        checksum_bases: int = 0
        if self.include_checksum:
            self.checksum = CheckSum(type(self.indexer), self.include_error_correction)
            checksum_bases = self.checksum.get_checksum_bases()

        self.payload_length: int = self.oligo_length - len(self.start_primer) - len(self.end_primer) - checksum_bases - self.indexer.width




    def frame_oligo(self, index: int, payload: str) -> str:
        """
        Frames an oligo with the specified index and payload.

        Args:
            index: The index of the oligo.
            payload: The payload sequence to be included in the oligo.

        Returns:
            str: The framed oligo sequence.
        """
        # Generate the index sequence using the indexer
        index_sequence: str = self.indexer.unrank(index, primer=self.start_primer, payload=payload)

        # If checksum is included, calculate and append it
        if self.include_checksum and self.checksum is not None:
            checksum_value: int = self.checksum.calculate_checksum(payload)
            checksum_sequence: str = self.checksum.encode_to_dna(checksum_value, primer=self.start_primer, payload=payload)
            framed_oligo: str = f"{self.start_primer}{index_sequence}{payload}{checksum_sequence}{self.end_primer}"
        else:
            framed_oligo: str = f"{self.start_primer}{index_sequence}{payload}{self.end_primer}"

        return framed_oligo