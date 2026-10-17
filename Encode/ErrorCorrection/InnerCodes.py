from fastcrc import crc16, crc32
import reedsolo
from typing import Callable
from Encode.Indexer import Indexer


class CheckSum:
    """
    A class for calculating and verifying checksums. Uses Cyclic Redundancy Check (CRC) for error detection. In our
    specific case we will use CRC-32 on each oligio, if we have an inner ECC code, e.g. RS, we will use CRC-16 to save
    space as the inner code will already provide some error correction.
    """

    def __init__(self, indexer: type[Indexer], inner_code: bool = False, max_run_length: int | None = None):
        """
        Initializes the CheckSum instance.

        Args:
            inner_code (bool): A flag indicating whether to use inner code. 
        """

        self.inner_code: bool = inner_code

        # Default to CRC-32 for checksum calculation if we have an inner_code, use CRC-16 instead
        #   Everything comes from: https://fastcrc.readthedocs.io/en/latest/
        self.width: int = 0
        self.crc_func: Callable
        if self.inner_code:
            self.crc_func = crc16.xmodem
            self.width = 16
        else:
            self.crc_func = crc32.iso_hdlc
            self.width = 32


        # Bit shift left to get 2^required_width, which is number of unique values we can have
        self.indexer: Indexer = indexer(
            width=indexer.determine_optimal_width(1 << self.width),
            max_run_length=max_run_length
        )


    def get_checksum_bases(self) -> int:
        """
        Gets the length of bits used for the checksum based on the configuration,

        Returns:
            int: The length of the checksum in bases
        """

        return self.indexer.width
    

    def get_checksum_length(self) -> int:
        """
        Gets the length of bits used for the checksum based on the configuration,

        Returns:
            int: The length of the checksum in bases
        """

        return self.width


    def calculate_checksum(self, data: str) -> int:
        """
        Calculates the checksum for the given data.

        Args:
            data: The index + after data for which the checksum is to be calculated.

        Returns:
            int: The calculated checksum.
        """
        # Convert data into bytes
        byte_stream: bytes = data.encode("utf-8")

        return self.crc_func(byte_stream)


    def create_checksum_segments(self, unencoded_data: list[str]) -> list[int]:
        """
        Calculates the checksum for each string in the provided list.

        Args:
            unencoded_data: A list of strings for which checksums will be calculated - not DNA bases.

        Returns:
            list[int]: A new list of integers with checksums appended.
        """
        checksums: list[int] = []

        for item in unencoded_data:
            checksum: int = self.calculate_checksum(item)
            checksums.append(checksum)

        return checksums


    def verify_checksum(self, data: str, checksum: int) -> bool:
        """
        Verifies the checksum of the provided data.

        Args:
            data: The data string with the appended checksum.
            checksum: The checksum to verify against.

        Returns:
            bool: True if the checksum is valid, False otherwise.
        """

        # Calculate the checksum for the original data
        calculated_checksum: int = self.calculate_checksum(data)

        # Compare the calculated checksum with the provided checksum
        return calculated_checksum == checksum



    def encode_to_dna(self, checksum: int, before_sequence: str, after_sequence: str) -> str:
        """
        Converts the checksum integer to a DNA sequence.

        Args:
            checksum: The checksum integer to be converted.
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            indexer: The indexer to use for the conversion.

        Returns:
            str: The corresponding DNA sequence.
        """
        # Convert the checksum integer to a DNA sequence using the indexer
        dna_sequence: str = self.indexer.unrank(checksum, 
                                                before=before_sequence, 
                                                after=after_sequence)

        return dna_sequence



    def decode_from_dna(self, dna_sequence: str, before_sequence: str, after_sequence: str) -> int:
        """
        Converts a DNA sequence back to the corresponding checksum integer.

        Args:
            dna_sequence: The DNA sequence to be converted.
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            indexer: The indexer to use for the conversion.

        Returns:
            int: The corresponding checksum integer.
        """
        # Convert the DNA sequence back to the checksum integer using the indexer
        checksum: int = self.indexer.rank(dna_sequence, 
                                          before=before_sequence, 
                                          after=after_sequence
                                          )

        return checksum


class ReedSolomon:
    """
    Reed-Solomon error correction for DNA strands.
    """
    def __init__(self, parity_bytes: int):
        """
        Initializes the ReedSolomon instance.

        Args:
            parity_bytes: The number of Reed-Solomon parity bytes to append.
        """
        if not 1 <= parity_bytes <= 254:
            raise ValueError("parity_bytes must be between 1 and 254.")

        self.parity_bytes: int = parity_bytes
        self.RSCodec: reedsolo.RSCodec = reedsolo.RSCodec(parity_bytes)