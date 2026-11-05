"""
Encoding module for DNA sequences.

This module provides functionality as the parent class for all encoding methods. It includes methods for encoding and 
decoding strings and files, as well as calculating the size of the encoded data. The module also provides a method to 
count runs of DNA bases in the encoded data. Most methods are abstract and should be implemented in subclasses that 
define specific encoding strategies.
"""


from abc import abstractmethod
from typing import Final, Any


class Encoding:
    """
    Base class for encoding methods. Provides common functionality for encoding and decoding DNA sequences, as well
    as methods for calculating the size of the encoded data and counting runs of DNA bases. Subclasses should implement
    specific encoding strategies by overriding the abstract methods defined in this class.
    """

    def __init__(self):
        """
        Initializes a new instance of the Encoding class. Sets up necessary constants for children classes.
        """

        # Huffman coding storage
        # Each character is stored as 8 bits (1 byte)
        self.KEY_MULTIPLIER: int = 8  
        # Each bit in the code
        self.VALUE_MULTIPLIER: int = 1  
        # Each estimated base per byte 
        self.ESTIMATED_BASES_PER_BYTE: float = 2.0


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   GETTER METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this encoder when loading a saved framer. Encoders
        without constructor arguments don't need to override this.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # After encoding, use the binary string to map to DNA bases with its reverse
    DNA_BASES: Final[set[str]] = {'A', 'C', 'G', 'T'}
    DNA_MAP: Final[dict[str, str]] = {
        '00': 'A',
        '01': 'C',
        '10': 'G',
        '11': 'T'
    }
    REVERSE_DNA_MAP: Final[dict[str, str]] = {v: k for k, v in DNA_MAP.items()}

    BINARY_SYMBOLS: Final[set[str]] = {'0', '1'}
    TRINARY_SYMBOLS: Final[set[str]] = {'0', '1', '2'}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def binary_to_ascii(binary_string: str) -> str:
        """
        Converts a binary string to an ASCII string.

        Args:
            binary_string: The binary string to be converted.

        Returns:
            An ASCII string representation of the binary string.

        Raises:
            ValueError: If cannot convert a byte to an ASCII character.
        """
        ascii_string: str = ""
        for index in range(0, len(binary_string), 8):
            byte: str = binary_string[index:index+8]
            ascii_string += chr(int(byte, 2))
        return ascii_string


    @staticmethod
    def ascii_to_binary(ascii_string: str) -> str:
        """
        Converts an ASCII string to a binary string.

        Args:
            ascii_string: The ASCII string to be converted.

        Returns:
            A binary string representation of the ASCII string.
        """
        return "".join(format(ord(char), '08b') for char in ascii_string)


    @staticmethod
    def as_sequences(encoded_data: str | list[str]) -> list[str]:
        """
        Gets encoded data as a list of sequences, so a whole file encoding (one string) and framed or segmented data
        (a list of strings) can be measured the same way.

        Args:
            encoded_data: The encoded data as one string or a list of strings.

        Returns:
            The encoded data as a list of strings.
        """
        if isinstance(encoded_data, str):
            return [encoded_data]
        return encoded_data


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @abstractmethod
    def _add_padding(self, data: str, payload_length: int) -> str:
        raise NotImplementedError()


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#
    

    @abstractmethod
    def encode_segments(self, data: str, payload_length: int, *args, **kwargs) -> tuple[dict[str, str], list[str], Any]:
        raise NotImplementedError()
    

    @abstractmethod
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        raise NotImplementedError()


    @abstractmethod
    def decode_segments(self, encoded_data: list[str | None], *args, **kwargs) -> list[str | None]:
        raise NotImplementedError()
    

    @abstractmethod
    def decode(self, data: str, *args, **kwargs) -> str:
        raise NotImplementedError()


    @abstractmethod
    def estimate_bases_per_byte(self) -> float:
        raise NotImplementedError()


    @abstractmethod
    def calculate_overall_size(self, encoded_data: str | list[str], huffman_codes: dict[str, str]) -> tuple[int, int]:
        raise NotImplementedError()

        
    def count_runs(self, data: str, k: int = 4) -> tuple[int, list[tuple[int, int]]]:
        """
        Counts the number of runs of DNA bases in a given string that are at least of length k.

        Args:
            data: The string containing DNA bases.
            k: The minimum length of a run to be counted. Default is 4.

        Returns:
            The number of runs of DNA bases in the string that are at least of length k. And where they occur in the
                string, as a list of tuples of (start_index, end_index) for each run. If no runs are found, r
                returns an empty list.

        Raises:
            ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        if not data:
            return 0, []

        count: int = 0
        current_run_length: int = 1
        current_run_char: str = data[0]
        indices: list[tuple[int, int]] = []

        # Iterate through the string and count runs
        for index, letter in enumerate(data[1:]):
            # If we are not dealing with a DNA base, don't bother counting
            if letter not in self.DNA_BASES:
                return -1, []
            
            # If the current letter is the same as the last, increment the run length
            if letter == current_run_char:
                current_run_length += 1
            # If the current letter is different, check if the run length is at least k and reset
            else:
                if current_run_length >= k:
                    count += 1
                    # Append the last index of where the run
                    indices.append((index - current_run_length + 1, index))
                current_run_char = letter
                current_run_length = 1

        # Check the last run
        if current_run_length >= k:
            count += 1
            indices.append((len(data) - current_run_length + 1, len(data)))

        return count, indices


    def print_info(self,
                   encoded_data: str | list[str],
                   huffman_codes: dict[str, str],
                   original_size: int,
                   k: int = 3) -> None:
        """
        Prints the size information of the encoded data. Works on a whole file encoding (one string) or on a list of
        sequences (e.g. framed oligos), where runs are counted within each sequence so none span two sequences.

        Args:
            encoded_data: The encoded data as one string or a list of strings.
            huffman_codes: The codes the encoder used (the DNA map for DNAMap).
            original_size: The size of the original data in bits.
            k: The minimum length of a DNA run to be counted.

        Returns:
            None
        """
        sequences: list[str] = Encoding.as_sequences(encoded_data)

        # Calculate the size of the encoded data and the table
        encoded_size: int
        table_size: int
        encoded_size, table_size = self.calculate_overall_size(sequences, huffman_codes)

        print(f"{self.__class__.__name__} Encoding")
        print(f"\tEncoded data size: {encoded_size} bits")
        print(f"\tTable size: {table_size} bits")
        print(f"\tTotal size: {encoded_size + table_size} bits")

        # Standard huffman isn't directly in DNA, so there are no bases to measure
        if not all(set(sequence) <= self.DNA_BASES for sequence in sequences):
            return

        num_bases: int = sum(len(sequence) for sequence in sequences)
        if num_bases == 0:
            return

        # Count the runs and G/C bases one sequence at a time, so a run never spans two sequences
        runs: int = 0
        gc_bases: int = 0
        for sequence in sequences:
            # Count runs of length >= k in the sequence
            runs += self.count_runs(sequence, k)[0]
            # Count G/C bases for GC content calculation
            gc_bases += sequence.count('G') + sequence.count('C')

        # Calculate GC content as a percentage of total bases
        GC_content: float = gc_bases / num_bases * 100

        print(f"\tNumber of runs of length >= {k}: {runs}")
        print(f"\tNumber of bases in encoded data: {num_bases}")
        print(f"\tGC content: {GC_content:.2f}%")
        print(f"\tEffective Bits per Nucleotide: {original_size / num_bases:.2f} bits/base")