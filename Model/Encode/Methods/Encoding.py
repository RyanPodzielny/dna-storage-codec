"""
Encoding module for DNA sequences.

This module provides functionality as the parent class for all encoding methods. It includes methods for encoding and 
decoding strings and files, as well as calculating the size of the encoded data. The module also provides a method to 
count runs of DNA bases in the encoded data. Most methods are abstract and should be implemented in subclasses that 
define specific encoding strategies.
"""


from abc import abstractmethod
from enum import StrEnum
from typing import Final, Any


class Base(StrEnum):
    """
    A DNA base. The single place that says what the bases are - Encoding.DNA_BASES and the simulator configs both come
    from here. Using enum instead of string so we can use pydantic's built in validation instead of writing our own
    and still keep up with linting errors. 
    """
    A = "A"
    C = "C"
    G = "G"
    T = "T"



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
    # Plain strings from the Base enum, so the rest of the encoder can keep checking strings against it
    DNA_BASES: Final[set[str]] = {base.value for base in Base}
    DNA_MAP: Final[dict[str, str]] = {
        '00': Base.A.value,
        '01': Base.C.value,
        '10': Base.G.value,
        '11': Base.T.value
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


    def parameters(self) -> dict[str, Any]:
        """
        The settings that change how this encoder encodes (e.g. K for BitScanQuaternary, m for Pgram). Subclasses with
        settings override this so they show up when describing the encoder.

        Returns:
            A dictionary of setting name to value. Empty if the encoder has no settings.
        """
        return {}


    def describe(self) -> str:
        """
        Describes the encoder by its class name and its settings, e.g. "Pgram (m=8)".

        Returns:
            The description as a string.
        """
        settings: str = ", ".join(f"{name}={value}" for name, value in self.parameters().items())
        return f"{type(self).__name__} ({settings})" if settings else type(self).__name__



class EncodingStats:
    """
    Measurements of data encoded by an encoder. Works on a whole file encoding (one string) or on a list of sequences
    (e.g. framed oligos), where runs are counted within each sequence so none span two sequences. The DNA measurements
    are left at 0 when the data is not DNA (e.g. standard Huffman).
    """


    def __init__(self, encoder: Encoding, encoded_data: str | list[str], huffman_codes: dict[str, str], k: int = 3):
        """
        Measures the encoded data.

        Args:
            encoder: The encoder that encoded the data.
            encoded_data: The encoded data as one string or a list of strings.
            huffman_codes: The codes the encoder used (the DNA map for DNAMap).
            k: The minimum length of a DNA run to be counted.
        """
        sequences: list[str] = Encoding.as_sequences(encoded_data)

        self.k: int = k
        self.encoded_size: int
        self.table_size: int
        self.encoded_size, self.table_size = encoder.calculate_overall_size(sequences, huffman_codes)

        self.num_bases: int = 0
        self.runs: int = 0
        self.gc_content: float = 0.0

        # Standard huffman isn't directly in DNA, so there are no bases to measure
        if not all(set(sequence) <= Encoding.DNA_BASES for sequence in sequences):
            return

        # Count the bases, runs and G/C bases one sequence at a time, so a run never spans two sequences
        gc_bases: int = 0
        for sequence in sequences:
            self.num_bases += len(sequence)
            self.runs += encoder.count_runs(sequence, k)[0]
            gc_bases += sequence.count(Base.G.value) + sequence.count(Base.C.value)

        if self.num_bases > 0:
            self.gc_content = gc_bases / self.num_bases * 100




    def print_info(self, original_size: int, indent: str = "\t", describe: bool = False) -> None:
        """
        Prints the measurements.

        Args:
            original_size: The size of the original data in bits.
            indent: What to put before each line.

        Returns:
            None
        """
        # If describe is True, print the encoder name and settings. Otherwise, just print the measurements.
        if describe:
            print(f"Encoder: {type(self).__name__} (k={self.k})")

        print(f"{indent}Encoded data size: {self.encoded_size} bits")
        print(f"{indent}Table size: {self.table_size} bits")
        print(f"{indent}Total size: {self.encoded_size + self.table_size} bits")

        # If there are no bases (not DNA, or empty), don't bother with the DNA measurements
        if self.num_bases == 0:
            return

        print(f"{indent}Number of bases: {self.num_bases}")
        print(f"{indent}Number of runs of length >= {self.k}: {self.runs}")
        print(f"{indent}GC content: {self.gc_content:.2f}%")
        print(f"{indent}Effective bits per nucleotide: {original_size / self.num_bases:.3f} bits/base")
