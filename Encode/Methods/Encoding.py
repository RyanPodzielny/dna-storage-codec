"""
Encoding module for DNA sequences.

This module provides functionality as the parent class for all encoding methods. It includes methods for encoding and 
decoding strings and files, as well as calculating the size of the encoded data. The module also provides a method to 
count runs of DNA bases in the encoded data. Most methods are abstract and should be implemented in subclasses that 
define specific encoding strategies.
"""

from abc import abstractmethod
from typing import Final, Any
import json

from Encode.Helpers.Errors import JSONError


class Encoding:
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

    # JSON keys for saving and reading encoded data
    ENCODED_DATA: Final[str] = "encoded_data"
    HUFFMAN_CODES: Final[str] = "huffman_codes"
    HAS_SINGLE_PADDING: Final[str] = "has_single_padding"
    ENCODED_TYPE: Final[str] = "encoded_type"
    INDEXER_TYPE: Final[str] = "indexer_type"

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


    @staticmethod
    def read_json_table(json_inpath: str) -> dict[str, Any]:
        """
        Reads the encoded data and padding information from a JSON file.

        Args:
            json_path: The path to the input JSON file containing the encoded data and padding information.

        Returns:
            A dictionary containing the encoded data, padding information, and Huffman codes (if applicable).

        Raises:
            FileNotFoundError: If the input JSON file does not exist.
            JSONError: If the JSON file is empty or does not contain the expected keys.
        """
        # Read the encoded data and padding information from the JSON file
        with open(json_inpath, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

            if json_data.keys() != {Encoding.ENCODED_DATA, Encoding.HUFFMAN_CODES, Encoding.HAS_SINGLE_PADDING, 
                                    Encoding.ENCODED_TYPE, Encoding.INDEXER_TYPE}:
                raise JSONError(
                    message=f"JSON file does not contain the expected keys.",
                    json_path=json_inpath
                )
            return json_data

    
    def save_json_table(self, 
                        json_path: str, 
                        encoded_data: str | list[str], 
                        huffman_codes: dict[str, str] = {},
                        has_padding: bool = False,
                        indexer_type: str = "") -> None:
        """
        Saves the encoded data and padding information to a file. Baseline for all children classes. Must have some 
        encoded data to save, otherwise raises a ValueError. All other paramters are optional, unless the encoding is 
        not a DNAMap, in which case the huffman_codes must be provided.

        Args:
            json_path: The path to the output file where the encoded data will be written.
            encoded_data: The encoded DNA sequence as a string or a list of strings (for segmented data).
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code. Default is
            has_padding: A boolean indicating whether padding was added during encoding.
            indexer_type: A string indicating the type of indexer used for encoding.

        Returns:
            None

        Raises:
            ValueError: If the encoded data is None.
            FileNotFoundError: If the output file cannot be created or opened.
        """
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")

        class_name: str = self.__class__.__name__
        if class_name != "DNAMap":
            if not huffman_codes:
                raise ValueError("Huffman codes cannot be empty for non-DNAMap encodings")

        # Write the DNA string to the output file
        with open(json_path, 'w', encoding='utf-8') as encoded_file:
            save: dict[str, Any] = {
                Encoding.ENCODED_DATA: encoded_data,
                Encoding.HUFFMAN_CODES: huffman_codes,
                Encoding.HAS_SINGLE_PADDING: has_padding,
                Encoding.INDEXER_TYPE: indexer_type,
                Encoding.ENCODED_TYPE: class_name
            }
            json.dump(save, encoded_file, indent=4)


    @abstractmethod
    def _add_padding(self, data: str, payload_length: int) -> str:
        raise NotImplementedError()
    

    @abstractmethod
    def encode_segments(self, data: str, payload_length: int, *args, **kwargs) -> tuple[dict[str, str], list[str], Any]:
        raise NotImplementedError()
    

    @abstractmethod
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        raise NotImplementedError()


    @abstractmethod
    def encode_from_file(self, in_path: str, out_path: str, as_binary: bool = False) -> str:
        raise NotImplementedError()


    @abstractmethod
    def decode_segments(self, data: list[str], *args, **kwargs) -> str:
        raise NotImplementedError()


    @abstractmethod
    def decode(self, data: str, *args, **kwargs) -> str:
        raise NotImplementedError()


    @abstractmethod
    def decode_to_file(self, in_path: str, out_path: str, as_binary: bool = False) -> str:
        raise NotImplementedError()


    @abstractmethod
    def estimate_bases_per_byte(self) -> float:
        raise NotImplementedError()


    @abstractmethod
    def calculate_overall_size(self, json_path: str) -> tuple[int, int]:
        raise NotImplementedError()


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


    def print_info(self, json_path: str, original_size: int, k: int = 3) -> None:
        """
        Prints the size information of the encoded data.

        Args:
            encoded_data: The encoded data as a string.
            original_size: The size of the original data in bits.
            k: The minimum length of a DNA run to be counted.

        Returns:
            None

        Raises:
            FileNotFoundError: If the input JSON file does not exist.
            JSONError: If the JSON file is empty or does not contain the expected keys.
        """
        json_data: dict[str, Any] = Encoding.read_json_table(json_path)
        encoded_data: str = json_data[Encoding.ENCODED_DATA]
        huffman_codes: dict[str, str] = json_data[Encoding.HUFFMAN_CODES]
        has_padding: bool = json_data[Encoding.HAS_SINGLE_PADDING]
        encoded_type: str = json_data[Encoding.ENCODED_TYPE]
        indexer_type: str = json_data[Encoding.INDEXER_TYPE]

        # Calculate the size of the encoded data and the table
        encoded_size: int
        table_size: int
        encoded_size, table_size = self.calculate_overall_size(json_path)

        print(f"{encoded_type} Encoding")
        print(f"\tEncoded data size: {encoded_size} bits")
        print(f"\tTable size: {table_size} bits")
        print(f"\tTotal size: {encoded_size + table_size} bits")
        print(f"\tIndexer type: {indexer_type}")

        if encoded_type in {"DNAMap", "DNAMapHuffman"}:
            print(f"\tPadding added: {has_padding}")

        # Standard huffman isn't directly in DNA
        if encoded_type == "Huffman":
            return

        bases: str = encoded_data
        runs: int = self.count_runs(bases, k)[0]
        GC_content: float = (bases.count('G') + bases.count('C')) / len(bases) * 100
        
        print(f"\tNumber of runs of length >= {k}: {runs}")
        print(f"\tNumber of bases in encoded data: {len(bases)}")
        print(f"\tGC content: {GC_content:.2f}%")
        print(f"\tEffective Bits per Nucleotide: {original_size / len(bases):.2f} bits/base")