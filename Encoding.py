"""
Encoding module for DNA sequences.

This module provides functionality as the parent class for all encoding methods. It includes methods for encoding and 
decoding strings and files, as well as calculating the size of the encoded data. The module also provides a method to 
count runs of DNA bases in the encoded data. Most methods are abstract and should be implemented in subclasses that 
define specific encoding strategies.
"""

from abc import abstractmethod
from typing import Final


class Encoding:
    def __init__(self):
        """
        Initializes a new instance of the Encoding class. Sets up necessary constants and mappings for DNA encoding and 
        decoding.
        """

        # Separator to separate the table and encoded data in the output file.
        # Note: This is a theoretical implementation and should be stored differently in a real application. Speifcally,
        #   we would sotre the table as a serialized tree as bits, and have the actual encoded data as a bits. 
        #   Purely to showcase and use for testing - makes it much easier to work with when implementing DNA idea
        self.SEPARATOR: Final[str] = "=====\n"

        self.DNA_BASES: Final[set[str]] = {'A', 'C', 'G', 'T'}

        # After encoding, use the binary string to map to DNA bases with its reverse
        self.DNA_MAP: Final[dict[str, str]] = {
            '00': 'A',
            '01': 'C',
            '10': 'G',
            '11': 'T'
        }
        self.REVERSE_DNA_MAP: Final[dict[str, str]] = {v: k for k, v in self.DNA_MAP.items()}

        # Default multiplier for binary encoding (1 bit per symbol)
        self.bit_multiplier: int = 1  


    @abstractmethod
    def encode(self, string: str) -> str:
        pass


    @abstractmethod
    def encode_from_file(self, in_path: str, out_path: str) -> str:
        pass


    @abstractmethod
    def decode(self, string: str) -> str:
        pass


    @abstractmethod
    def decode_to_file(self, in_path: str, out_path: str) -> str:
        pass


    @abstractmethod
    def calculate_size(self, encoded_data: str) -> tuple[int, int]:
        pass


    def count_runs(self, data: str, k: int = 4) -> int:
        """
        Counts the number of runs of DNA bases in a given string that are at least of length k.

        Args:
            data: The string containing DNA bases.
            k: The minimum length of a run to be counted. Default is 3.

        Returns:
            The number of runs of DNA bases in the string that are at least of length k.
        """
        count: int = 0
        current_run_length: int = 1
        current_run_char: str = data[0]

        # Iterate through the string and count runs
        for letter in data:
            # If we are not dealing with a DNA base, don't bother counting
            if letter not in self.DNA_BASES:
                return 0
            
            # If the current letter is the same as the last, increment the run length
            if letter == current_run_char:
                current_run_length += 1
            # If the current letter is different, check if the run length is at least k and reset
            else:
                if current_run_length >= k:
                    count += 1
                current_run_char = letter
                current_run_length = 1

        # Check the last run
        if current_run_length >= k:
            count += 1

        return count


    def print_info(self, encoded_data: str, k: int = 4) -> None:
        """
        Prints the size information of the encoded data.

        Args:
            encoded_data: The encoded data as a string.
            k: The minimum length of a DNA run to be counted. Default is 3.

        Returns:
            None
        """
        encoded_size: int
        table_size: int
        encoded_size, table_size = self.calculate_size(encoded_data)

        print(f"{self.__class__.__name__} Encoding:")
        print(f"\tEncoded data size: {encoded_size} bits")
        print(f"\tTable size: {table_size} bits")
        print(f"\tTotal size: {encoded_size + table_size} bits")

        # Count the number of runs of length >= k in the encoded data
        runs: int = self.count_runs(encoded_data.split(self.SEPARATOR)[0].strip(), k)
        # If it is standard mapping, don't show
        if runs > 0:
            print(f"\tNumber of runs of length >= {k}: {runs}")