"""
A class for mapping binary strings to DNA sequences and vice versa.

This module provides functionality to encode binary strings into DNA sequences and decode DNA sequences back into binary 
strings. It also includes methods for encoding and decoding files, as well as calculating the size of the encoded data.
Uses both standard DNA mapping and Huffman coding for compression.
"""

import json
from typing import Any, override
from Encoding import Encoding
from HuffmanCoding import Huffman


class DNAMap(Encoding):
    def __init__(self) -> None:
        super().__init__()

        # Each DNA base represents 2 bits, so we set the bit multiplier accordingly
        self.BIT_MULTIPLIER = 2

    @override
    def read_json_table(self, json_path: str) -> tuple[str, bool]:
        """
        Reads the encoded data and padding information from a JSON file.

        Args:
            json_path: The path to the input JSON file containing the encoded data and padding information.

        Returns:
            A tuple containing the encoded DNA sequence and a boolean indicating whether padding was added during 
            encoding.

        Raises:
            FileNotFoundError: If the input JSON file does not exist.
            ValueError: If the JSON file is empty or does not contain the expected keys.
        """
        json: dict[str, Any] = Encoding.read_json_table(json_path)
        return json[Encoding.ENCODED_DATA], json[Encoding.HAS_PADDING]

        
    @override
    def encode(self, binary_string: str) -> tuple[str, bool]:
        """
        Encodes a binary string into a DNA sequence.

        Maps each pair of bits in the binary string to a corresponding DNA base. If the length of the binary string is 
        odd, a '0' is appended to the end to make it even, and a flag is set to indicate that padding was added.

        Args:
            binary_string: The binary string to be encoded.
        
        Returns:
            A tuple containing a boolean indicating whether padding was added and the resulting DNA sequence.
        """
        # Remove any leading/trailing whitespace, causes problems
        binary_string = binary_string.strip()

        # Pad the binary string to make its length a multiple of 2
        dna_string: str = ""
        has_padding: bool = False

        # When decoding if the padding was added, we will chop it off
        if len(binary_string) % 2 != 0:
            binary_string += '0'
            has_padding = True

        # Now encode the binary string to DNA
        for index in range(0, len(binary_string), 2):
            pair: str = binary_string[index:index+2]
            dna_string += self.DNA_MAP[pair]

        return dna_string, has_padding


    @override
    def encode_from_file(self, in_path: str, json_outpath: str, as_binary: bool = False) -> str:
        """
        Encodes a binary string from a file into a DNA sequence and writes it to another file.

        Args:
            in_path: The path to the input file containing the raw text.
            json_outpath: The path to the output JSON file where the encoded data will be written.
            as_binary: A boolean indicating whether the input file is already in binary format. Default is True.

        Returns:
            The encoded DNA sequence.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is empty or if the binary string is empty.
        """
        # Read the original final
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as input_file:
            raw_text: str = input_file.read()

        # Convert the raw text to a binary string
        if not as_binary:
            binary_string: str = self.ascii_to_binary(raw_text)

        # Check if the binary string is empty
        if not binary_string:
            raise ValueError("Input file is empty.")

        # Encode the binary string to DNA
        dna_string: str
        has_padding: bool
        dna_string, has_padding = self.encode(binary_string)

        self.save_json_table(json_outpath, dna_string, has_padding=has_padding)

        return dna_string


    @override
    def decode(self, dna_string: str, has_padding: bool) -> str:
        """
        Decodes a DNA sequence back to a binary string.

        Args:
            dna_string: The DNA sequence to be decoded.
            has_padding: A boolean indicating whether padding was added during encoding.

        Returns:
            The decoded binary string.
        """
        # Decode the DNA string back to binary
        binary_string: str = ""
        for base in dna_string:
            binary_string += self.REVERSE_DNA_MAP[base]

        # If padding was added, we need to remove the last bit after decoding
        if has_padding:
            binary_string = binary_string[:-1]

        return binary_string


    @override
    def decode_to_file(self, json_path: str, out_path: str, as_binary: bool = True) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            json_path: The path to the input JSON file containing the encoded DNA sequence and padding information.
            out_path: The path to the output file where the decoded binary string will be written.

        Returns:
            The decoded binary string.
        """
        dna_string: str = ""
        has_padding: bool
        dna_string, has_padding = self.read_json_table(json_path)

        # Decode the DNA string back to binary and then to ASCII
        binary_string: str = self.decode(dna_string, has_padding)
        decoded_data: str = Encoding.binary_to_ascii(binary_string)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data


    @override
    def calculate_size(self, json_path: str) -> tuple[int, int]:
        """
        Calculates the size of the encoded data in bits.

        Args:
            json_path: The path to the JSON file containing the encoded data.

        Returns:
            The size of the encoded data in bits and the size of the table in bits.

        Raises:
            ValueError: If the encoded data format is invalid.
        """
        encoded_size: int = 0
        table_size: int = 0

        dna_string: str = ""
        has_padding: bool
        dna_string, has_padding = self.read_json_table(json_path)

        # Each DNA base represents 2 bits
        encoded_size = len(dna_string.strip()) * self.BIT_MULTIPLIER

        # 1 bit for padding stored with table
        return (encoded_size, table_size + 1)


class DNAMapHuffman(DNAMap):
    def __init__(self) -> None:
        """
        Initializes the DNAMapHuffman class.
        """
        super().__init__()


    @override
    def read_json_table(self, json_path: str) -> tuple[str, bool, dict[str, str]]:
        """
        Reads the encoded data, padding information, and Huffman codes from a JSON file.

        Args:
            json_path: The path to the input JSON file containing the encoded data, padding information, and Huffman codes.

        Returns:
            A tuple containing the encoded DNA sequence, a boolean indicating whether padding was added during encoding,
            and a dictionary of Huffman codes.

        Raises:
            FileNotFoundError: If the input JSON file does not exist.
            ValueError: If the JSON file is empty or does not contain the expected keys.
        """
        json: dict[str, Any] = Encoding.read_json_table(json_path)
        return json[Encoding.ENCODED_DATA], json[Encoding.HAS_PADDING], json[Encoding.HUFFMAN_CODES]


    @override
    def encode_from_file(self, json_path: str, out_path: str, as_binary: bool = False) -> str:
        """
        Encodes a binary string from a file into a DNA sequence and writes it to another file.

        Args:
            json_path: The path to the input file containing the binary string.
            out_path: The path to the output file where the DNA sequence will be written.
            as_binary: A boolean indicating whether to convert to binary after reading

        Returns:
            The encoded DNA sequence with tabel.

        Raises:
            ValueError: If the input file is empty or if the binary string is empty.
            FileNotFoundError: If the input file does not exist.
        """

        binary_string: str 
        has_padding: bool 
        huffman_codes: dict[str, str] | None
        binary_string, _, huffman_codes = self.read_json_table(json_path)

        if as_binary:
            binary_string = Encoding.ascii_to_binary(binary_string)

        dna_string, has_padding = self.encode(binary_string)

        self.save_json_table(out_path, dna_string, huffman_codes, has_padding)

        return dna_string


    @override
    def decode_to_file(self, json_path: str, out_path: str, as_binary: bool = False) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            json_path: The path to the input file containing the DNA sequence.
            out_path: The path to the output file where the decoded binary string will be written.

        Returns:
            The decoded binary string.

        Raises:
            ValueError: If the input format is invalid or if the Huffman table cannot be parsed
            FileNotFoundError: If the input file does not exist.
        """
        # Read the DNA string from the input file
        dna_string: str
        has_padding: bool
        huffman_codes: dict[str, str]
        dna_string, has_padding, huffman_codes = self.read_json_table(json_path)

        # Decode the DNA string back to binary then decode binary back to original
        binary_string: str = self.decode(dna_string, has_padding)
        
        # Should already be in binary, but if not, convert to binary first
        if as_binary:
            binary_string = Encoding.binary_to_ascii(binary_string)

        decoded_data: str = Huffman().decode(binary_string, huffman_codes)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data


    @override
    def calculate_size(self, json_path: str) -> tuple[int, int]:
        """
        Calculates the size of the encoded data in bits.

        Args:
            json_path: The path to the JSON file containing the encoded data.

        Returns:
            The size of the encoded data in bits and the size of the table in bits.
        """
        return Huffman().calculate_size(json_path)