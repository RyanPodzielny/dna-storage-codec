"""
A class for mapping binary strings to DNA sequences and vice versa.

This module provides functionality to encode binary strings into DNA sequences and decode DNA sequences back into binary 
strings. It also includes methods for encoding and decoding files, as well as calculating the size of the encoded data.
Uses both standard DNA mapping and Huffman coding for compression.
"""

from typing import override
from Encoding import Encoding
from HuffmanCoding import Huffman


class DNAMap(Encoding):
    def __init__(self) -> None:
        super().__init__()

        # Each DNA base represents 2 bits, so we set the bit multiplier accordingly
        self.BIT_MULTIPLIER = 2

    @override
    def encode(self, binary_string: str) -> tuple[bool, str]:
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

        return has_padding, dna_string


    @override
    def encode_from_file(self, in_path: str, out_path: str, as_binary: bool = False) -> str:
        """
        Encodes a binary string from a file into a DNA sequence and writes it to another file.

        Args:
            in_path: The path to the input file containing the raw text.
            out_path: The path to the output file where the DNA sequence will be written.
            as_binary: A boolean indicating whether the input file is already in binary format. Default is True.

        Returns:
            The encoded DNA sequence.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is empty or if the binary string is empty.
        """
        # Read the original final
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text: str = f.read()

        # Convert the raw text to a binary string
        if not as_binary:
            binary_string: str = self.ascii_to_binary(raw_text)

        # Check if the binary string is empty
        if not binary_string:
            raise ValueError("Input file is empty.")

        # Encode the binary string to DNA
        has_padding: bool 
        dna_string: str 
        has_padding, dna_string = self.encode(binary_string)

        # Format the output to include the DNA string, the Huffman table (if applicable), and the padding indicator
        output: str = dna_string + "\n"
        output += self.SEPARATOR + ("1" if has_padding else "0")

        # Write the DNA string to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(output)

        return output


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
    def decode_to_file(self, in_path: str, out_path: str, as_binary: bool = True) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            in_path: The path to the input file containing the DNA sequence.
            out_path: The path to the output file where the decoded binary string will be written.

        Returns:
            The decoded binary string.
        """
        dna_string: str = ""
        raw_padding: str = ""

        # Read the DNA string from the input file
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            data: str = f.read()
            parts: list[str] = data.split(self.SEPARATOR)

            # Something went wrong
            if len(parts) != 2:
                raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

        # Padding is always last
        dna_string = parts[0].strip()
        raw_padding: str = parts[-1].strip()

        # Make sure the padding indicator is valid
        if raw_padding not in {"0", "1"}:
            raise ValueError("Invalid padding indicator. Expected '0' or '1'.")
        has_padding: bool = raw_padding == "1"

        # Decode the DNA string back to binary and then to ASCII
        binary_string: str = self.decode(dna_string, has_padding)
        decoded_data: str = self.binary_to_ascii(binary_string)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data


    @override
    def calculate_size(self, encoded_data: str) -> tuple[int, int]:
        """
        Calculates the size of the encoded data in bits.

        Args:
            encoded_data: The encoded data as a string.

        Returns:
            The size of the encoded data in bits and the size of the table in bits.

        Raises:
            ValueError: If the encoded data format is invalid.
        """
        encoded_size: int = 0
        table_size: int = 0

        parts: list[str] = encoded_data.split(self.SEPARATOR)

        # In huffman
        if len(parts) == 3:
            table_part: str = parts[1].rstrip()
            huffman_codes: dict[str, str] = Huffman().format_table(table_part)

            # Each character is 8 bits, plus the length of the code
            for code, char in huffman_codes.items():
                table_size += 8 + len(code)
        # In dna mapping
        elif len(parts) == 2:
            pass
        else:
            raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

        # Each DNA base represents 2 bits
        encoded_size = len(parts[0].strip()) * self.BIT_MULTIPLIER

        # 1 bit for padding stored with table
        return (encoded_size, table_size + 1)


class DNAMapHuffman(DNAMap):
    def __init__(self) -> None:
        """
        Initializes the DNAMapHuffman class.
        """
        super().__init__()


    @override
    def encode_from_file(self, in_path: str, out_path: str, as_binary: bool = False) -> str:
        """
        Encodes a binary string from a file into a DNA sequence and writes it to another file.

        Args:
            in_path: The path to the input file containing the binary string.
            out_path: The path to the output file where the DNA sequence will be written.
            is_huffman: A flag indicating whether Huffman coding is used.


        Returns:
            The encoded DNA sequence with tabel.

        Raises:
            ValueError: If the input file is empty or if the binary string is empty.
            FileNotFoundError: If the input file does not exist.
        """
        # Read the binary string from the input file
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            data: str = f.read()
            binary_string: str = data.strip()

            # Should already be in binary
            if as_binary:
                binary_string = self.ascii_to_binary(binary_string)


            # Split the table and encoded data
            parts: list[str] = binary_string.split(self.SEPARATOR)
            if len(parts) != 2:
                raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

            # Get the data itself
            binary_string = parts[0].strip()
            # Get the table - will remain untouched, think of it as metadata
            raw_table: str = parts[1].rstrip()

        # Encode the binary string to DNA
        has_padding: bool 
        dna_string: str 
        has_padding, dna_string = self.encode(binary_string)

        # Format the output to include the DNA string, the Huffman table (if applicable), and the padding indicator
        output: str = dna_string + "\n"
        output += self.SEPARATOR + raw_table + "\n"
        output += self.SEPARATOR + ("1" if has_padding else "0")

        # Write the DNA string to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(output)

        return output


    @override
    def decode_to_file(self, in_path: str, out_path: str, as_binary: bool = False) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            in_path: The path to the input file containing the DNA sequence.
            out_path: The path to the output file where the decoded binary string will be written.

        Returns:
            The decoded binary string.

        Raises:
            ValueError: If the input format is invalid or if the Huffman table cannot be parsed
            FileNotFoundError: If the input file does not exist.
        """
        # Read the DNA string from the input file
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            data: str = f.read()
            parts: list[str] = data.split(self.SEPARATOR)

            # Had table, means is in huffman format
            if len(parts) != 3:
                raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

        dna_string: str = parts[0].strip()
        raw_table: str = parts[1].rstrip()
        raw_padding: str = parts[-1].strip()

        # Make sure the padding indicator is valid
        if raw_padding not in {"0", "1"}:
            raise ValueError("Invalid padding indicator. Expected '0' or '1'.")
        has_padding: bool = raw_padding == "1"

        # Format our table and ensure its proper
        huffman_codes: dict[str, str] = Huffman().format_table(raw_table)
        if raw_table and not huffman_codes:
            raise ValueError("Invalid table format. Could not parse the Huffman table.")

        # Decode the DNA string back to binary then decode binary back to original
        binary_string: str = self.decode(dna_string, has_padding)
        
        # Should already be in binary, but if not, convert to binary first
        if as_binary:
            binary_string = self.binary_to_ascii(binary_string)

        decoded_data: str = Huffman().decode(binary_string, huffman_codes)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data