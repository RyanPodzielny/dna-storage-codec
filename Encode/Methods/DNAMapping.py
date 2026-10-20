"""
A class for mapping binary strings to DNA sequences and vice versa.

This module provides functionality to encode binary strings into DNA sequences and decode DNA sequences back into binary 
strings. It also includes methods for encoding and decoding files, as well as calculating the size of the encoded data.
Uses both standard DNA mapping and Huffman coding for compression.
"""

from typing import Any, Final, override
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import Huffman


class DNAMap(Encoding):
    # Null chracter used to pad oligos for DNAMap where a empty string cannot be represenented
    CHUNKING_PADDING_CHAR: Final[str] = '\x00' 
    # For normal encoding without segmenting (sepfically for Huffman) chance we have odd bits, so need to add 1 oon
    FINAL_PADDING_CHAR: Final[str] = '0'

    def __init__(self) -> None:
        super().__init__()

        # Each DNA base represents 2 bits, so we set the bit multiplier accordingly
        self.BIT_MULTIPLIER = 2


    @override
    @staticmethod
    def read_json_table(json_path: str) -> tuple[str, bool]:
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
        return json[Encoding.ENCODED_DATA], json[Encoding.HAS_SINGLE_PADDING]


    @override
    def _add_padding(self, segment: str, payload_length: int) -> str:
        """
        Adds padding to the last segment if it is shorter than the specified payload length.

        Args:
            segment: The last segment of the encoded data.
            payload_length: The desired length of each segment.
        
        Returns:
            The segment with padding added if it was shorter than the payload length.
        """
        if len(segment) < payload_length:
            # ConveTr padding character to binary and then to DNA, we only need the DNA part
            padding: str
            padding, _ = self.encode(self.ascii_to_binary(DNAMap.CHUNKING_PADDING_CHAR))

            # Continously add the padding character until we reach the desired length
            while len(segment) < payload_length:
                segment += padding

            segment = segment[:payload_length]  # Ensure we don't exceed the payload length
            
        return segment


    @override
    def encode_segments(self, binary_string: str, payload_length: int, *args, **kwargs) -> list[str]:
        """
        Encodes a binary string into DNA segments of a specified payload length.

        Args:
            binary_string: The binary string to be encoded.
            payload_length: The length of each DNA segment.
            *args and **kwargs are used to keep polymorphism with other encoding methods

        Returns:
            A list of DNA segments, each of the specified payload length.
        """

        dna_string: str
        dna_string, _ = self.encode(binary_string)

        segments: list[str] = []
        # Split the DNA string into segments of the specified payload length
        for index in range(0, len(dna_string), payload_length):
            segment: str = dna_string[index:index + payload_length]
            segments.append(segment)
        
        # Now add padding if necessary to the last segment to ensure it is of the correct length
        segments[-1] = self._add_padding(segments[-1], payload_length)

        return segments

        
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
            binary_string += DNAMap.FINAL_PADDING_CHAR
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
    def decode_segments(self, encoded_data: list[str], *args, **kwargs) -> list[str]:
        """
        Decodes a list of DNA segments back into binary strings.

        Args:
            encoded_data: A list of DNA segments to be decoded.
            *args and **kwargs used to keep polymorphism
        
        Returns:
            A list of binary strings corresponding to the decoded DNA segments.
        """
        binary_segments: list[str] = []

        # Go through each segment and decode back to binary (decoding each ascii does not work here, need entire string)
        for segment in encoded_data:
            binary_segment: str = ""
            for base in segment:
                binary_segment += self.REVERSE_DNA_MAP[base]

            binary_segments.append(binary_segment)

        return binary_segments


    @override
    def decode(self, dna_string: str, has_padding: bool, *args, **kwargs) -> str:
        """
        Decodes a DNA sequence back to a binary string.

        Args:
            dna_string: The DNA sequence to be decoded.
            has_padding: A boolean indicating whether padding was added during encoding.
            args and kwargs are used to keep polymorphism with other encoding methods

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
    def decode_to_file(self, json_path: str, out_path: str, *args, **kwargs) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            json_path: The path to the input JSON file containing the encoded DNA sequence and padding information.
            out_path: The path to the output file where the decoded binary string will be written.
            *args and **kwargs used to keep polymorphism 

        Returns:
            The decoded binary string.
        """
        dna_string: str = ""
        has_padding: bool
        dna_string, has_padding = self.read_json_table(json_path)

        # Decode the DNA string back to binary and then to ASCII
        binary_string: str = self.decode(dna_string, has_padding=has_padding)
        decoded_data: str = Encoding.binary_to_ascii(binary_string)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data


    def calculate_num_bases(self, encoded_data: str) -> int:
        """
        Calculates the number of DNA bases in the encoded data.

        Args:
            encoded_data: The encoded DNA sequence.

        Returns:
            The number of DNA bases in the encoded data. 
        """
        return len(encoded_data.strip())


    @override
    def calculate_overall_size(self, json_path: str) -> tuple[int, int]:
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
        dna_string, _ = self.read_json_table(json_path)

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
    @staticmethod
    def read_json_table(json_path: str) -> tuple[str, dict[str, str], bool]:
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
        return json[Encoding.ENCODED_DATA], json[Encoding.HUFFMAN_CODES], json[Encoding.HAS_SINGLE_PADDING]


    @override
    def encode_segments(self, 
                        data: str, 
                        payload_length: int, 
                        *args, **kwargs) -> tuple[dict[str, str], list[str], bool]:
        """
        Encodes a binary string into DNA segments of a specified payload length using Huffman coding.

        Args:
            data: The raw ascii string to be encoded.
            payload_length: The length of each DNA segment.
            *args and **kwargs are used to keep polymorphism with other encoding methods

        Returns:
            A tuple containing a dictionary of Huffman codes and a list of DNA segments, each of the specified payload length.
        """
        huffman_codes: dict[str, str]
        encoded_data: list[str]

        # Remove 1 bit from payload length to account for padding - need to be done to ensure we have correct payload
        #   length for the sequence. No matter what we are adding padding to the end. We also multiply by 2 since we
        #   are encoding to binary first, then to DNA. This is because the payload length is in terms of DNA bases, but 
        #   we need to account for the fact that each base represents 2 bits.
        huffman_codes, encoded_data = Huffman().encode_segments(data, payload_length * self.BIT_MULTIPLIER - 1)
        for index, segment in enumerate(encoded_data):
            segment = segment + DNAMap.FINAL_PADDING_CHAR

            # Now encode the binary segment to DNA
            dna_segment: str
            dna_segment, _ = super().encode(segment)

            encoded_data[index] = dna_segment

        # Since every segment is padded, we can just return True for has_padding, no need to check
        return huffman_codes, encoded_data, True
        

    @override
    def encode(self, data: str) -> tuple[dict[str, str], str, bool]:
        """
        Encodes a binary string into a DNA sequence using Huffman coding.

        Args:
            data: The ascii string to be encoded.
            *args and **kwargs are used to keep polymorphism with other encoding methods
        
        Returns:
            A tuple containing the encoded DNA sequence, a boolean indicating whether padding was added during encoding,
            and a dictionary of Huffman codes.
        """

        huffman_codes: dict[str, str]
        encoded_data: str
        dna_string: str
        has_padding: bool

        huffman_codes, encoded_data = Huffman().encode(data)
        dna_string, has_padding = super().encode(encoded_data)

        return huffman_codes, dna_string, has_padding


    # TODO: this is kind jank, we rely on huffman coding. Think I did it to avoid double encoding while testing
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
        binary_string, huffman_codes, has_padding = self.read_json_table(json_path)

        if as_binary:
            binary_string = Encoding.ascii_to_binary(binary_string)

        dna_string, has_padding = super().encode(binary_string)

        self.save_json_table(out_path, dna_string, huffman_codes, has_padding)

        return dna_string


    @override
    def decode_segments(self, encoded_data: list[str], *args, **kwargs) -> list[str]:
        """
        Decodes a list of DNA segments back into binary. DOES NOT RETURN HUFFMAN DECODED, NEEDS TO BE DECODED AFTER.
        Since we cannot decode by segment, we need to decode the entire string after decoding the segments. The decode 
        segments part will be needed to add inner error correction in the future. Basically, to decode we combine 
        segements then perform Huffman decoding on the entire string (and remove the padding 0).

        Args:
            encoded_data: A list of DNA segments to be decoded.
            *args and **kwargs used to keep polymorphism

        Returns:
            The decoded binary strings BEFORE Huffman decoding, which can then be decoded using the Huffman codes.
        """

        return super().decode_segments(encoded_data)
        

    @override
    def decode(self, encoded_data: str, huffman_codes: dict[str, str], has_padding: bool, *args, **kwargs) -> str:
        """
        Decodes a DNA sequence back to a binary string using Huffman coding.

        Args:
            encoded_data: The DNA sequence to be decoded.
            huffman_codes: A dictionary of Huffman codes used for decoding.
            has_padding: A boolean indicating whether padding was added during encoding.
            *args and **kwargs used to keep polymorphism

        Returns:
            The decoded binary string.
        """
        binary_string: str = super().decode(encoded_data, has_padding=has_padding)
        ascii_string: str = Encoding.binary_to_ascii(binary_string)
        decoded_data: str = Huffman().decode(ascii_string, huffman_codes)

        return decoded_data


    # TODO: again very jank probably should change
    @override
    def decode_to_file(self, json_path: str, out_path: str, as_binary: bool = False) -> str: 
        """
        Decodes a DNA sequence from a file back to a binary string and writes it to another file.

        Args:
            json_path: The path to the input file containing the DNA sequence.
            out_path: The path to the output file where the decoded binary string will be written.
            as_binary: A boolean indicating whether to convert to binary after reading

        Returns:
            The decoded binary string.

        Raises:
            ValueError: If the input format is invalid or if the Huffman table cannot be parsed
            FileNotFoundError: If the input file does not exist.
        """
        # Read the DNA string from the input file
        dna_string: str
        huffman_codes: dict[str, str]
        has_padding: bool

        dna_string, huffman_codes, has_padding = self.read_json_table(json_path)

        # Decode the DNA string back to binary then decode binary back to original
        binary_string: str = super().decode(dna_string, has_padding=has_padding)
        
        # Should already be in binary, but if not, convert to binary first
        if as_binary:
            binary_string = Encoding.binary_to_ascii(binary_string)

        decoded_data: str = Huffman().decode(binary_string, huffman_codes)

        # Write the decoded data to the output file
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(decoded_data)

        return decoded_data


    @override
    def calculate_overall_size(self, json_path: str) -> tuple[int, int]:
        """
        Calculates the size of the encoded data in bits.

        Args:
            json_path: The path to the JSON file containing the encoded data.

        Returns:
            The size of the encoded data in bits and the size of the table in bits.
        """
        return Huffman().calculate_overall_size(json_path)