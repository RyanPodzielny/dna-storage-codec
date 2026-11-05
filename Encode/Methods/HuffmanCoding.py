"""
Huffman Coding implementation for data compression. Includes classes for standard Huffman encoding and a quaternary 
version that uses DNA bases (A, C, G, T) for encoding. 

This module provides classes and methods to perform Huffman coding, a lossless data compression algorithm. It includes 
functionality for building a Huffman tree, generating Huffman codes, encoding data, and decoding encoded data. It is
purely a theoretical implementation, storing the Huffman table and encoded data in a simple JSON format, and should
be stored differently in a real application.
"""


import math
from typing import Any, Final, override
from Helpers.Errors import EncodingError, IndexingError, PayloadLengthError
from Encode.Methods.Encoding import Encoding
from Helpers.PriorityQueue import PriorityQueue


class Huffman(Encoding):
    """
    A class for performing Huffman coding on input data.
    """

    def __init__(self) -> None:
        """
        Initializes the Huffman coding class with an empty priority queue and a separator for encoding and decoding.
        """
        super().__init__()

        self.priority_queue: PriorityQueue = PriorityQueue()

        # Each character is stored as 8 bits (1 byte)
        self.KEY_MULTIPLIER: int = 8  
        # Each bit in the code
        self.VALUE_MULTIPLIER: int = 1  

        # Standard binary has 2 children
        self.CHILDREN: int = 2

        self.ORDER = Huffman.BINARY_ORDER

        self.ESTIMATED_BASES_PER_BYTE: float = 2.6


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#

    # Order for tree when Huffman is binary
    BINARY_ORDER: Final[tuple[str, ...]] = ('0', '1')
    # For converting into trits
    GOLDMAN_ORDER: Final[tuple[str, ...]] = ('0', '1', '2')
    # Order for tree when Huffman has more than up to 4 children
    TREE_ORDER: Final[tuple[str, ...]] = ('A', 'C', 'G', 'T')
    # Certain methods require a starting base to not have homopolymers
    START_BASE: Final[str] = 'A'

    # Character used to pad oligos, needs to be added to trees (preferred over null bytes as nothing is encoded)
    CHUNKING_PADDING_CHAR: Final[str] = '' 


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#

    def _count_frequencies(self, data: str, k: int = 1) -> dict[str, int]:
        """
        Counts the frequency of each character in the input data.

        Args:
            data: The input data to count character frequencies for.
            k: The length of substrings to count. Default is 1 (individual characters).
        
        Returns:
            A dictionary mapping each character to its frequency in the input data.
        """
        # Count the frequency of each character in the input data
        frequencies: dict[str, int] = {}
        for index in range(0, len(data) - k + 1, k):
            substring: str = data[index:index + k]
            if substring in frequencies:
                frequencies[substring] += 1
            else:
                frequencies[substring] = 1

        # Add remaining characters if the length of the string is not a multiple of k
        if len(data) % k != 0:
            remaining_substring: str = data[len(data) - (len(data) % k):]
            if remaining_substring in frequencies:
                frequencies[remaining_substring] += 1
            else:
                frequencies[remaining_substring] = 1
        
        return frequencies


    def _build_tree(self, data: str, k: int = 1, add_padding: bool = False) -> None:
        """
        Builds the Huffman tree based on the character frequencies in the input data.

        Uses a min heap priority queue to build the tree by merging the two smallest frequency nodes until only one node 
        remains, which becomes the root of the tree.

        Args:
            data: The input data for which the Huffman tree is to be built.
            k: The length of substrings to consider for frequency counting. Default is 1 (individual characters).
            add_padding: A boolean indicating whether to add a padding character to the frequencies. 

        Raises:
            ValueError: If the input data is empty.
        """
        frequencies: dict[str, int] = self._count_frequencies(data, k)

        # Add a padding char for encoded segementing to ensure last segment is full
        if add_padding:
            frequencies[Huffman.CHUNKING_PADDING_CHAR] = 0

        # Set up our priority queue with the frequencies
        self.priority_queue.build_queue(frequencies)
        leaves: int = self.priority_queue.size

        # The first merge takes between 2 and our children nodes so that (leaves - first_merge) is a multiple of 
        #   (childnre - 1); after that every subsequent merge takes exactly 'arity' nodes. For arity==2 this is always 2
        #   Basically ensures that our n-ary Huffman makes the first merge the small one so every other merge is full.
        take: int = 0
        if leaves >= 2:
            take = ((leaves - 2) % (self.CHILDREN - 1)) + 2
        else:
            take = self.CHILDREN

        # Merged values
        merged_frequency: int = 0
        merged_node: tuple

        # While there is more than one node in the priority queue, we need to merge nodes
        while self.priority_queue.size > 1:
            merged_frequency = 0

            children: list[tuple[int, str]] = []

            # Ensure we don't take too much
            if self.priority_queue.size < take:
                take = self.priority_queue.size

            for _ in range(take):
                if not self.priority_queue.is_empty():
                    children.append(self.priority_queue.dequeue())
                else:
                    break

            # Sum up the frequencies
            for child in children:
                merged_frequency += child[0]

            # Create a new internal node and put pack onto queue
            merged_node = (merged_frequency, tuple(children))

            self.priority_queue.enqueue(merged_node)

            # After the first merge, we always take 'arity' nodes for subsequent merges
            take = self.CHILDREN  


    def _generate_codes(self) -> dict[str, str]:
        """
        Generates the Huffman codes for each DNA base pair on the constructed quaternary Huffman tree.

        Returns:
            A dictionary mapping each character to its corresponding Huffman code {character: code}.

        Raises:
            ValueError: If the priority queue does not contain exactly one node (the root of the quaternary Huffman).
        """
        # Ensure we only have one node left
        if self.priority_queue.size != 1:
            raise ValueError("Priority queue should have exactly one node left to generate codes.")

        # Initialize an empty dictionary to hold the codes
        codes: dict[str, str] = {}

        # Helper function to recursively generate codes
        def _create_code(node: tuple, current_code: str="") -> None:
            # If we have a string it is a leaf, otherwise it is an internal node
            if isinstance(node[1], str):  # Leaf node
                codes[node[1]] = current_code
                return

            # Traverse the children, and append the proper DNA base based on the current child number
            for index, child in enumerate(node[1]):
                _create_code(child, current_code + self.ORDER[index]) 

            
        # Start the recursive code generation from the root node
        root_node = self.priority_queue.top()
        _create_code(root_node)

        return codes


    def _add_padding(self, segment: str, payload_length: int, huffman_codes: dict[str, str]) -> str:
        """
        Adds padding to the last segment if it is shorter than the specified payload length.

        Args:
            segment: The last segment of encoded data that may require padding.
            payload_length: The desired length of each segment.
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.
        
        Returns:
            The segment with added padding if necessary.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0, or if segment length is greater than 
                payload_length.
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        # This is a big problem, if segment is longer than payload it means our last segment is too long, which 
        #   should never happen.
        if len(segment) > payload_length:
            raise PayloadLengthError(
                message="Segment length must be less than or equal to payload length",
                payload_length=payload_length
            )

        # If the segment is shorter than the payload length, we need to add padding
        if len(segment) < payload_length:
            padding_length: int = payload_length - len(segment)
            segment += huffman_codes[Huffman.CHUNKING_PADDING_CHAR] * padding_length
        segment = segment[:payload_length]
        
        return segment


    def _encode_segment_payload(self,
                                current_segment: str,
                                encoded_substring: str,
                                payload_length: int,
                                segments: list[str],
                                huffman_codes: dict[str, str]) -> str:
        """
        Encodes the current segment and adds it to the list of segments. If the current segment is longer than the 
        payload length, it will be truncated and padded.

        Args:
            current_segment: The current segment of encoded data to be added to the list of segments.
            encoded_substring: The encoded substring to append to the current segment.
            payload_length: The desired length of each segment.
            segments: List of segments which current segments will be added (pass by refernece, orginal gets modified).
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0, or if current_segment length is greater 
                than payload_length.
        """
        current_segment += encoded_substring

        # If the current segment is longer than the payload length, we need to truncate it and add padding
        if len(current_segment) > payload_length:
            current_segment = current_segment[:-len(encoded_substring)]
            current_segment = self._add_padding(current_segment, payload_length, huffman_codes)
            segments.append(current_segment)
            return encoded_substring

        # If the current segment is exactly the payload length, we can add it to the list of segments
        if len(current_segment) == payload_length:
            segments.append(current_segment)
            return ""

        return current_segment


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def encode_segments(self, 
                        data: str, 
                        payload_length: int, 
                        k = 1,
                        *args, **kwargs) -> tuple[dict[str, str], list[str]]:
        """
        Encodes the given data using the generated Huffman codes into segments of a specified payload length.

        Args:
            data: The string to be encoded.
            payload_length: The length of each segment in the encoded data.

        Returns:
            A list of encoded segments as strings.

        Raises:
            PayloadLengthError: If the specified payload length is less than or equal to 0.
            ValueError: If we cannot generate valid Huffman codes.
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        huffman_codes: dict[str, str]
        encoded_data: str

        self._build_tree(data, k, add_padding=True)
        huffman_codes = self._generate_codes()

        segments: list[str] = []
        current_segment: str = ""

        # We want to ech huffman code up until the payload length, if we go over, drop, if we are exactly, add to 
        #   segments and reset
        for index in range(0, len(data) - k + 1, k):
            substring: str = data[index:index + k]
            current_segment = self._encode_segment_payload(
                current_segment=current_segment,
                encoded_substring=huffman_codes[substring],
                payload_length=payload_length,
                segments=segments,
                huffman_codes=huffman_codes,
            )

        # If not multiple of K, add the remaining characters
        if len(data) % k != 0:
            remaining_substring: str = data[len(data) - (len(data) % k):]
            current_segment = self._encode_segment_payload(
                current_segment=current_segment,
                encoded_substring=huffman_codes[remaining_substring],
                payload_length=payload_length,
                segments=segments,
                huffman_codes=huffman_codes,
            )

        # If there is any remaining data in the current segment, add it to the segments list
        if len(current_segment) > 0:
            segments.append(current_segment)
       
        # If the last segment is not the full length, add padding
        if len(segments[-1]) < payload_length:
            segments[-1] = self._add_padding(segments[-1], payload_length, huffman_codes)
        
        return huffman_codes, segments
    

    @override
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        """
        Encodes the given data using the generated Huffman codes.

        Args:
            data: The string to be encoded.
            *args and **kwargs are used to keep polymorphism with other encoding methods 

        Returns:
            A tuple containing the Huffman codes and the encoded data.

        Raises:
            ValueError: If we cannot generate valid Huffman codes.
        """

        # Build tree and generate Huffman codes
        self._build_tree(data)
        huffman_codes: dict[str, str] = self._generate_codes()
        encoded_data: str = "".join(huffman_codes[char] for char in data)

        # Return the table and encoded data as or string of encoded data
        return huffman_codes, encoded_data


    @override
    def decode_segments(self, 
                        encoded_data: list[str | None], 
                        huffman_codes: dict[str, str], 
                        *args, **kwargs) -> list[str | None]:
        """
        Decodes the encoded data using the provided Huffman codes.

        Args:
            encoded_data: A list of encoded segments to be decoded.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            *args and **kwargs are used to keep polymorphism with other encoding methods that may require additional parameters.

        Returns:
            The decoded data as a list of strings, where each string corresponds to a decoded segment.
        """
        decoded_segments: list[str | None] = []
        for segment in encoded_data:
            # Skip data we couldn't correct for
            if segment is None:
                decoded_segments.append(None)
                continue

            decoded_segment: str = self.decode(segment, huffman_codes)
            decoded_segments.append(decoded_segment)
        return decoded_segments


    @override
    def decode(self, encoded_data: str, huffman_codes: dict[str, str], reverse: bool = True) -> str:
        """
        Decodes the encoded data using the provided Huffman codes.

        Args:
            encoded_data: The encoded data to be decoded.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            reverse: A boolean indicating whether to reverse the Huffman codes for decoding.
            
        Returns:
            The decoded data as a string.
        """
        if reverse:
            huffman_codes = {v: k for k, v in huffman_codes.items()}

        # Decode the encoded data
        decoded_data: str = ""
        current_code: str = ""

        # For all our bits, try to match them to a code, match = append and reset
        for bit in encoded_data:
            current_code += bit

            # If match found, append and reset current_code
            if current_code in huffman_codes:
                decoded_data += huffman_codes[current_code]
                current_code = ""

        return decoded_data


    def estimate_bases_per_byte(self) -> float:
        """
        Estimates the number of DNA bases required to encode one byte of data using Huffman coding. Calculated based on
        the maximum from the average_bases_per_byte script.

        Returns:
            The estimated number of DNA bases per byte as a float.
        """
        return self.ESTIMATED_BASES_PER_BYTE
    

    @override
    def calculate_overall_size(self, encoded_data: str | list[str], huffman_codes: dict[str, str]) -> tuple[int, int]:
        """
        Calculates the size of the encoded data and table in bits.

        Args:
            encoded_data: The encoded data as one string or a list of strings.
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.

        Returns:
            A tuple of the size of the encoded data in bits and size of the table in bits.
        """
        # Each symbol in the encoded data is VALUE_MULTIPLIER bits
        num_symbols: int = sum(len(sequence) for sequence in Encoding.as_sequences(encoded_data))
        encoded_bits: int = num_symbols * self.VALUE_MULTIPLIER

        return encoded_bits, self.calculate_huffman_table_size(huffman_codes)


    def calculate_huffman_table_size(self, huffman_codes: dict[str, str]) -> int:
        """
        Calculates the size of the Huffman table in bits.

        Args:
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.
        Returns:
            The size of the Huffman table in bits.
        """
        table_size: int = 0
        for char, code in huffman_codes.items():
            table_size += len(char) * self.KEY_MULTIPLIER + len(code) * self.VALUE_MULTIPLIER

        return table_size



class Ternary(Huffman):
    """
    A class for performing ternary Huffman coding on input data. This class extends the Huffman class to implement a
    ternary version of Huffman coding, which uses three symbols (A, C, G) instead of the traditional binary symbols 
    (0, 1). The ternary Huffman coding prevent homopolymers greater than length 2 by not allowing a child to have the 
    same code as its parent.
    """


    def __init__(self) -> None:
        """
        Initializes a new instance of the Huffman class. Parent base != child base.
        """
        super().__init__()
        self.CHILDREN: int = 3

        self.ORDER = Huffman.TREE_ORDER

        self.ESTIMATED_BASES_PER_BYTE: float = 3.3


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def _generate_codes(self) -> dict[str, str]:
        """
        Generates the Huffman codes for each DNA base pair on the constructed ternary Huffman tree.

        Args:
            children_order: The order of the children in the Huffman tree.

        Returns:
            A dictionary mapping each character to its corresponding Huffman code.

        Raises:
            ValueError: If the priority queue does not contain exactly one node (the root of the ternary Huffman). OR
                if there are not enough characters in the tree order to assign to children.
        """
        # Ensure we only have one node left
        if self.priority_queue.size != 1:
            raise ValueError("Priority queue should have exactly one node left to generate codes.")

        # Initialize an empty dictionary to hold the codes
        codes: dict[str, str] = {}

        # Nodes are designated as {A, C, G, T} children cannot have the same code as its parent. This means no code can
        #   repeat the same character as its parent prevent long runs of the same character in the encoded data.
        #    This is important for DNA sequences, as long runs of the same base can be problematic.
        def _create_code(node: tuple, current_code: str = "", tree_order: tuple = self.ORDER[:-1]) -> None:
            # If we have a string it is a leaf, otherwise it is an internal node
            if isinstance(node[1], str):  # Leaf node
                codes[node[1]] = current_code
                return

            if current_code and current_code[-1] in tree_order:
                # If the last character of the current code is in the tree order, we need to skip it for the children
                next_tree_order = tuple(char for char in self.ORDER if char != current_code[-1])
            else:
                next_tree_order = tree_order

            # Internal node, iterate over children
            for index, child in enumerate(node[1]):
                # Assign the next character in the tree order to this child
                if index < len(next_tree_order):
                    next_char = next_tree_order[index]
                else:
                    raise ValueError("Not enough characters in tree order to assign to children.")

                _create_code(child, current_code + next_char, next_tree_order)

            
        # Start the recursive code generation from the root node
        root_node = self.priority_queue.top()
        _create_code(root_node, tree_order=self.ORDER[:-1])  # Exclude the last character for the first level

        return codes


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def estimate_bases_per_byte(self) -> float:
        """
        Estimates the number of DNA bases required to encode one byte of data using ternary Huffman coding. Calculated 
        based on the maximum from the average_bases_per_byte script.


        Returns:
            The estimated number of DNA bases per byte as a float.
        """
        return self.ESTIMATED_BASES_PER_BYTE



class Quaternary(Huffman):
    """
    This class extends the Huffman class to implement a quaternary version of Huffman coding, which uses four symbols 
    (A, C, G, T) instead of the traditional binary symbols (0, 1).
    """


    def __init__(self) -> None:
        """
        Initializes a new instance of the HuffmanQuaternary class.
        """
        super().__init__()

        # Each DNA base represents 2 bits, so we set the bit multiplier accordingly
        self.VALUE_MULTIPLIER: int = 2

        self.CHILDREN: int = 4

        self.ORDER = Huffman.TREE_ORDER

        self.ESTIMATED_BASES_PER_BYTE: float = 2.6

        """
        All encoding methods can be created with standard tree and different constants
        """


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#
        

    @override
    def estimate_bases_per_byte(self) -> float:
        """
        Estimates the number of DNA bases required to encode one byte of data using quaternary Huffman coding. 
        Calculated based on the maximum from the average_bases_per_byte script.

        Returns:
            The estimated number of DNA bases per byte as a float.
        """
        return self.ESTIMATED_BASES_PER_BYTE


class BitScanQuaternary(Quaternary):
    """
    Utilizes a quaternary Huffman coding approach with a bit-scan method to encode data. This class allows for
    scanning a specified number of bits (k) at a time, which can improve encoding efficiency but can drastically 
    increase the size of the Huffman table.
    """


    def __init__(self, k: int = 4) -> None:
        """
        Initializes a new instance of the OptimizedQuaternary class.

        Args:
            k: The length of bits to scan for encoding.
        """
        super().__init__()

        # Length of bits to scan
        self.K: int = k

        # Each character is stored as a bit
        self.KEY_MULTIPLIER: int = 1
        # Each value (DNA base) is stored as 2 bits
        self.VALUE_MULTIPLIER: int = 2

        # Assume max, we will use function regardless
        self.ESTIMATED_BASES_PER_BYTE: float = max(BitScanQuaternary.BITSCAN_BASES_PER_BYTE.values())


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   GETTER METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this instance when loading a saved framer.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {"k": self.K}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Measured bases per byte by bit-scan length maximum, rounded up I know constants aren't great. But easiest way
    #   to get a good estimate of the average bases per byte for each bit-scan length. They are also pretty innacurate
    #   the larger you go (just by variance of the data) but are a good starting point to determine width. Worst case
    #   is we need to regenerate the encoded segs with a new width if we are too innacurate with num oligos.
    BITSCAN_BASES_PER_BYTE: Final[dict[int, float]] = {
        4: 3.7, 5: 4.1, 6: 3.9, 7: 4.0, 8: 2.6, 9: 3.8, 10: 3.4, 11: 3.5,
        12: 2.8, 13: 3.3, 14: 2.9, 15: 3.0, 16: 2.1, 17: 2.9, 18: 2.6, 19: 2.7,
        20: 2.2, 21: 2.5, 22: 2.3, 23: 2.4, 24: 1.8, 25: 2.3, 26: 2.1, 27: 2.2,
        28: 1.9, 29: 2.1, 30: 1.9, 31: 2.0, 32: 1.6, 33: 1.9, 34: 1.8, 35: 1.8,
        36: 1.6, 37: 1.7, 38: 1.6, 39: 1.6, 40: 1.4, 41: 1.6, 42: 1.5, 43: 1.5,
        44: 1.4, 45: 1.4, 46: 1.4, 47: 1.4, 48: 1.3, 49: 1.3, 50: 1.3, 51: 1.3,
        52: 1.2
    }


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _calculate_coding_density(self, huffman_codes: dict[str, str]) -> float:
        """
        Calculates the coding density of the encoded data.

        Args:
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.
            encoded_data: The encoded data as a string.
            original_size: The size of the original data in bits.

        Returns:
            The coding density as a float.
        """
        total_code_length: int = sum(len(code) for code in huffman_codes.values())
        average_code_length: float = total_code_length / len(huffman_codes)
        
        coding_density: float = self.K / average_code_length
        return coding_density


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def encode_segments(self, data: str, payload_length: int, *args, **kwargs) -> tuple[dict[str, str], list[str]]:
        # Data should be in bit form
        data = Encoding.ascii_to_binary(data)
        return super().encode_segments(data, payload_length, self.K, *args, **kwargs)


    @override
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        """
        Encodes the given data using the generated Huffman codes.

        Args:
            data: The string to be encoded.
            *args and **kwargs are used to keep polymorphism with other encoding methods

        Returns:
            A tuple containing the Huffman codes and the encoded data.

        Raises:
            ValueError: If we cannot generate valid Huffman codes.
        """

        # Build tree and generate Huffman codes
        self._build_tree(data, self.K)
        huffman_codes: dict[str, str] = self._generate_codes()
        encoded_data: str = ""

        # Encode the data using the generated Huffman codes
        for index in range(0, len(data) - self.K + 1, self.K):
            substring: str = data[index:index + self.K]
            if substring in huffman_codes:
                encoded_data += huffman_codes[substring]

        # Add remaining characters if the length of the data is not a multiple of k
        if len(data) % self.K != 0:
            remaining_substring: str = data[len(data) - (len(data) % self.K):]
            if remaining_substring in huffman_codes:
                encoded_data += huffman_codes[remaining_substring]

        # Return the table and encoded data as or string of encoded data
        return huffman_codes, encoded_data


    @override 
    def decode_segments(self, 
                        encoded_data: list[str | None], 
                        huffman_codes: dict[str, str], 
                        *args, **kwargs) -> list[str | None]:
        # Get our decoded segments (will be in binary form)
        decoded_segments: list[str | None] = super().decode_segments(encoded_data, huffman_codes, *args, **kwargs)
        
        ascii_segments: list[str | None] = []
        remainder: str = ""

        # For every segment, append the remainder from the last segment and convert to ascii, then save the new 
        #   remainder for the next segment
        for segment in decoded_segments:
            # Skip undecodable segments, we will just return None for those
            if segment is None:
                ascii_segments.append(None)
                continue

            decoded_data: str = remainder + segment

            # Gets the length of the complete bits (multiple of 8)
            complete_bit_length: int = len(decoded_data) - (len(decoded_data) % 8)

            # Get up to the complete bits and convert to ascii, append to the list of ascii segments
            ascii_segments.append(Encoding.binary_to_ascii(decoded_data[:complete_bit_length]))

            # Save the remainder for the next segment
            remainder = decoded_data[complete_bit_length:]

        return ascii_segments


    def find_optimal_k(self, in_path: str, min_k: int = 4, max_k: int = 52) -> tuple[int, float, dict[str, str], str]:
        """
        Finds the optimal value of k that maximizes the coding density for the given data, and sets the class's K 
        attribute to that value.

        Args:
            in_path: The path to the input file.
            min_k: The minimum value of k to consider. Default is 4.
            max_k: The maximum value of k to consider. Default is 16.

        Returns:
            A tuple containing the optimal value of k, the maximum coding density, the best Huffman codes, and the 
            best encoded data.

        Raises:
            ValueError: If the input data is empty or if min_k is greater than max_k. Or if we cannot generate valid
                Huffman codes.
            FileNotFoundError: If the input file does not exist.
        """
        with open(in_path, 'r', encoding='utf-8') as f:
            data: str = f.read()

        optimal_k: int = min_k
        max_density: float = 0.0

        if min_k > max_k:
            raise ValueError("min_k cannot be greater than max_k.")

        for k in range(min_k, max_k + 1):
            self.K = k
            huffman_codes, encoded_data = self.encode(data)
            density: float = self._calculate_coding_density(huffman_codes)

            if density > max_density:
                max_density = density
                optimal_k = k
                best_huffman_codes = huffman_codes
                best_encoded_data = encoded_data

        return optimal_k, max_density, best_huffman_codes, best_encoded_data


    @override
    def estimate_bases_per_byte(self) -> float:
        """
        Estimated bases per byte for this scanner's K (self.K).

        The rate is not smooth in K: values that are multiples of 8 align with character boundaries and do much better 
        (K=8 is ~2.4 but K=7 and K=9 are ~3.9 and ~3.6). So this uses a measured table where K is covered, and otherwise
        a guaranteed upper bound: a quaternary Huffman code over K-bit symbols averages at most ceil(K / 2) bases per 
        symbol, with 8 / K symbols per byte. The bound can overshoot by 1.2-1.7x.

        For large K the real rate also depends on file size (more distinct K-bit symbols as the file grows).
        """
        if self.K in BitScanQuaternary.BITSCAN_BASES_PER_BYTE:
            return BitScanQuaternary.BITSCAN_BASES_PER_BYTE[self.K]
        return (8 / self.K) * math.ceil(self.K / 2)


    def print_info(self, 
                   encoded_data: str | list[str], 
                   huffman_codes: dict[str, str], 
                   original_size: int, 
                   k: int = 3) -> None:
        super().print_info(encoded_data, huffman_codes, original_size, k)
        print(f"\tK Used: {self.K}")



class Pgram(Huffman):
    """
    A class for performing p-gram Huffman coding on input data. This class extends the Huffman class to implement a
    p-gram version of Huffman coding, which uses variable-length substrings (p-grams) of the input data to create a more
    efficient encoding. The maximum length of p-grams can be specified during initialization, allowing for flexibility 
    in the encoding process.
    """


    def __init__(self, m: int = 3) -> None:
        """
        Initializes a new instance of the Pgram class.

        Args:
            m: The maximum length of p-grams to consider for encoding

        Raises:
            ValueError: If the maximum p-gram length is less than 1.
        """
        super().__init__()

        # Max length of pgram
        self.M = m  

        if self.M < 1:
            raise ValueError("Maximum p-gram length must be greater than 0")

        self.EVEN: tuple[str, str] = ('G', 'T')
        self.ODD: tuple[str, str] = ('A', 'C')

        self.COMPLEMENT: dict[str, str] = {'A': 'G', 'C': 'T',
                                           'G': 'A', 'T': 'C'}

        # Will use function, placeholder
        self.ESTIMATED_BASES_PER_BYTE: float = max(Pgram.PGRAM_BASES_PER_BYTE.values())


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   GETTER METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this instance when loading a saved framer.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {"m": self.M}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Maximum (rounded up) measured bases per byte by max p-gram length
    PGRAM_BASES_PER_BYTE: Final[dict[int, float]] = {
        2: 4.5, 3: 4.0, 4: 3.5, 5: 3.2, 6: 2.9, 7: 2.7, 8: 2.5, 9: 2.3,
        10: 2.1, 11: 1.9, 12: 1.8, 13: 1.7, 14: 1.6, 15: 1.5, 16: 1.4,
    }


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#
        

    @override
    def _add_padding(self, segment: str, payload_length: int, huffman_codes: dict[str, str]) -> str:
        """
        Adds padding to the last segment if it is shorter than the specified payload length.

        Args:
            segment: The last segment of encoded data that may require padding.
            payload_length: The desired length of each segment.
            huffman_codes: A dictionary mapping each character to its corresponding Huffman code.
        
        Returns:
            The segment with added padding if necessary.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0, or if segment length is greater than 
                payload_length.
            ValueError: If segment length is empty
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        if len(segment) > payload_length:
            raise PayloadLengthError(
                message="Segment length must be less than or equal to payload length",
                payload_length=payload_length
            )

        if len(segment) == 0:
            raise ValueError("Segment length must be greater than 0")

        # Get the padding character from the Huffman codes, and its complement for the complementary tree
        padding_code: str = huffman_codes[Huffman.CHUNKING_PADDING_CHAR]
        complement_code: str = ""
        for base in padding_code:
            complement_code += self.COMPLEMENT[base]

        # Same as _build_optimal_coding: after an even base (G/T) the next code starts complemented (A/C)
        is_complementary: bool = segment[-1] in self.EVEN

        padding: str = ""
        while len(segment) + len(padding) < payload_length:
            code: str = complement_code if is_complementary else padding_code
            padding += code

            # An odd length code ends on the same level it started, so the next copy has to swap - otherwise repeating
            #   the same copy can put the same base on both sides of the boundary (a homopolymer)
            if len(code) % 2 == 1:
                is_complementary = not is_complementary

        segment = (segment + padding)[:payload_length]

        return segment


    @override
    def _count_frequencies(self, data: str, a: int = 10) -> dict[str, int]:
        """
        Counts the frequency of each k-gram (substring of length k) in the input data.

        Args:
            data: The input data to count k-gram frequencies for.
            a: The scaling factor for the pgram length. Higher lengths will be scaled higher.

        Returns:
            A dictionary mapping each k-gram to its frequency in the input data.
        """

        frequencies: dict[str, int] = {}

        # Length of the code word we are scanning for
        clength: int = 1
        # Position in the data 
        index: int = 0

        # For every possible length of pgram, count the frequencies of all substrings of that length
        for clength in range(self.M + 1):
            # At every position, record the substring of the current length and scle
            index = 0
            for index in range(len(data) - clength + 1):
                substring: str = data[index:index + clength]
                # Scaling used is 10 ^ (code_length - 1) from:
                #   https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11079985
                if substring not in frequencies:
                    frequencies[substring] = clength * (a ** (clength - 1))
                else:
                    frequencies[substring] += clength * (a ** (clength - 1))

        return frequencies


    @override
    def _generate_codes(self) -> dict[str, str]:
        """
        Generates the Huffman codes for each character based on the constructed Huffman tree. Odd levels are
        assigned 'A' and 'C', while even levels are assigned 'G' and 'T'. The codes are generated by traversing the tree
        from the root to each leaf node.

        Returns:
            A dictionary mapping each character to its corresponding Huffman code.

        Raises:
            ValueError: If the priority queue does not contain exactly one node (the root of the Huffman tree).
        """

        # Ensure we only have one node left
        if self.priority_queue.get_size() != 1:
            raise ValueError("Priority queue should have exactly one node left to generate codes.")

        # Initialize an empty dictionary to hold the codes
        codes: dict[str, str] = {}

        # Helper function to recursively generate codes
        def _create_code(node: tuple, current_code: str = "", level: int = 0) -> None:
            # If we have a string it is a leaf, otherwise it is an internal node
            if isinstance(node[1], str):  # Leaf node
                codes[node[1]] = current_code
                return

            code: tuple[str, str] = self.EVEN if level % 2 == 0 else self.ODD

            # Internal node, traverse left and right
            _create_code(node[1][0], current_code + code[0], level + 1)  # Left child
            _create_code(node[1][1], current_code + code[1], level + 1)  # Right child

        # Start the recursive code generation from the root node
        root_node = self.priority_queue.top()
        _create_code(root_node)

        return codes


    def _build_optimal_coding(self, 
                             data: str, 
                             huffman_codes: dict[str, str],
                             payload_length: int | None = None,
                             segment_flag: bool = False,
                             previous_base: str = Huffman.START_BASE) -> tuple[str, int]:
        """
        Builds the optimal coding for the given data using p-gram encoding. Essentially, for every position in the data,
        we find the substring of length 1 to M that has the best ratio of length of substring to length of code. 
        We then append the code for that substring to the encoded message.

        Args:
            data: The segement of data we are encoding.
            huffman_codes: A dictionary mapping each p-gram to its corresponding Huffman code.
            payload_length: The length of the payload.
            segment_flag: A boolean indicating whether we are encoding a segment of data.
            previous_base: The previous base.

        Returns:
            A tuple containing the encoded data as a string and the index of the last processed character.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0.
            ValueError: If previous base is not 1 and not in DNA sets
            KeyError: If a substring is not found in the Huffman codes.
        """
        if len(previous_base) != 1 or previous_base not in Encoding.DNA_BASES:
            raise ValueError("Previous base must be a single character and one of the DNA bases.")

        if not payload_length:
            payload_length = len(data)

        if payload_length is not None and payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )
        
        coded_message: str = ""
        index: int = 0

        # Check to see if we are starting with an even or odd level based on the previous base
        is_complementary: bool = False if previous_base in self.ODD else True

        # For every pos in data, find the substring of length 1 to M that has best ration 
        while index < len(data):
            # Reset ratio and substring length for this position
            ratio: float = 0
            substring_length: int = 0

            # Find the best substring for this position
            for clength in range(1, self.M + 1):
                # If the substring length goes beyond stopping lengths, stop checking for longer substrings
                if index + clength > payload_length or index + clength > len(data):
                    break

                # Get the substring
                substring: str = data[index:index + clength]
                # Get the code for the substring
                code: str = huffman_codes[substring]
                # Calculate the ratio of length of substring to length of code
                current_ratio: float = len(substring) / len(code)

                # See if we have new best
                if current_ratio > ratio:
                    ratio = current_ratio
                    substring_length = clength

            lookup: str = data[index:index + substring_length]
            huffman_code: str = huffman_codes[lookup]

            if is_complementary:
                huffman_code = "".join(self.COMPLEMENT[char] for char in huffman_code)

            # We start with even level, unless if previous level was odd we swap
            if len(huffman_code) % 2 == 1:
                is_complementary = not is_complementary

            # Add to the coded message
            coded_message += huffman_code
            if segment_flag and len(coded_message) > payload_length:
                coded_message = coded_message[:payload_length]
                return coded_message, index

            # Move the index forward by the length of the substring we just encoded
            index += substring_length

        return coded_message, index


    def _get_codes(self, huffman_codes: dict[str, str], reverse: bool = True) -> tuple[dict[str, str], dict[str, str]]:
        """
        Reverses the huffman if needed and retreives the complementary mapping of the Huffman codes. Helper to decode
        normally and by segements.

        Args:
            huffman_codes: The input code to be complemented.
            reverse: A boolean indicating whether to reverse the Huffman codes for decoding.
        
        Returns:
            A tuple containing the (possibly reversed) Huffman codes and the complementary mapping of the Huffman codes.
        """
        if reverse:
            huffman_codes = {v: k for k, v in huffman_codes.items()}

        # Create a complementary mapping of the Huffman codes to serve as complementary tree
        complement_codes: dict[str, str] = {}
        for code, char in huffman_codes.items():
            complement_codes["".join(self.COMPLEMENT[base] for base in code)] = char

        return huffman_codes, complement_codes


    def _decode_part(self, huffman_codes: dict[str, str], complement_codes: dict[str, str], encoded_data: str) -> str:
        """
        Decodes a part of the encoded data using the provided Huffman codes and their complementary mapping.

        Args:
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            complement_codes: A dictionary mapping complementary Huffman codes to their corresponding characters.
            encoded_data: The encoded data to be decoded.

        Returns:
            The decoded data as a string.
        """
        decoded_data: str = ""
        current_code: str = ""

        # For all our bits, try to match them to a code, match = append and reset
        for base in encoded_data:
            current_code += base

            is_even: bool = current_code[0] in self.EVEN

            # If match found, append and reset current_code
            if is_even and current_code in huffman_codes:
                decoded_data += huffman_codes[current_code]
                current_code = ""
            elif not is_even and current_code in complement_codes:
                decoded_data += complement_codes[current_code]
                current_code = ""

        return decoded_data


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def encode_segments(self, 
                        data: str, 
                        payload_length: int, 
                        *,
                        previous_sequences: list[str]) -> tuple[dict[str, str], list[str]]:
        """
        Encodes the given data into segments using the Huffman coding method.

        Args:
            data: The input data to be encoded.
            payload_length: The length of each payload segment.
            previous_sequences: A list of previous sequences (like the index or primer) for the mapping.

        Returns:
            A tuple containing the Huffman codes and the list of encoded segments.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0.
            IndexingError: If there are not enough previous sequences to cover the number of segments.
            ValueError: If we cannot generate valid Huffman codes.
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        # Build our trees and get our codes
        self._build_tree(data, add_padding=True)
        huffman_codes: dict[str, str] = self._generate_codes()

        segments: list[str] = []

        data_length: int = len(data)

        encoded_message: str
        last_data_index: int = 0
        bases_index: int = 0

        # While we still have data to encode, build the optimal coding for next segment and append it
        while data_length > 0:
            # IF we go out of bounds we need to know that we don't have enough previous sequences (indices) available
            if bases_index >= len(previous_sequences):
                raise IndexingError(
                    message="Not enough previous sequences provided for the number of segments",
                    index=bases_index
                )

            # Build the optimal coding for the next segment of data
            encoded_message, last_data_index = self._build_optimal_coding(
                data[len(data) - data_length:], 
                huffman_codes, 
                payload_length, 
                True,
                previous_sequences[bases_index][-1]
            )
            segments.append(encoded_message)
            bases_index += 1

            # Keep our data length check seperated to avoid issues with the last segment being shorter than 
            #   payload length creating an infinite loop
            data_length -= last_data_index

        # Ensure our last segment is our payload length
        segments[-1] = self._add_padding(segments[-1], payload_length, huffman_codes)

        return huffman_codes, segments
    

    @override
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        """
        Finds the optimal coding for the given data using p-gram encoding.

        Args:
            data: The input data to be encoded.
            args and **kwargs are used to keep polymorphism with other encoding methods

        Returns:
            A tuple containing the Huffman codes and the encoded data.

        Raises:
            ValueError: If we cannot generate valid Huffman codes or if the input data is empty.
            PayloadLengthError: If data is less than or equal to 0.
            KeyError: If a substring is not found in the Huffman codes.
        """
        self._build_tree(data)
        huffman_codes: dict[str, str] = self._generate_codes()
        encoded_message: str 
        encoded_message, _ = self._build_optimal_coding(data, huffman_codes)
        return huffman_codes, encoded_message
    

    @override
    def decode_segments(self, 
                        encoded_data: list[str | None], 
                        huffman_codes: dict[str, str], 
                        reverse: bool = True,
                        *args, **kwargs) -> list[str | None]:
        """
        Decodes the encoded data using the provided Huffman codes.

        Args:
            encoded_data: A list of encoded segments to be decoded.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            reverse: A boolean indicating whether to reverse the Huffman codes for decoding.
            args and **kwargs are used to keep polymorphism with other encoding methods.

        Returns:
            The decoded data as a list of strings, where each string corresponds to a decoded segment.
        """

        # Can't just call decode because creating he complementary mapping takes a bunch of time and we don't want to
        #   do that for every single segment
        complement_codes: dict[str, str] = {}
        huffman_codes, complement_codes = self._get_codes(huffman_codes, reverse=reverse)
        
        decoded_data: list[str | None] = []

        # For every segment, decode it and append to the decoded data
        for segment in encoded_data:
            # Skip data we couldn't error correct on
            if segment is None:
                decoded_data.append(None)
                continue

            decoded_segment: str = self._decode_part(huffman_codes, complement_codes, segment)
            decoded_data.append(decoded_segment)

        return decoded_data


    @override
    def decode(self, encoded_data: str, huffman_codes: dict[str, str], reverse: bool = True) -> str:
        """
        Decodes the encoded data using the provided Huffman codes. Uses a complementary mapping of the Huffman codes to 
        serve as a complementary tree for decoding. As shown in paper: 
        https://www.researchgate.net/publication/341064927_Compressed_DNA_Coding_Using_Minimum_Variance_Huffman_Tree

        Args:
            encoded_data: The encoded data to be decoded.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            reverse: A boolean indicating whether to reverse the Huffman codes for decoding.

        Returns:
            The decoded data as a string.
        """
        # Create a complementary mapping of the Huffman codes to serve as complementary tree, and update huffman
        complement_codes: dict[str, str] = {}
        huffman_codes, complement_codes = self._get_codes(huffman_codes, reverse=reverse)

        # Actually decode with the codes and complementary codes
        decoded_data: str = self._decode_part(huffman_codes, complement_codes, encoded_data)

        return decoded_data


    @override
    def estimate_bases_per_byte(self) -> float:
        """
        Estimated bases per byte for this Pgram's max p-gram length (self.M).

        The rate drops as m grows, since longer p-grams get shorter codes per character. For m above the measured range 
        this returns the m=6 value, which is conservative because the rate keeps falling.

        Returns:
            The estimated number of DNA bases per byte as a float.

        Raises:
            ValueError: If self.M is less than 1.
        """
        if self.M < 1:
            raise ValueError("Pgram m must be at least 1.")
        return Pgram.PGRAM_BASES_PER_BYTE[min(self.M, max(Pgram.PGRAM_BASES_PER_BYTE))]


    def print_info(self, 
                   encoded_data: str | list[str], 
                   huffman_codes: dict[str, str], 
                   original_size: int, 
                   k: int = 2) -> None:
        super().print_info(encoded_data, huffman_codes, original_size, k)
        print(f"\tM Used: {self.M}")



class Goldman(Huffman):
    """
    Class for performing Goldman's rotating tree Huffman coding on input data. This class extends the Huffman class to
    implement a rotating tree version of Huffman coding, which uses a rotating mapping of DNA bases to create a more
    efficient encoding. The rotating mapping is based on the previous base used in the encoding process, ensuring that
    no homopolymers are created in the encoded DNA sequence. 
    """


    def __init__(self) -> None:
        """
        Initializes a new instance of the Goldman class.
        """
        super().__init__()

        self.CHILDREN: int = 3

        self.ORDER = Huffman.GOLDMAN_ORDER

        self.ESTIMATED_BASES_PER_BYTE: float = 3.2


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    ROTATING_MAP: Final[dict[str, dict[str, str]]] = {
        'A': {'0': 'C', '1': 'G', '2': 'T'},
        'C': {'0': 'G', '1': 'T', '2': 'A'},
        'G': {'0': 'T', '1': 'A', '2': 'C'},
        'T': {'0': 'A', '1': 'C', '2': 'G'}
    }
    # Construct the complementary rotating map
    ROTATING_MAP_COMPLEMENT: Final[dict[str, dict[str, str]]] = {}
    for base, mapping in ROTATING_MAP.items():
        ROTATING_MAP_COMPLEMENT[base] = {v: k for k, v in mapping.items()}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#

    def trit_encode(self, data: str) -> str:
        """
        Encodes the given data into trits using the generated Huffman codes.

        Args:
            data: The string to be encoded.

        Returns:
            The encoded data as a string of trits.

        Raises:
            ValueError: If we cannot generate valid Huffman codes.
        """
        self._build_tree(data)
        huffman_codes: dict[str, str] = self._generate_codes()
        encoded_data: str = "".join(huffman_codes[char] for char in data)

        return encoded_data


    def dna_to_trit(self, encoded_data: str, start_base: str = Huffman.START_BASE) -> str:
        """
        Maps the DNA bases back to trits using the rotating map complement based on the previous base.

        Args:
            encoded_data: The DNA data to be mapped.
            start_base: The starting base for the mapping.

        Returns:
            The a string of the mapped trits.

        Raises:
            EncodingError: If a base is not found in the rotating map complement or if the start base is not valid.
        """
        trit_string: str = ""
        # Always assume we start with the specified base
        previous_base: str = start_base

        for base in encoded_data:
            if base not in Goldman.ROTATING_MAP_COMPLEMENT[previous_base]:
                raise EncodingError(
                    message=f"Base '{base}' not found in the rotating map complement for base '{previous_base}'",
                    encoding_method=self.__class__.__name__
                )

            # Get the corresponding trit from the rotating map complement
            trit: str = Goldman.ROTATING_MAP_COMPLEMENT[previous_base][base]
            trit_string += trit

            # Update the previous base for the next iteration
            previous_base = base

        return trit_string


    def trit_to_dna(self, trit_data: str, start_base: str = Huffman.START_BASE) -> str:
        """
        Maps the trit data to DNA bases using the rotating map based on the previous base.

        Args:
            trit_data: The trit data to be mapped.
            start_base: The starting base for the mapping.

        Returns:
            The a string of the mapped DNA bases with no homopolymers.

        Raises:
            EncodingError: If a trit is not found in the rotating map or if the start base is not valid.
            ValueError: If the start base is not a valid DNA base for the rotating map.
        """
        if start_base not in Goldman.ROTATING_MAP:
            raise ValueError(f"Start base '{start_base}' is not a valid DNA base for the rotating map.")

        dna_string: str = ""
        # Always assume we start with the specified base
        previous_base: str = start_base

        for trit in trit_data:
            if trit not in Goldman.ROTATING_MAP[previous_base]:
                raise EncodingError(
                    message=f"Trit '{trit}' not found in the rotating map for base '{previous_base}'",
                    encoding_method=self.__class__.__name__
                )

            # Get the next base from the rotating map
            next_base: str = Goldman.ROTATING_MAP[previous_base][trit]
            dna_string += next_base

            # Update the previous base for the next iteration
            previous_base = next_base

        return dna_string


    @override
    def encode_segments(self, 
                        data: str, 
                        payload_length: int,
                        *,
                        previous_sequences: list[str]) -> tuple[dict[str, str], list[str]]:
        """
        Encodes the given data with Goldman's rotating tree approach using the generated Huffman codes and a specified
        starting base into segments of a specified payload length.

        Args:
            data: The string to be encoded.
            payload_length: The length of each segment in the encoded data.
            previous_sequences: A list of previous sequences (like the index or primer) for the mapping.

        Returns:
            A tuple containing the Huffman codes and the encoded data.

        Raises:
            PayloadLengthError: If payload_length is less than or equal to 0.
            ValueError: If the length of the previous bases list is less than the number of segments. Or if we cannot 
                generate valid Huffman codes. Or if the previous base is not a valid DNA base for the rotating map.
            IndexingError: If there are not enough previous sequences or if the sequence length is too small.
            EncodingError: If a trit is not found in the rotating map or if the start base is not valid.
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        self._build_tree(data, add_padding=True)
        huffman_codes: dict[str, str] = self._generate_codes()

        trit_segments: list[str] = []
        current_segment: str = ""
        # For every character in the data, encode it and add it to the current segment. If the current segment is full,
        #   append it to the list of segments and start a new segment.
        for char in data:
            current_segment = self._encode_segment_payload(
                current_segment=current_segment,
                encoded_substring=huffman_codes[char],
                payload_length=payload_length,
                segments=trit_segments,
                huffman_codes=huffman_codes,
            )

        # If any remaining add
        if len(current_segment) > 0:
            trit_segments.append(current_segment)

        # And add some paddding
        trit_segments[-1] = self._add_padding(trit_segments[-1], payload_length, huffman_codes)

        # Now convert to trits, to dna, and append to the encoded message list
        encoded_message: list[str] = []
        for index, trit_segment in enumerate(trit_segments):
            # Ensure we have enough previous sequences to cover the number of segments we are encoding, and the sequence
            if index >= len(previous_sequences) or not previous_sequences[index]:
                raise IndexingError(
                    message="Not enough previous sequences provided for the number of segments to encode or sequence " \
                            "length is too small",
                    index=index
                )

            # Map back to dna and append
            mapped_dna: str = self.trit_to_dna(trit_segment, start_base=previous_sequences[index][-1])
            encoded_message.append(mapped_dna)

        return huffman_codes, encoded_message


    @override
    def encode(self, data: str) -> tuple[dict[str, str], str]:
        """
        Encodes the given data with Goldman's rotating tree approach using the generated Huffman codes.

        Args:
            data: The string to be encoded.
            *args and **kwargs are used to keep polymorphism with other encoding methods

        Returns:
            A tuple containing the Huffman codes and the encoded data.

        Raises:
            ValueError: If we cannot generate valid Huffman codes. Or if the stat base is not a valid DNA base for 
                the rotating map. Or if the input data is empty.
            EncodingError: If a trit is not found in the rotating map or if the start base is not valid.
        """
        self._build_tree(data)
        huffman_codes: dict[str, str] = self._generate_codes()

        encoded_message: str = "".join(huffman_codes[char] for char in data)
        encoded_message = self.trit_to_dna(encoded_message)

        return huffman_codes, encoded_message


    @override
    def decode_segments(self, 
                        encoded_data: list[str | None], 
                        huffman_codes: dict[str, str],
                        *,
                        previous_sequences: list[str]) -> list[str | None]:
        """
        Decodes the encoded data with Goldman's rotating tree approach using the provided Huffman codes and a specified
        starting base.

        Args:
            encoded_data: The encoded data to be decoded as a list of strings.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            previous_sequences: A list of previous sequences (like the index or primer) for the mapping.

        Returns:
            A list of strings, where each string corresponds to a decoded segment. If it couldnt be decoded, it is None.

        Raises:
            EncodingError: If a base is not found in the rotating map complement or if the start base is not valid.
            IndexingError: If there are not enough previous sequences or if the sequence length is too small.
        """

        decoded_segments: list[str | None] = []

        # Append the trit segments to a list with being aware what starting base we are using for each segment
        for index, segment in enumerate(encoded_data):
            if segment is None:
                decoded_segments.append(None)
                continue

            # Must have enough previous sequences to cover the number of segments we are decoding, and the sequence 
            #   length must be longer than 0 so we can get last base
            if index >= len(previous_sequences) or 0 >= len(previous_sequences[index]):
                raise IndexingError(
                    message="Not enough previous sequences provided for the number of segments to decode or sequence "\
                            "length is too small",
                    index=index
                )

            previous_base: str = previous_sequences[index][-1]
            trit_segment: str = self.dna_to_trit(segment, start_base=previous_base)

            # Now decode each trit segment with the huffman codes, and append to the decoded segments list
            decoded_segment: str = super().decode(trit_segment, huffman_codes, reverse=True)
            decoded_segments.append(decoded_segment)

        return decoded_segments


    @override
    def decode(self, encoded_data: str, huffman_codes: dict[str, str], reverse: bool = True) -> str:
        """
        Decodes the encoded data using the provided Huffman codes and Goldman's rotating tree approach.

        Args:
            encoded_data: The encoded data to be decoded.
            huffman_codes: A dictionary mapping Huffman codes to their corresponding characters.
            reverse: A boolean indicating whether to reverse the Huffman codes for decoding.

        Returns:
            The decoded data as a string.

        Raises:
            EncodingError: If a base is not found in the rotating map complement.
        """
        # Decode the encoded data
        trit_data: str = self.dna_to_trit(encoded_data)

        return super().decode(trit_data, huffman_codes, reverse=reverse)


    @override
    def estimate_bases_per_byte(self) -> float:
        """
        Estimates the number of DNA bases required to encode one byte of data using Goldman's rotating tree approach.

        Returns:
            The estimated number of DNA bases per byte as a float.
        """
        return self.ESTIMATED_BASES_PER_BYTE


    @override
    def print_info(self, 
                   encoded_data: str | list[str], 
                   huffman_codes: dict[str, str], 
                   original_size: int, 
                   k: int = 2) -> None:
        super().print_info(encoded_data, huffman_codes, original_size, k)