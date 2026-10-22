import math
from functools import lru_cache
from reedsolo import ReedSolomonError
from typing import Any
from pathlib import Path
from Encode.Helpers.Errors import Error, ExpectedLengthError, PayloadLengthError, RankError
from Encode.Methods.Encoding import Encoding
from Encode.Indexer import Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, ReedSolomon
from Encode.Methods.HuffmanCoding import Huffman

class OligoFramer:
    """
    A class for framing oligos with specific sequences and lengths.
    """
    # Maximum iterations for index width convergence
    MAX_ITERATIONS: int = 32  

    def __init__(self, 
                 in_path: str,
                 oligo_length: int, overhead_margin: float,
                 start_primer: str, end_primer: str, 
                 encoder: Encoding, indexer: Indexer, 
                 checksum: CheckSum, inner_error_correction: ReedSolomon | None = None):

        """
        Initializes the OligoFramer with the given parameters.
        
        Changes indexer width to the minimum required to support the number of oligos needed to encode the input file.
        
        Raises:
            PayloadLengthError: If the payload length is less than or equal to 0 after accounting for primers, index, 
                checksum, and inner error correction.
            FileNotFoundError: If the input file does not exist.
            RuntimeError: If the index width does not converge within the maximum number of iterations.
        """

        self.in_path: str = in_path
        self.oligo_length: int = oligo_length
        self.start_primer: str = start_primer if len(start_primer) > 0 else Huffman.START_BASE
        self.end_primer: str = end_primer if len(end_primer) > 0  else Huffman.START_BASE
        self.encoder: Encoding = encoder
        self.indexer: Indexer = indexer
        self.checksum: CheckSum = checksum
        self.inner_error_correction: ReedSolomon | None = inner_error_correction

        print(f"Start primer: {self.start_primer}, End primer: {self.end_primer}")

        # Need to see how much width is left for the payload to determine the number of oligos and the index width
        width_left: int = 0
        width_left += oligo_length - len(self.start_primer) - len(self.end_primer) - self.checksum.get_dna_bases()
        # If we have inner codes, we need to ensure we count them
        if self.inner_error_correction:
            width_left -= self.inner_error_correction.get_dna_bases()
        
        self.num_oligos: int = 0
        # Determine the number of oligos and the index width based on the available width and overhead margin
        self.num_oligos, self.indexer.width = OligoFramer.determine_oligos_and_width(
            overhead_margin, width_left, self.MAX_ITERATIONS,
            self.in_path,
            self.encoder, self.indexer
        )
        width_left -= self.indexer.width

        # Finally, the most important value, the payload length, which can be found after the index width is determined
        self.payload_length: int = width_left

        if self.payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0 after accounting for primers, index, checksum, and inner error correction", 
                payload_length=self.payload_length
            )

        self.huffman_codes: dict[str, str] | None = None


    @staticmethod
    def determine_oligos_and_width(overhead_margin: float, width_left: int, max_iterations: int,
                                   in_path: str,
                                   encoder: Encoding, indexer: Indexer) -> tuple[int, int]:
        """
        Determines the number of oligos needed to encode the input file, and the
        index width required to address them.

        Runs a single encoding pass to count payload bases, inflates that count by
        `overhead_margin`, then finds the smallest index width that is consistent
        with the resulting number of oligos.

        Args:
            overhead_margin: Fractional safety margin applied to the base count (e.g. 0.10).
            width_left: The number of bases available for the index and payload.
            max_iterations: The maximum number of iterations to attempt to find a suitable index width.
            in_path: The path to the input file.
            encoder: An instance of the Encoding class to use for encoding the file.
            indexer: An instance of the Indexer class to use for determining the index width.

        Returns:
            A tuple of (number of oligos, index width in bases).

        Raises:
            FileNotFoundError: If the input file does not exist.
            PayloadLengthError: If bases left does not allow for an index width
            RuntimeError: If the index width does not converge.
        """
        # Open up and do a single encode base to get the number of bases 
        bytes: int = Path(in_path).stat().st_size

        # Get the estimated bytes per base 
        estimated_bases: float = bytes * encoder.estimate_bases_per_byte()
        # Add a bit of overhead to the base count to be conservative
        margined_bases: int = math.ceil(estimated_bases * (1 + overhead_margin))

        # For each iteration, calculate the number of oligos and the required index width
        index_width: int = 0
        for _ in range(max_iterations):
            payload_bases: int = width_left - index_width

            if payload_bases <= 0:
                raise PayloadLengthError(
                    message="No payload room remains at the current index width", 
                    payload_length=payload_bases
                )

            num_oligos: int = math.ceil(margined_bases / payload_bases)
            required_index_width: int = indexer.determine_optimal_width(num_oligos, max_run_length=indexer.max_run_length)

            if required_index_width == index_width:
                return num_oligos, index_width

            index_width = required_index_width

        raise RuntimeError(f"Index width did not converge after {max_iterations} iterations.")


    def frame_oligo(self, index_sequence: str, payload_sequence: str, include_primers: bool = True) -> str:
        """
        Frames a single oligo with the given index and payload sequences, adding primers, checksum, and optional
        Reed-Solomon parity.

        Args:
            index_sequence: The index sequence for the oligo.
            payload_sequence: The payload sequence for the oligo.

        Returns:
            The fully framed oligo as a string.

        Raises:
            ValueError: If the checksum data is empty or None.
            PayloadLengthError: If the payload length is less than or equal to 0.
            UnrankError: If the index sequence is not in bounds of capacity.
        """
        if self.payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=self.payload_length
            )

        # Calculate the checksum sequence
        checksum_sequence: str = self.checksum.encode_to_dna(
            self.checksum.calculate_checksum(index_sequence + payload_sequence),
            before_sequence=payload_sequence,
            after_sequence=self.end_primer
        )

        # Calculate the Reed-Solomon parity sequence if inner error correction is enabled
        rs_parity_sequence: str = ""
        if self.inner_error_correction:
            protected_sequence: str = index_sequence + payload_sequence + checksum_sequence
            rs_parity_sequence: str = self.inner_error_correction.encode_to_dna(
                protected_sequence,
                before_sequence=checksum_sequence,
                after_sequence=self.end_primer
            )

        # Add primers if requested
        framed_oligo: str = ""
        if include_primers:
            framed_oligo = self.start_primer +\
                           index_sequence + payload_sequence +\
                           checksum_sequence + rs_parity_sequence +\
                           self.end_primer
        else:
            framed_oligo = index_sequence + payload_sequence +\
                           checksum_sequence + rs_parity_sequence
        
        return framed_oligo


    def frame_all_oligos(self):
        """
        Frames all oligos for the input file, returning a list of fully framed oligos.

        Returns:
            A list of fully framed oligos as strings.

        Raises:
            FileNotFoundError: If the input file does not exist.
            UnicodeDecodeError: If the input file cannot be read as UTF-8.
            UnrankError: If the index sequence is not in bounds of capacity.
        """
        # Get our file content
        with open(self.in_path, "r", encoding="utf-8") as file:
            file_content: str = file.read()

        # Get all indices and previous bases for the indexer
        indices: list[str] = []
        for index in range(self.num_oligos):
            index_sequence: str = self.indexer.unrank(
                index, 
                before=self.start_primer[-1], 
                after=Huffman.START_BASE
            )
            indices.append(index_sequence)

        # Encode the file content into segments
        items: tuple[dict[str, str], list[str], Any] = self.encoder.encode_segments(
            file_content, 
            self.payload_length, 
            previous_sequences=indices
        )
        self.huffman_codes = items[0]
        encoded_segments: list[str] = items[1]

        framed_oligos: list[str] = []
        for index, payload_sequence in enumerate(encoded_segments):
            framed_oligo: str = self.frame_oligo(indices[index], payload_sequence) 
            framed_oligos.append(framed_oligo)

        return framed_oligos


    def unframe_oligo(self, framed_oligo: str, expected_length: int) -> tuple[str, str, str, str]:
        """
        Unframes a single oligo, extracting the index and payload sequences.

        Args:
            framed_oligo: The fully framed oligo as a string.
        
        Returns:
            A tuple containing the index sequence, payload sequence, checksum sequence, and RS parity sequence.

        Raises:
            ExpectedLengthError: If the length of the framed oligo does not match the expected length
        """
        if len(framed_oligo) != expected_length:
            raise ExpectedLengthError(
                sequence=framed_oligo,
                expected_length=expected_length
            )

        # Calculate the lengths of the various components
        start_primer_length: int = len(self.start_primer)
        end_primer_length: int = len(self.end_primer)
        checksum_length: int = self.checksum.get_dna_bases()
        rs_parity_length: int = self.inner_error_correction.get_dna_bases() if self.inner_error_correction else 0

        # Extract the index sequence
        index_sequence: str = framed_oligo[start_primer_length:start_primer_length + self.indexer.width]

        # Extract the payload sequence
        payload_start: int = start_primer_length + self.indexer.width
        payload_end: int = len(framed_oligo) - (checksum_length + rs_parity_length + end_primer_length)
        payload_sequence: str = framed_oligo[payload_start:payload_end]

        # Extract the checksum sequence
        checksum_start: int = payload_end
        checksum_end: int = checksum_start + checksum_length
        checksum_sequence: str = framed_oligo[checksum_start:checksum_end]

        # Extract the RS parity sequence if inner error correction is enabled
        rs_parity_sequence: str = ""
        if self.inner_error_correction:
            rs_parity_start: int = checksum_end
            rs_parity_end: int = rs_parity_start + rs_parity_length
            rs_parity_sequence: str = framed_oligo[rs_parity_start:rs_parity_end]

        return index_sequence, payload_sequence, checksum_sequence, rs_parity_sequence


    def unframe_all_oligos(self, framed_oligos: list[str]) -> list[tuple[str, str, str, str]]:
        """
        Unframes all oligos, extracting the index and payload sequences for each.

        Args:
            framed_oligos: A list of fully framed oligos as strings.
        
        Returns:
            A list of tuples, each containing the index sequence, payload sequence, checksum sequence, and RS parity 
                sequence for each oligo.

        Raises:
            ExpectedLengthError: If the length of any framed oligo does not match the expected length.
        """
        expected_length: int = self.oligo_length
        unframed_oligos: list[tuple[str, str, str, str]] = []
        for framed_oligo in framed_oligos:
            # TODO: UnpectedLengthError will occur on indel, we will eventually want to try to rebuild it with the inner
            #   code, but will mark for erasure if can't
            unframed_oligo: tuple[str, str, str, str] = self.unframe_oligo(framed_oligo, expected_length)
            unframed_oligos.append(unframed_oligo)

        return unframed_oligos

    """
    Decodes all framed oligos back into the original file content.
    """
    def decode_all_oligos(self, framed_oligos: list[str]) -> list[str | None]:
        """
        Decodes all framed oligos back into the original file content.

        Args:
            framed_oligos: A list of fully framed oligos as strings.
        
        Returns:
            The decoded original file content as a string.

        Raises:
            ExpectedLengthError: If the length of any framed oligo does not match the expected length.
            UnrankError: If the index sequence is not in bounds of capacity.
        """
        # Retrieve all our parts in lock step
        items: list[tuple[str, str, str, str]] = self.unframe_all_oligos(framed_oligos)
        index_sequences: list[str] = []
        payload_sequences: list[str] = []
        checksum_sequences: list[str] = []
        rs_parity_sequences: list[str] = []
        for index_sequence, payload_sequence, checksum_sequence, rs_parity_sequence in items:
            index_sequences.append(index_sequence)
            payload_sequences.append(payload_sequence)
            checksum_sequences.append(checksum_sequence)
            rs_parity_sequences.append(rs_parity_sequence)

        return []