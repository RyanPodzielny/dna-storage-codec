"""
Module enables framing and deframing of oligos with specific sequences and lengths, including primers, checksums, and 
error correction codes. Basically the driving class for the encoding process, it takes in a file and outputs a list of 
oligos that can be synthesized. And the for the decoding process, it takes in a list of oligos and outputs the original 
file content. 
"""


import math
from reedsolo import ReedSolomonError
from typing import Any, Generator
from pathlib import Path
from Encode.Helpers.Errors import Error, ExpectedLengthError, IndexingError, NotEnoughPreviousSegments, PayloadLengthError, RankError
from Encode.Methods.Encoding import Encoding
from Encode.Indexer import Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, RSInner
from Encode.ErrorCorrection.OuterCodes import OuterCodes
from Encode.Methods.HuffmanCoding import Goldman, Huffman


class OligoFramer:
    """
    A class for framing oligos with specific sequences and lengths.
    """


    def __init__(self, 
                 in_path: str,
                 oligo_length: int, overhead_margin: float,
                 start_primer: str, end_primer: str, 
                 encoder: Encoding, indexer: Indexer, 
                 checksum: CheckSum, 
                 inner_correction: RSInner | None = None, outer_correction: OuterCodes | None = None):
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
        self.inner_correction: RSInner | None = inner_correction
        self.outer_correction: OuterCodes | None = outer_correction

        # Need to see how much width is left for the payload to determine the number of oligos and the index width
        self.width_left: int = 0
        self.width_left += oligo_length - len(self.start_primer) - len(self.end_primer) - self.checksum.get_dna_bases()
        # If we have inner codes, we need to ensure we count them
        if self.inner_correction:
            self.width_left -= self.inner_correction.get_dna_bases()
        
        self.num_oligos: int = 0
        # Determine the number of oligos and the index width based on the available width and overhead margin
        self.num_oligos, self.indexer.width = OligoFramer.determine_oligos_and_width(
            overhead_margin=overhead_margin, width_left=self.width_left,
            in_path=self.in_path,
            encoder=self.encoder, indexer=self.indexer,
            outer_error_correction=self.outer_correction
        )
        self.width_left -= self.indexer.width

        # Finally, the most important value, the payload length, which can be found after the index width is determined
        self.payload_length: int = 0
        self.filler_length: int = 0

        # Set the payload length based on the width left after accounting for primers, index, checksum, 
        #   inner/outer(filling)correction
        self._set_payload_length()

        if self.payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0 after accounting for primers, index, checksum, and inner error correction", 
                payload_length=self.payload_length
            )

        """
        TODO: TEMP -remove, table codes will be given by json file on decoding
        
        """
        self.huffman_codes: dict[str, str] = {}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Maximum iterations for index width convergence
    MAX_ITERATIONS: int = 32


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def determine_oligos_and_width(overhead_margin: float, width_left: int,
                                   in_path: str,
                                   encoder: Encoding, indexer: Indexer,
                                   outer_error_correction: OuterCodes | None = None) -> tuple[int, int]:
        """
        Determines the number of oligos needed to encode the input file, and the
        index width required to address them.

        Runs a single encoding pass to count payload bases, inflates that count by
        `overhead_margin`, then finds the smallest index width that is consistent
        with the resulting number of oligos.

        Args:
            overhead_margin: Fractional safety margin applied to the base count (e.g. 0.10).
            width_left: The number of bases available for the index and payload.
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
        for _ in range(OligoFramer.MAX_ITERATIONS):
            payload_bases: int = width_left - index_width
            if outer_error_correction:
                payload_bases = outer_error_correction.align_payload_length(payload_bases)

            # Not enough payload room, raise an error
            if payload_bases <= 0:
                raise PayloadLengthError(
                    message="No payload room remains at the current index width", 
                    payload_length=payload_bases
                )

            # Calculate the number of oligos needed to encode the margined bases, and the required width to address them
            num_oligos: int = math.ceil(margined_bases / payload_bases)
            if outer_error_correction:
                num_oligos = outer_error_correction.calculate_total_strands(num_oligos)

            # Determine the required index width to address the number of oligos
            required_index_width: int = indexer.determine_optimal_width(
                required_capacity=num_oligos, 
                max_run_length=indexer.max_run_length
            )

            if required_index_width == index_width:
                return num_oligos, index_width

            index_width = required_index_width

        raise RuntimeError(f"Index width did not converge after {OligoFramer.MAX_ITERATIONS} iterations.")


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _set_payload_length(self) -> None:
        """
        Sets the payload length from the width left. The outer code may need a shorter payload (RSOuter needs full
        six-base columns), so any bases it can't use become filler between the payload and checksum. This keeps the
        oligo length the same as requested.
        """
        self.payload_length = self.width_left
        if self.outer_correction:
            self.payload_length = self.outer_correction.align_payload_length(self.width_left)

        self.filler_length = self.width_left - self.payload_length


    def _generate_filler(self, payload_sequence: str) -> str:
        """
        Generates the filler bases that go after the payload. The filler holds no data, it is a fixed run of Goldman
        zero trits so it never repeats the previous base (keeps the homopolymer constraint) and can be regenerated
        from the payload alone during decoding.

        Args:
            payload_sequence: The payload sequence the filler follows.

        Returns:
            The filler bases, empty if there is no filler.
        """
        if self.filler_length == 0:
            return ""

        previous_base: str = payload_sequence[-1] if payload_sequence else Huffman.START_BASE
        # Not using specific indexer type, this works fine
        filler_sequence: str = Goldman().trit_to_dna("0" * self.filler_length, previous_base)

        return filler_sequence


    def _generate_indices(self) -> list[str]:
        """
        Generates a list of index sequences for the number of oligos needed to encode the input file.

        Returns:
            A list of index sequences for the number of oligos needed to encode the input file.

        Raises:
            PayloadLengthError: If the payload length is less than or equal to 0.
            UnrankError: If the index sequence is not in bounds of capacity.
        """

        # Generate all indices needed by payload and outer-code strands.
        indices: list[str] = []
        for index in range(self.num_oligos):
            index_sequence: str = self.indexer.unrank(
                index=index, 
                before_sequence=self.start_primer[-1], 
                after_sequence=Huffman.START_BASE
            )
            indices.append(index_sequence)

        return indices

    
    def _generate_segments(self, file_content: str) -> tuple[list[str], dict[str, str], list[str]]:
        """
        Generates a list of index sequences for the number of oligos needed to encode the input file.

        Args:
            file_content: The content of the input file as a string.
        
        Returns:
            A tuple containing:
                - A list of index sequences for the number of oligos needed to encode the input file.
                - A dictionary mapping each unique symbol in the input file to its corresponding Huffman code.
                - A list of encoded segments corresponding to the input file content.

        Raises:
            FileNotFoundError: If the input file does not exist.
            PayloadLengthError: If the payload length is less than or equal to 0.
            UnrankError: If the index sequence is not in bounds of capacity.
        """
        indices: list[str] = self._generate_indices()

        # Select the payload indices based on the outer error correction scheme, if any
        payload_indices: list[str] = indices
        if self.outer_correction:
            payload_indices = self.outer_correction.select_payload_indices(indices)

        # Encode the file content into segments
        items: tuple[dict[str, str], list[str], Any] = self.encoder.encode_segments(
            data=file_content, 
            payload_length=self.payload_length, 
            previous_sequences=payload_indices
        )

        huffman_codes: dict[str, str] = items[0]
        encoded_segments: list[str] = items[1]

        # Update our number of oligos based on the actual number of encoded segments
        self.num_oligos = len(encoded_segments)

        return payload_indices, huffman_codes, encoded_segments


    def _extend_indices(self) -> tuple[list[str], dict[str, str], list[str]]:
        """
        Extends the index width until there are enough indices to address all encoded segments.

        Args:
            in_path: The path to the input file.

        Returns:
            A tuple containing:
                - A list of index sequences for the number of oligos needed to encode the input file.
                - A dictionary mapping each unique symbol in the input file to its corresponding Huffman code.
                - A list of encoded segments corresponding to the input file content.

        Raises:
            FileNotFoundError: If the input file does not exist.
            IndexerError: If the index sequence is not in bounds of capacity (shoulnd't happen since we adjust before 
                but it can be raised while framing if index goes out of bounds).
            PayloadLengthError: If the payload length is less than or equal to 0.
            UnrankError: If the index sequence is not in bounds of capacity.
            UnicodeDecodeError: If the input file cannot be read as UTF-8.
        """

        # Get our file content
        with open(self.in_path, "r", encoding="utf-8") as file:
            file_content: str = file.read()

        indices: list[str] 
        huffman_codes: dict[str, str]
        encoded_segments: list[str]
        # While we don't have enough indicies for number of segments, keep increasing the index width and trying again
        while True:
            try:
                indices, huffman_codes, encoded_segments = self._generate_segments(file_content)
                self.huffman_codes = huffman_codes
            except IndexingError:
                # The estimate did not provide enough indices for the encoder's actual segment count.
                self.width_left -= 1
                self.indexer.width += 1
                self._set_payload_length()
                continue

            # Recalculate the number of strands from the actual encoded payload (depending on if we have outer ECC)
            if self.outer_correction:
                self.num_oligos = self.outer_correction.calculate_total_strands(len(encoded_segments))

                # Ensure we have enough indices now that we know how much more we need for the parity strands
                indices = self._generate_indices()
            
            # Ensure the indexer can address every payload and parity strand.
            if self.indexer.calculate_total_valid_indices() >= self.num_oligos:
                break

            # Increase the index width and recalculate the payload length
            self.width_left -= 1
            self.indexer.width += 1
            self._set_payload_length()

        return indices, huffman_codes, encoded_segments


    def _checksum_is_valid(self, index_sequence: str, payload_sequence: str, checksum_sequence: str) -> bool:
        """
        Checks whether an oligo's checksum matches its index and payload.

        Args:
            index_sequence: The index sequence for the oligo.
            payload_sequence: The payload sequence for the oligo.
            checksum_sequence: The checksum sequence for the oligo.

        Returns:
            True if the checksum is valid, False otherwise.
        """
        try:
            # Decode the checksum from the DNA sequence and verify it against the index and payload
            checksum: int = self.checksum.decode_from_dna(
                encoded_data=checksum_sequence,
                before_sequence=payload_sequence + self._generate_filler(payload_sequence),
                after_sequence=self.end_primer
            )
            return self.checksum.verify_checksum(
                encoded_data=index_sequence + payload_sequence,
                checksum=checksum
            )
        # Any custom error means we fail the checksum 
        except Error:
            return False


    def _correct_oligo(self, 
                       index_sequence: str,
                       payload_sequence: str,
                       checksum_sequence: str, 
                       rs_sequence: str) -> tuple[str, str, str, str] | None:
        """
        Corrects an oligo with RS and returns its data plus corrected parity.
        
        Args:
            index_sequence: The index sequence for the oligo.
            payload_sequence: The payload sequence for the oligo.
            checksum_sequence: The checksum sequence for the oligo.
            rs_sequence: The RS parity sequence for the oligo.

        Returns:
            A tuple containing the corrected:
                - index sequence
                - payload sequence
                - checksum sequence
                - RS parity sequence
            If correction was successful, or None if correction failed.

        Raises:
            RankError: If the index sequence is not in bounds of capacity.
            ReedSolomonError: If the Reed-Solomon correction fails.
            KeyError: If the indexer fails to unrank the corrected parity bytes back to DNA sequences.
            ExpectedLengthError: If the length of the corrected oligo does not match the expected length.
        """
        if not self.inner_correction:
            return None

        # Correct the oligo using RS codes
        encoded_data: str = index_sequence + payload_sequence + checksum_sequence
        corrected_data: str
        corrected_parity: str
        corrected_data, corrected_parity = self.inner_correction.decode_from_dna(
            encoded_data=encoded_data,
            before_sequence=checksum_sequence,
            after_sequence=self.end_primer,
            parity_dna_sequence=rs_sequence,
        )

        # The inner RS does not cover the filler, so put it back after the payload before unframing
        filler_start: int = self.indexer.width + self.payload_length
        corrected_data = corrected_data[:filler_start] +\
                         self._generate_filler(corrected_data[:filler_start]) +\
                         corrected_data[filler_start:]

        # Unframe the corrected oligo to get the corrected index, payload, and checksum sequences
        corrected_index: str
        corrected_payload: str
        corrected_checksum: str
        corrected_index, corrected_payload, corrected_checksum, corrected_parity = self.unframe_oligo(
            framed_oligo=corrected_data + corrected_parity,
            expected_length=self.oligo_length,
            include_primers=False
        )

        # Verify the corrected checksum
        if not self._checksum_is_valid(corrected_index, corrected_payload, corrected_checksum):
            return None

        # Return the corrected data plus the corrected parity
        return corrected_index, corrected_payload, corrected_checksum, corrected_parity


    def _add_primers(self, oligo: str) -> str:
        """
        Adds the configured primers to an oligo.

        Args:
            oligo: The oligo sequence to which primers should be added.

        Returns:
            The oligo sequence with primers added.
        """

        return self.start_primer + oligo + self.end_primer


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


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

        # Filler pads the payload up to the width left when the outer code needs a shorter payload. It holds no data so
        #   it is not part of the checksum or inner RS, but the checksum DNA follows it
        filler_sequence: str = self._generate_filler(payload_sequence)

        # Calculate the checksum sequence
        checksum_sequence: str = self.checksum.encode_to_dna(
            before_sequence=payload_sequence + filler_sequence,
            after_sequence=self.end_primer,
            checksum=self.checksum.calculate_checksum(index_sequence + payload_sequence),
        )

        # Calculate the Reed-Solomon parity sequence if inner error correction is enabled
        rs_parity_sequence: str = ""
        if self.inner_correction:
            protected_sequence: str = index_sequence + payload_sequence + checksum_sequence
            rs_parity_sequence: str = self.inner_correction.encode_to_dna(
                before_sequence=checksum_sequence,
                after_sequence=self.end_primer,
                encoded_data=protected_sequence
            )

        # Add primers if requested
        framed_oligo: str = ""
        if include_primers:
            framed_oligo = self.start_primer +\
                           index_sequence +\
                           payload_sequence +\
                           filler_sequence +\
                           checksum_sequence +\
                           rs_parity_sequence +\
                           self.end_primer
        else:
            framed_oligo = index_sequence +\
                           payload_sequence +\
                           filler_sequence +\
                           checksum_sequence +\
                           rs_parity_sequence
        
        return framed_oligo


    def frame_all_oligos(self):
        """
        Frames all oligos for the input file, returning a list of fully framed oligos.

        Returns:
            A list of fully framed oligos as strings.

        Raises:
            FileNotFoundError: If the input file does not exist.
            IndexerError: If the index sequence is not in bounds of capacity (shoulnd't happen since we adjust before hand
                but it can be raised while framing if index goes out of bounds).
            UnicodeDecodeError: If the input file cannot be read as UTF-8.
            UnrankError: If the index sequence is not in bounds of capacity.
        """

        indices: list[str]
        huffman_codes: dict[str, str]
        encoded_segments: list[str]
        # Extend the amount of indicides/width of the index until we are sure we will have enough for num of segments
        indices, huffman_codes, encoded_segments = self._extend_indices()

        # If we have outer correction, we need to add the parity segments to the end of the payload segments, so that they are all in order
        if self.outer_correction:
            # Get our parity segments - these are the extra strands
            parity_segments = self.outer_correction.add_redundancy(indices, encoded_segments)

            # Split on number of parity segments
            blocked_parity_segments: list[list[str]] = []

            for index in range(0, len(parity_segments), self.outer_correction.parity_per_block):
                blocked_parity_segments.append(parity_segments[index:index + self.outer_correction.parity_per_block])

            # Now travel through each block of payload segments and add the parity segments to the end of each block,
            #   so that they all stay in order
            encoded_segments_with_parity: list[str] = []
            parity_block_index: int = 0
            for index in range(0, len(encoded_segments), self.outer_correction.payload_per_block):
                payload_block: list[str] = encoded_segments[index:index + self.outer_correction.payload_per_block]
                parity_block: list[str] = blocked_parity_segments[parity_block_index]

                # Add the parity segments to the end of the payload segments for this block
                encoded_segments_with_parity.extend(payload_block + parity_block)
                parity_block_index += 1

            # Update the encoded segments to include the parity segments
            encoded_segments = encoded_segments_with_parity

        # And frame them
        framed_oligos: list[str] = []
        for index, payload_sequence in enumerate(encoded_segments):
            framed_oligo: str = self.frame_oligo(indices[index], payload_sequence) 
            framed_oligos.append(framed_oligo)

        return framed_oligos


    def unframe_oligo(self, 
                      framed_oligo: str, 
                      expected_length: int, include_primers: bool = True) -> tuple[str, str, str, str]:
        """
        Unframes a single oligo, extracting the index and payload sequences.

        Args:
            framed_oligo: The fully framed oligo as a string.
            expected_length: The expected length of the framed oligo.
        
        Returns:
            A tuple containing the index sequence, payload sequence, checksum sequence, and RS parity sequence.

        Raises:
            ExpectedLengthError: If the length of the framed oligo does not match the expected length
        """
        # Calculate the lengths of the various components
        start_primer_length: int = len(self.start_primer)
        end_primer_length: int = len(self.end_primer)
        if not include_primers:
            expected_length -= len(self.start_primer) + len(self.end_primer)
            start_primer_length = 0
            end_primer_length = 0
        checksum_length: int = self.checksum.get_dna_bases()
        rs_parity_length: int = self.inner_correction.get_dna_bases() if self.inner_correction else 0

        # Ensure the framed oligo is of the expected length
        if len(framed_oligo) != expected_length:
            raise ExpectedLengthError(
                sequence=framed_oligo,
                expected_length=expected_length
            )

        # Extract the index sequence
        index_sequence: str = framed_oligo[start_primer_length:start_primer_length + self.indexer.width]

        # Extract the payload sequence
        payload_start: int = start_primer_length + self.indexer.width
        payload_end: int = payload_start + self.payload_length
        payload_sequence: str = framed_oligo[payload_start:payload_end]

        # Extract the checksum sequence, skipping over the filler
        checksum_start: int = payload_end + self.filler_length
        checksum_end: int = checksum_start + checksum_length
        checksum_sequence: str = framed_oligo[checksum_start:checksum_end]

        # Extract the RS parity sequence if inner error correction is enabled
        rs_sequence: str = ""
        if self.inner_correction:
            rs_parity_start: int = checksum_end
            rs_parity_end: int = rs_parity_start + rs_parity_length
            rs_sequence: str = framed_oligo[rs_parity_start:rs_parity_end]

        return index_sequence, payload_sequence, checksum_sequence, rs_sequence


    def unframe_all_oligos(self, framed_oligos: list[str]) -> Generator[tuple[str, str, str, str]]:
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

        # Go through each framed oligo and unframe it, yielding the results
        for framed_oligo in framed_oligos:
            try:
                yield self.unframe_oligo(framed_oligo, expected_length)
            # Yield empty strings for any oligo that does not match the expected length, as we can't unframe it
            except ExpectedLengthError as _:
                yield ("", "", "", "")


    def correct_all_oligos(self, framed_oligos: list[str]) -> list[str | None]:
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
        payload_segments: list[str | None] = []
        expected_length: int = self.oligo_length - len(self.start_primer) - len(self.end_primer)

        # For all our unframed oligos, check if the checksum is valid, if not, try to correct it with RS codes
        for index_sequence, payload_sequence, checksum_sequence, rs_sequence in self.unframe_all_oligos(framed_oligos):
            # Oligo without the primers, the filler isn't returned when unframing so count it seperately
            oligo: str = index_sequence + payload_sequence + checksum_sequence + rs_sequence

            # Mark for erasure if the oligo is not the expected length
            if len(oligo) + self.filler_length != expected_length:
                payload_segments.append(None)
                continue

            # Check if we can validate the checksum, if so our payload is good
            if self._checksum_is_valid(index_sequence, payload_sequence, checksum_sequence):
                payload_segments.append(payload_sequence)
                continue

            # If we are here, we couldn't validate checksum, so try to correct with inner RS if we have
            if self.inner_correction:
                try:
                    items: tuple[str, str, str, str] | None = self._correct_oligo(
                        index_sequence=index_sequence,
                        payload_sequence=payload_sequence, 
                        checksum_sequence=checksum_sequence, 
                        rs_sequence=rs_sequence
                    )
                    # Couldn't correct it, mark as erasure
                    if items is None:
                        payload_segments.append(None)
                        continue

                    index_sequence, payload_sequence, checksum_sequence, rs_sequence = items
                    payload_segments.append(payload_sequence)
                    continue

                # Any custom error means we fail the checksum and can't correct it
                except Error:
                    payload_segments.append(None)
                    continue
                # Reed-Solomon correction failed, mark as erasure
                except ReedSolomonError:
                    payload_segments.append(None)
                    continue

                continue

            # If we can't correct it (no inner correction available), mark as erasure
            payload_segments.append(None)

        """
        TODO: This wont be in order. Will have to sort by known indices and mark the unknowns to know which block it 
        goes in. Then we can correct the payload segments with outer error correction if we have it.

        REmove this garbage later
        """

        # Generate all indicies
        indices: list[str] = self._generate_indices()

        print(len(payload_segments), len(indices))

        # Correct the payload segments with outer error correction if we have it
        if self.outer_correction:
            payload_segments = self.outer_correction.perform_correction(indices, payload_segments)

            # Print amount of nones
            num_nones: int = sum(1 for segment in payload_segments if segment is None)
            print(f"After outer correction, {num_nones} segments are still None.")

        decoded_segments = self.encoder.decode_segments(encoded_data=payload_segments, huffman_codes=self.huffman_codes)

        thing = "".join(decoded_segments) # type: ignore
        print(thing)

        with open(self.in_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            if thing != original_content:
                print("Decoded content does not match original content.")
            else:
                print("Decoded content matches original content.")

        return decoded_segments