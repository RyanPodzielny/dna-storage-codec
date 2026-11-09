"""
Module enables framing and deframing of oligos with specific sequences and lengths, including primers, checksums, and 
error correction codes. The OligoFramer drives the encoding process, it takes in a file and outputs a list of oligos 
that can be synthesized. The OligoDeframer drives the decoding process, it takes in a list of (possibly erroneous) 
oligos, corrects them, and outputs the original file content. Both share the layout in OligoFormat.
"""


import hashlib
import math

from reedsolo import ReedSolomonError
from typing import Annotated, Any, Final, Generator
from pathlib import Path

# Using pydantic to take care of the config 
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt
from pydantic.dataclasses import dataclass

from Helpers.Errors import Error, ExpectedLengthError, IndexingError, PayloadLengthError, UnrankError
from Encode.Methods.Encoding import Base, Encoding, EncodingStats
from Encode.Indexer import Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, RSInner
from Encode.ErrorCorrection.OuterCodes import OuterCodes
from Encode.Methods.HuffmanCoding import Goldman, Huffman


#------------------------------------------------------------------------------#
#                                                                              #
#   TYPES                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#


# A primer, any number of bases (can be empty). Pattern is built from Base so the allowed letters come from Encoding,
#   e.g. ^[ACGT]*$ - pydantic rejects anything else. Also works for full DNA strands
Primer = Annotated[str, Field(pattern=f"^[{''.join(Base)}]*$")]



class FramerConfig(BaseModel):
    """
    Everything used to frame oligos for a single file. The in path is the file to read: the original file when framing,
    or the JSON file the framer saved when deframing (see Serializer.load_config).

    The components are mutable. The framer changes the indexer's width while running, so give each framer its own
    indexer rather than reusing one across files.

    All the checks are single field checks written on the types, so there is no model validator. A bad config raises
    pydantic's ValidationError listing every bad field.
    """

    # Same as the error configs - no extra fields and validate on assignment, so the layout the framer writes back is
    #   checked too. Arbitrary types lets the components (encoder, indexer, etc) be fields - they aren't pydantic so
    #   only checks they are an instance of the right class, and keeps the same object
    model_config = ConfigDict(
        extra="forbid", 
        validate_assignment=True, 
        arbitrary_types_allowed=True
    )

    # The path to the input file to be framed, or the JSON file the framer saved when deframing.
    in_path: str

    # The length of single oligo
    oligo_length: PositiveInt

    # The encoder, indexer, checksum, and optional inner and outer error correction codes used to frame the oligos.
    encoder: Encoding
    indexer: Indexer
    checksum: CheckSum
    # Only used to find the index width when framing, can't be negative
    overhead_margin: NonNegativeFloat = 0.1

    # Primers can be empty - so framer will just use Huffman's default
    start_primer: Primer = ""
    end_primer: Primer = ""

    # Don't always use correction, so can be none
    inner_correction: RSInner | None = None
    outer_correction: OuterCodes | None = None

    # Layout, set by the framer after framing and by the Serializer when loading. 0 until the framer sizes the oligos.
    num_oligos: NonNegativeInt = 0
    payload_length: NonNegativeInt = 0
    filler_length: NonNegativeInt = 0



class OligoFormat:
    """
    The layout and components shared by the OligoFramer and OligoDeframer. Holds what goes where in an oligo, and the
    helpers both sides need to rebuild the same indices and filler.
    """


    def __init__(self, config: FramerConfig):
        """
        Initializes the shared layout from the config. The framer derives the layout from the input file and writes it
        back to the config once framing is done, the deframer gets it from the saved config.

        Args:
            config: The config holding the oligo settings and components.
        """

        # All errors are resolved in the FramerConfig, no need to check here. Everything reads its settings and layout
        #   straight from the config (self.config.oligo_length, self.config.indexer, etc) instead of copying them, so
        #   there is only one copy - the framer sets the layout on the config, and every change to it is validated.
        self.config: FramerConfig = config


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def hash_content(content: str) -> str:
        """
        Hashes file content so a decode can be checked against the original without needing the original file.

        Args:
            content: The file content, as read by the framer.

        Returns:
            The SHA-256 hex digest of the content.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


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
        if self.config.filler_length == 0:
            return ""

        previous_base: str = payload_sequence[-1] if payload_sequence else Huffman.START_BASE
        # Not using specific indexer type, this works fine
        filler_sequence: str = Goldman().trit_to_dna("0" * self.config.filler_length, previous_base)

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
        for index in range(self.config.num_oligos):
            index_sequence: str = self.config.indexer.unrank(
                index=index, 
                before_sequence=self.config.start_primer[-1] if self.config.start_primer else Huffman.START_BASE,
                after_sequence=Huffman.START_BASE
            )
            indices.append(index_sequence)

        return indices


    def _add_primers(self, oligo: str) -> str:
        """
        Adds the configured primers to an oligo.

        Args:
            oligo: The oligo sequence to which primers should be added.

        Returns:
            The oligo sequence with primers added.
        """

        return self.config.start_primer + oligo + self.config.end_primer



class OligoFramer(OligoFormat):
    """
    A class for framing oligos with specific sequences and lengths.
    """


    def __init__(self, config: FramerConfig):
        """
        Initializes the OligoFramer with the given config, which holds the input path, oligo settings, and components.

        Changes indexer width to the minimum required to support the number of oligos needed to encode the input file.
        
        Raises:
            PayloadLengthError: If the payload length is less than or equal to 0 after accounting for primers, index, 
                checksum, and inner error correction.
            FileNotFoundError: If the input file does not exist.
            RuntimeError: If the index width does not converge within the maximum number of iterations.
        """
        super().__init__(config)
        self.in_path: str = config.in_path

        # Need to see how much width is left for the payload to determine the number of oligos and the index width
        self.width_left: int = 0
        self.width_left += self.config.oligo_length - len(self.config.start_primer) - len(self.config.end_primer) \
                           - self.config.checksum.get_dna_bases()
        # If we have inner codes, we need to ensure we count them
        if self.config.inner_correction:
            self.width_left -= self.config.inner_correction.get_dna_bases()
        
        # Determine the number of oligos and the index width based on the available width and overhead margin
        self.config.num_oligos, self.config.indexer.width = OligoFramer.determine_oligos_and_width(
            overhead_margin=config.overhead_margin, 
            width_left=self.width_left,
            in_path=self.in_path,
            encoder=self.config.encoder, 
            indexer=self.config.indexer,
            outer_error_correction=self.config.outer_correction
        )
        self.width_left -= self.config.indexer.width

        # Finally, the payload length, which can be found after the index width is determined. We set it based on the 
        #   width left after accounting for primers, index, checksum, inner/outer(filling)correction
        self._set_payload_length()

        # Filled in by frame_all_oligos, saved so the deframer can decode
        self.content_length: int = 0
        self.content_hash: str = ""
        self.huffman_codes: dict[str, str] = {}     # Note: standard DNA mapping just is the dict for the map
        self.framed_oligos: list[str] = []


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
        Determines the number of oligos needed to encode the input file, and the index width required to address them.

        Runs a single encoding pass to count payload bases, inflates that count by overhead_margin, then finds the 
        smallest index width that is consistent with the resulting number of oligos.

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
        oligo length the same.

        Raises:
            PayloadLengthError: If the payload length is less than or equal to 0 after accounting for primers, index,
                checksum, and inner error correction.
        """
        payload_length: int = self.width_left
        # Update the payload length based on the outer error correction scheme, if any
        if self.config.outer_correction:
            payload_length = self.config.outer_correction.align_payload_length(self.width_left)

        # If payloadlength is 0 here, means we don't have enough room
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0 after accounting for " \
                        "primers, index, checksum, and inner error correction",
                payload_length=payload_length
            )

        self.config.payload_length = payload_length
        self.config.filler_length = self.width_left - payload_length


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
        if self.config.outer_correction:
            payload_indices = self.config.outer_correction.select_payload_indices(indices)

        # Encode the file content into segments
        items: tuple[dict[str, str], list[str], Any] = self.config.encoder.encode_segments(
            data=file_content, 
            payload_length=self.config.payload_length, 
            previous_sequences=payload_indices
        )

        huffman_codes: dict[str, str] = items[0]
        encoded_segments: list[str] = items[1]

        # Update our number of oligos based on the actual number of encoded segments
        self.config.num_oligos = len(encoded_segments)

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
        with open(self.in_path, 'r', encoding='utf-8') as file:
            file_content: str = file.read()
        # Needed (not really needed but nice) as meta data for the deframer ensures proper reconstruction
        self.content_length = len(file_content)
        self.content_hash = OligoFormat.hash_content(file_content)

        indices: list[str]
        huffman_codes: dict[str, str]
        encoded_segments: list[str]
        # While we don't have enough indicies for number of segments, keep increasing the index width and trying again
        while True:
            try:
                indices, huffman_codes, encoded_segments = self._generate_segments(file_content)
                self.huffman_codes = huffman_codes

            except (IndexingError, UnrankError):
                # The estimate did not provide enough indices for the encoder's actual segment count. Unrank errors 
                #   come from generating indices for a count that needs more than one width step from the last pass
                self.width_left -= 1
                self.config.indexer.width += 1
                self._set_payload_length()
                
                continue

            # Recalculate the number of strands from the actual encoded payload (depending on if we have outer ECC)
            if self.config.outer_correction:
                self.config.num_oligos = self.config.outer_correction.calculate_total_strands(len(encoded_segments))

            # Ensure the indexer can address every payload and parity strand.
            if self.config.indexer.calculate_total_valid_indices() >= self.config.num_oligos:
                # Now that we know the index fits, generate the indices for the parity strands too
                if self.config.outer_correction:
                    indices = self._generate_indices()
                break

            # Increase the index width and recalculate the payload length
            self.width_left -= 1
            self.config.indexer.width += 1
            self._set_payload_length()

        return indices, huffman_codes, encoded_segments


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
        if self.config.payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=self.config.payload_length
            )

        # Filler pads the payload up to the width left when the outer code needs a shorter payload. It holds no data so
        #   it is not part of the checksum or inner RS, but the checksum DNA follows it
        filler_sequence: str = self._generate_filler(payload_sequence)

        # Calculate the checksum sequence
        checksum_sequence: str = self.config.checksum.encode_to_dna(
            before_sequence=payload_sequence + filler_sequence,
            after_sequence=self.config.end_primer,
            checksum=self.config.checksum.calculate_checksum(index_sequence + payload_sequence),
        )

        # Calculate the Reed-Solomon parity sequence if inner error correction is enabled
        rs_parity_sequence: str = ""
        if self.config.inner_correction:
            protected_sequence: str = index_sequence + payload_sequence + checksum_sequence
            rs_parity_sequence: str = self.config.inner_correction.encode_to_dna(
                before_sequence=checksum_sequence,
                after_sequence=self.config.end_primer,
                encoded_data=protected_sequence
            )

        # Add primers if requested
        framed_oligo: str = ""
        if include_primers:
            framed_oligo = self.config.start_primer +\
                           index_sequence +\
                           payload_sequence +\
                           filler_sequence +\
                           checksum_sequence +\
                           rs_parity_sequence +\
                           self.config.end_primer
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

        # If we have outer correction, we need to add the parity segments to the payload segments, so that they are all
        #   in order. The outer code decides the order (and if the payload strands are framed at all)
        if self.config.outer_correction:
            # Get our parity segments - these are the extra strands
            parity_segments: list[str] = self.config.outer_correction.add_redundancy(indices, encoded_segments)

            # Update the encoded segments to include the parity segments
            encoded_segments = self.config.outer_correction.arrange_strands(encoded_segments, parity_segments)

        # And frame them
        framed_oligos: list[str] = []
        for index, payload_sequence in enumerate(encoded_segments):
            framed_oligo: str = self.frame_oligo(indices[index], payload_sequence)
            framed_oligos.append(framed_oligo)

        # The layout was set straight on the config as it was worked out, so it is already ready to save
        self.framed_oligos = framed_oligos

        # Copy so callers adding errors to the oligos don't change what gets saved
        return framed_oligos.copy()


    def print_info(self, original_size: int, k: int = 3) -> None:
        """
        Prints the components and layout of the framed oligos, then measurements of the oligos. The runs and GC content
        are measured without the primers, since the primers are the same on every oligo and chosen separately.

        Args:
            original_size: The size of the original file in bits.
            k: The minimum length of a DNA run to be counted.

        Returns:
            None

        Raises:
            RuntimeError: If frame_all_oligos has not been called yet.
        """
        if not self.framed_oligos:
            raise RuntimeError("Call frame_all_oligos before printing info, the layout is not final until then.")

        # Short name, this reads a lot of the config
        config: FramerConfig = self.config

        # Components and layout
        print(f"\tEncoder: {config.encoder.describe()}")
        print(f"\tIndexer: {type(config.indexer).__name__}")
        print(f"\tInner correction: {type(config.inner_correction).__name__ if config.inner_correction else 'None'}")
        print(f"\tOuter correction: {type(config.outer_correction).__name__ if config.outer_correction else 'None'}")

        # Oligo layout and measurements
        inner_length: int = config.inner_correction.get_dna_bases() if config.inner_correction else 0
        # Same order as the parts are framed in
        print(f"\tOligos: {config.num_oligos} of length {config.oligo_length} "
              f"(start primer {len(config.start_primer)}, index {config.indexer.width}, "
              f"payload {config.payload_length}, filler {config.filler_length}, "
              f"checksum {config.checksum.get_dna_bases()}, inner RS {inner_length}, "
              f"end primer {len(config.end_primer)})")

        # With primers, only the size changes, since the primers are the same on every oligo
        total_bases: int = 0
        for oligo in self.framed_oligos:
            total_bases += len(oligo)
        print(f"\tWith primers: {total_bases} bases, {original_size / total_bases:.3f} bits/base")

        # Measure the oligos without their primers
        start: int = len(config.start_primer)
        end: int = len(config.end_primer)
        bodies: list[str] = []
        for oligo in self.framed_oligos:
            bodies.append(oligo[start:len(oligo) - end])

        print("\tWithout primers:")
        EncodingStats(config.encoder, bodies, self.huffman_codes, k).print_info(original_size, indent="\t\t")



class OligoDeframer(OligoFormat):
    """
    A class for deframing and decoding oligos framed with the same layout.
    """


    def __init__(self,
                 config: FramerConfig,
                 *,
                 huffman_codes: dict[str, str], oligos: list[str],
                 filename: str, content_length: int, content_hash: str):
        """
        Initializes the OligoDeframer with the config the oligos were framed with (including its layout) and what the
        framer produced. Usually made from a saved file with Serializer.load_deframer.

        Args:
            config: The config the oligos were framed with, with its layout filled in.
            huffman_codes: The codes the encoder used, needed to decode the payloads.
            oligos: The oligos as framed - either with or withour errors.
            filename: The name of the original file.
            content_length: The number of characters in the original file.
            content_hash: The hash of the original file's content, from OligoFormat.hash_content.
        """
        super().__init__(config)
        self.in_path: str = config.in_path

        self.huffman_codes: dict[str, str] = huffman_codes
        self.oligos: list[str] = oligos

        self.filename: str = filename
        self.content_length: int = content_length
        self.content_hash: str = content_hash


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Placeholder for missing payloads in the sorted list
    MISSING_PAYLOAD: Final[str] = "\n\nMISSING\n\n"  


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


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
            checksum: int = self.config.checksum.decode_from_dna(
                encoded_data=checksum_sequence,
                before_sequence=payload_sequence + self._generate_filler(payload_sequence),
                after_sequence=self.config.end_primer
            )
            return self.config.checksum.verify_checksum(
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
        if not self.config.inner_correction:
            return None

        # Correct the oligo using RS codes
        encoded_data: str = index_sequence + payload_sequence + checksum_sequence
        corrected_data: str
        corrected_parity: str
        corrected_data, corrected_parity = self.config.inner_correction.decode_from_dna(
            encoded_data=encoded_data,
            before_sequence=checksum_sequence,
            after_sequence=self.config.end_primer,
            parity_dna_sequence=rs_sequence,
        )

        # The inner RS does not cover the filler, so put it back after the payload before unframing
        filler_start: int = self.config.indexer.width + self.config.payload_length
        corrected_data = corrected_data[:filler_start] +\
                         self._generate_filler(corrected_data[:filler_start]) +\
                         corrected_data[filler_start:]

        # Unframe the corrected oligo to get the corrected index, payload, and checksum sequences
        corrected_index: str
        corrected_payload: str
        corrected_checksum: str
        corrected_index, corrected_payload, corrected_checksum, corrected_parity = self.unframe_oligo(
            framed_oligo=corrected_data + corrected_parity,
            expected_length=self.config.oligo_length,
            include_primers=False
        )

        # Verify the corrected checksum
        if not self._checksum_is_valid(corrected_index, corrected_payload, corrected_checksum):
            return None

        # Return the corrected data plus the corrected parity
        return corrected_index, corrected_payload, corrected_checksum, corrected_parity


    def _sort_payloads(self, indices: list[str], recovered_payloads: dict[str, str]) -> list[str | None]:
        """
        Puts the recovered payloads back in index order. Strands are normally shuffled (and some lost or duplicated) by
        the time we read them, so the order they come in means nothing. We already generate every index for the
        previous bases, so a reverse lookup of index sequence to position places each payload without ranking.

        Args:
            indices: Every index sequence in order, from _generate_indices.
            recovered_payloads: The payloads that passed the checksum (or were corrected), keyed by their index
                sequence. Their index is covered by the checksum, so it can be trusted.

        Returns:
            A list with one slot per index. Slots with no recovered payload are None (an erasure), and their index is
            still in indices for the previous base.
        """
        positions: dict[str, int] = {}
        # Build a reverse lookup of index sequence to position in the list of indices
        for position, index_sequence in enumerate(indices):
            positions[index_sequence] = position

        sorted_payloads: list[str | None] = [None] * len(indices)
        for index_sequence, payload_sequence in recovered_payloads.items():
            # An index we never generated can't be placed (only possible if a bad strand slipped past the checksum)
            if index_sequence in positions:
                sorted_payloads[positions[index_sequence]] = payload_sequence

        return sorted_payloads


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def unframe_oligo(self, 
                      framed_oligo: str, 
                      expected_length: int, 
                      include_primers: bool = True) -> tuple[str, str, str, str]:
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
        start_primer_length: int = len(self.config.start_primer)
        end_primer_length: int = len(self.config.end_primer)
        if not include_primers:
            expected_length -= start_primer_length + end_primer_length
            start_primer_length = 0
            end_primer_length = 0

        checksum_length: int = self.config.checksum.get_dna_bases()
        rs_parity_length: int = self.config.inner_correction.get_dna_bases() if self.config.inner_correction else 0

        # Ensure the framed oligo is of the expected length
        if len(framed_oligo) != expected_length:
            raise ExpectedLengthError(
                sequence=framed_oligo,
                expected_length=expected_length
            )

        # Extract the index sequence
        index_sequence: str = framed_oligo[start_primer_length:start_primer_length + self.config.indexer.width]

        # Extract the payload sequence
        payload_start: int = start_primer_length + self.config.indexer.width
        payload_end: int = payload_start + self.config.payload_length
        payload_sequence: str = framed_oligo[payload_start:payload_end]

        # Extract the checksum sequence, skipping over the filler
        checksum_start: int = payload_end + self.config.filler_length
        checksum_end: int = checksum_start + checksum_length
        checksum_sequence: str = framed_oligo[checksum_start:checksum_end]

        # Extract the RS parity sequence if inner error correction is enabled
        rs_sequence: str = ""
        if self.config.inner_correction:
            rs_parity_start: int = checksum_end
            rs_parity_end: int = rs_parity_start + rs_parity_length
            rs_sequence: str = framed_oligo[rs_parity_start:rs_parity_end]

        return index_sequence, payload_sequence, checksum_sequence, rs_sequence


    def unframe_all_oligos(self, framed_oligos: list[str]) -> Generator[tuple[str, str, str, str]]:
        """
        Unframes all oligos, extracting the index and payload sequences for each. Done as a generator as some oligos
        may not be the expected length and we want to yield empty strings for those instead of raising an error.

        Args:
            framed_oligos: A list of fully framed oligos as strings.
        
        Returns:
            A generator yielding tuples, each containing the index sequence, payload sequence, checksum sequence, and 
                RS parity sequence for each oligo.

        Raises:
            ExpectedLengthError: If the length of any framed oligo does not match the expected length.
        """
        expected_length: int = self.config.oligo_length

        # Go through each framed oligo and unframe it, yielding the results
        for framed_oligo in framed_oligos:
            try:
                yield self.unframe_oligo(framed_oligo, expected_length)
            # Yield empty strings for any oligo that does not match the expected length, as we can't unframe it
            except ExpectedLengthError as _:
                yield ("", "", "", "")


    def correct_all_oligos(self) -> list[str | None]:
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
        # The payloads we could trust, keyed by their index sequence. Anything we can't trust is left out, and ends up 
        #   as an erasure once sorted
        recovered_payloads: dict[str, str] = {}

        # Primers don't matter here, so remove them from expected length
        expected_length: int = self.config.oligo_length - len(self.config.start_primer) - len(self.config.end_primer)

        # For all our unframed oligos, check if the checksum is valid, if not, try to correct it with RS codes
        for index_sequence, payload_sequence, checksum_sequence, rs_sequence in self.unframe_all_oligos(self.oligos):
            # Oligo without the primers, the filler isn't returned when unframing so count it seperately
            oligo: str = index_sequence + payload_sequence + checksum_sequence + rs_sequence

            # Skip (erasure) if the oligo is not the expected length
            if len(oligo) + self.config.filler_length != expected_length:
                continue

            # If we couldn't validate the checksum, try to correct with inner RS if we have it
            if not self._checksum_is_valid(index_sequence, payload_sequence, checksum_sequence):
                # No inner correction means we can't correct it, so skip (erasure)
                if not self.config.inner_correction:
                    continue

                # Try to correct the oligo with inner RS codes. If it fails, skip (erasure)
                try:
                    items: tuple[str, str, str, str] | None = self._correct_oligo(
                        index_sequence=index_sequence,
                        payload_sequence=payload_sequence,
                        checksum_sequence=checksum_sequence,
                        rs_sequence=rs_sequence
                    )
                # Any custom error or a failed Reed-Solomon correction means we can't correct it
                except (Error, ReedSolomonError):
                    continue

                # Couldn't correct it, or the checksum still fails after correction
                if items is None:
                    continue

                # The oligo was corrected, so use the corrected values
                index_sequence, payload_sequence, checksum_sequence, rs_sequence = items
                if not self._checksum_is_valid(index_sequence, payload_sequence, checksum_sequence):
                    continue

            # Our payload is good. Keep the first copy if the same strand was read more than once
            if index_sequence not in recovered_payloads:
                recovered_payloads[index_sequence] = payload_sequence

        # Generate all indicies, these are the previous bases for every strand, including the ones we lost
        indices: list[str] = self._generate_indices()

        # Strands come in shuffled, so put the payloads back in index order with None for the ones we lost
        payload_segments: list[str | None] = self._sort_payloads(indices, recovered_payloads)

        # Correct the payload segments with outer error correction if we have it
        if self.config.outer_correction:
            payload_segments = self.config.outer_correction.perform_correction(indices, payload_segments)

        # Rotation based encoders (Goldman) need the base before each payload, same indices we used when encoding.
        #   Outer correction removed the parity strands, so drop their indices too
        payload_indices: list[str] = indices
        if self.config.outer_correction:
            payload_indices = self.config.outer_correction.select_payload_indices(indices)

        # Decode the payload segments back into the original file content where we can
        decoded_segments = self.config.encoder.decode_segments(
            encoded_data=payload_segments,
            huffman_codes=self.huffman_codes,
            previous_sequences=payload_indices
        )

        return decoded_segments


    def decode_to_string(self, decoded_oligos: list[str | None]) -> str:
        """
        Decodes the given corrected oligos back into the original file content.

        Args:
            decoded_oligos: A list of corrected oligos as strings with the decoded content

        Returns:
            The decoded original file content as a string.
        """
        # Join the decoded segments into a single string
        decoded_content: str = ""

        # Put them all together, using a placeholder for any missing segments.
        for segment in decoded_oligos:
            if segment is not None:
                decoded_content += segment
            else:
                decoded_content += OligoDeframer.MISSING_PAYLOAD 

        return decoded_content


    def calculate_recovered(self, decoded_oligos: list[str | None]) -> float:
        """
        Calculates the percentage of the original file content that can be recovered from the given framed oligos.

        Args:
            decoded_oligos: A list of corrected oligos as strings with the decoded content.

        Returns:
            The fraction of the original file content that can be recovered.
        """
        # Sum up all the Nones
        valid_oligos: int = sum(1 for data in decoded_oligos if data is not None)

        # Divide by the decoded oligos, not num_oligos. Outer correction drops the parity strands, so num_oligos would
        #   count strands that are never in the decoded list
        fraction_recovered: float = 0.0
        if decoded_oligos:
            fraction_recovered = (valid_oligos / len(decoded_oligos)) * 100

        return fraction_recovered


    def matches_original(self, content: str) -> bool:
        """
        Checks decoded content against the original file using the length and hash saved when framing.

        Args:
            content: The decoded file content.

        Returns:
            True if the content matches the original file, False otherwise.
        """
        matches: bool = len(content) == self.content_length and OligoFormat.hash_content(content) == self.content_hash
        return matches