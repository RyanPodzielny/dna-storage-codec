from abc import abstractmethod
from collections import Counter
import math
from typing import ClassVar, Final, List, override
from numpy.typing import NDArray

# Can't use reedsolo because it doesn't support GF(3^6) and has a max of 255 symbols. We need to encode each column of
#   the oligos as 6-bit symbols, which gives us 3^6 = 729 symbols. This way we can retain the same fixed-length oligo
#   and use Goldman's rotation scheme to avoid homopolymers. It is much slower than reedsolo, but it the best option.
import galois
import numpy as np

from Encode.Helpers.Errors import EncodingError, MismatchSegments, NotEnoughPreviousSegments, OuterCodeError, \
    PayloadLengthError
from Encode.Indexer import GoldmanIndexer
from Encode.Methods.HuffmanCoding import Goldman


class OuterCodes():
    def __init__(self):
        self.payload_per_block: int = 0
        self.parity_per_block: int = 0
        pass


    @abstractmethod
    def calculate_total_strands(self, payload_strands: int) -> int:
        raise NotImplementedError()


    @abstractmethod
    def select_payload_indices(self, indices: List[str]) -> List[str]:
        raise NotImplementedError()


    @abstractmethod
    def add_redundancy(self, previous_sequences: List[str], payload_sequences: List[str]) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def perform_correction(self, previous_sequences: List[str], encoded_sequences: List[str | None]) -> List[str | None]:
        raise NotImplementedError()

    @abstractmethod
    def get_addtional_strands(self) -> int:
        raise NotImplementedError()

    def fit_payload_length(self, payload_length: int) -> int:
        """
        Returns the largest payload length the outer code can protect that is at most payload_length. By default, any 
        payload length works.
        """
        return payload_length


class GoldmanRedundancy(OuterCodes):
    """
    Creates four quarter-shifted copies of a payload stream.
    """

    LAYERS: int = 4

    def add_redundancy(self, segments: List[str], payload_length: int) -> None:
        pass

    
class XOR(OuterCodes):

    def calculate_total_strands(self, payload_strands: int) -> int:
        # Each 2 payload strands will create 1 parity strand, so we need to round up to the nearest even number of payload strands.
        if payload_strands <= 0:
            raise ValueError("Payload strand count must be positive.")
        
        return payload_strands + (payload_strands + 1) // 2


    def select_payload_indices(self, indices: List[str]) -> List[str]:
        if not indices:
            raise ValueError("Indices cannot be empty.")

        payload_indices: List[str] = []
        for block_start in range(0, len(indices), 3):
            block: List[str] = indices[block_start:block_start + 3]

            # Add the first 2 indices from the block to the payload_indices list
            payload_indices += block[:2]
        
        return payload_indices


class RSOuter(OuterCodes):
    """
    Reed-Solomon outer code for Goldman-encoded DNA.

    Goldman encoding represents each DNA base as a transition from the
    previous base, so each six-base column is represented by six trits.
    """

    COLUMN_LENGTH: Final[int] = 6

    GF_ORDER: int = 3**6
    GF: type[galois.FieldArray] = galois.GF(GF_ORDER)
    MAX_BLOCK_STRANDS: int = GF_ORDER - 1

    # Cache for RS codes when we have the same parity_strands value. Takes a while to initialize.
    _RS_CACHE: ClassVar[dict[tuple[int, int], galois.ReedSolomon]] = {}

    def __init__(self, redundancy_percentage: float = 0.1):
        """
        Initializes the Reed-Solomon outer code.

        Args:
            redundancy_percentage: The amount of parity relative to the original data. For example, 1.0 creates 100% 
                redundancy, making the parity as large as the data. This is a float between 0 and 1.

        Raises:
            ValueError: If redundancy_percentage is not greater than 0 and at most 1.
        """
        if not (0 < redundancy_percentage <= 1):
            raise ValueError("Redundancy percentage must be greater than 0 and at most 1.")

        self.redundancy_percentage: float = redundancy_percentage

        # Redundancy is measured relative to data strands, not the complete codeword.
        self.payload_per_block: int = int(self.MAX_BLOCK_STRANDS / (1 + redundancy_percentage))
        self.parity_per_block: int = self.MAX_BLOCK_STRANDS - self.payload_per_block

        self.num_blocks: int = 0

        # We cache the Reed-Solomon code for each parity_strands value to avoid reinitializing it multiple times. It 
        #   takes so long (usually seconds) so we are just going to lazy load it when we need it. Imagine multiple runs
        #   of the same parity_strands value, we don't want to reinitialize it every time.
        cache_key: tuple[int, int] = (self.GF_ORDER, self.parity_per_block)
        self.rs: galois.ReedSolomon = self._RS_CACHE.get(
            cache_key,
            galois.ReedSolomon(
                n=self.MAX_BLOCK_STRANDS,
                k=self.payload_per_block,
                field=self.GF
            )
        )
        if cache_key not in self._RS_CACHE:
            self._RS_CACHE[cache_key] = self.rs


    def _get_rs(self, parity_strands: int) -> galois.ReedSolomon:
        """
        Gets the Reed-Solomon code with the given number of parity symbols. A block with fewer payload strands uses a
        shortened version of this code (galois handles the shortening when given fewer message symbols than k), but 
        the parity count must match the code's n - k, so a partial block with less parity needs its own code.

        Args:
            parity_strands: The number of parity symbols (strands) per codeword.

        Returns:
            The cached Reed-Solomon code with that many parity symbols.
        """
        cache_key: tuple[int, int] = (self.GF_ORDER, parity_strands)
        if cache_key not in self._RS_CACHE:
            self._RS_CACHE[cache_key] = galois.ReedSolomon(
                n=self.MAX_BLOCK_STRANDS,
                k=self.MAX_BLOCK_STRANDS - parity_strands,
                field=self.GF
            )

        return self._RS_CACHE[cache_key]


    def _parity_for_block(self, payload_strands: int) -> int:
        """
        Gets the number of parity strands for a block holding payload_strands payload strands. Full blocks use 
        parity_per_block, while a partial block (only ever the last one) gets parity matching the redundancy percentage.

        Args:
            payload_strands: The number of payload strands in the block.

        Returns:
            The number of parity strands for the block.
        """
        if payload_strands >= self.payload_per_block:
            return self.parity_per_block

        return math.ceil(payload_strands * self.redundancy_percentage)


    def _payload_for_block(self, block_strands: int) -> int:
        """
        Inverse of _parity_for_block - recovers how many payload strands a block of block_strands total strands holds.
        Payload + parity strictly increases with the payload count, so there is at most one match.

        Args:
            block_strands: The total number of strands (payload and parity) in the block.

        Returns:
            The number of payload strands in the block.

        Raises:
            MismatchSegments: If no payload count produces a block of that size.
        """
        for payload_strands in range(block_strands, 0, -1):
            if payload_strands + self._parity_for_block(payload_strands) == block_strands:
                return payload_strands

        raise MismatchSegments(
            message=f"A block of {block_strands} strands does not match any payload and parity layout"
        )


    @override
    def fit_payload_length(self, payload_length: int) -> int:
        """
        Rounds the payload length down to a whole number of columns. A partial column cannot hold a full field symbol, 
        so the parity strands would lose part of every parity symbol in that column.

        Args:
            payload_length: The number of bases available for the payload.

        Returns:
            The payload length rounded down to a multiple of COLUMN_LENGTH.
        """
        return payload_length - payload_length % self.COLUMN_LENGTH


    def _is_valid_segment(self, segment: str, previous_base: str) -> bool:
        """
        Checks whether a segment can be converted into field symbols. A Goldman strand with a substitution that creates
        a homopolymer cannot be, so it has to be treated as an erasure instead.

        Args:
            segment: The DNA segment to check.
            previous_base: The base before the segment (the last base of its index).

        Returns:
            True if the segment can be converted, False otherwise.
        """
        try:
            Goldman().dna_to_trit(segment, previous_base)
            return True
        except EncodingError:
            return False


    @staticmethod
    def trits_to_field_symbol(trits: List[int]) -> int:
        """
        Converts six trits into one GF(3^6) symbol (basically trit to decimal with a list)

        Args:
            trits: The trits that make up a field symbol.

        Returns:
            The integer representation of the field symbol.
        """

        return int("".join(str(trit) for trit in trits), 3)


    def _validate_segments(self,
                           previous_segments: List[str] | List[str | None], 
                           payload_segments: List[str] | List[str | None]) -> None:
        """
        Validates the previous and payload segments for Reed-Solomon error correction.

        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put RS parity on.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
            MismatchSegments: If there are not enough previous sequences for the payload and parity strands.
        """
        if not previous_segments:
            raise ValueError("Previous segments cannot be empty.")
        if not payload_segments:
            raise ValueError("Payload segments cannot be empty.")
        if len(previous_segments) < len(payload_segments):
            raise MismatchSegments(
                message="There must be at least one previous sequence for each payload sequence"
            )


    def _segment_to_blocks(self, 
                           previous_segments: List[str], 
                           payload_segments: List[str]) -> tuple[List[List[str]], List[List[str]]]:
        """
        Segments are split into blocks of data_capacity strands - basically each segment into its own block.
        
        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put RS parity on.

        Returns:
            A tuple containing the previous blocks and the payload blocks.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
        """
        self._validate_segments(previous_segments, payload_segments)

        # Our previous blocks (likely our indices) need to be the entire block length as each strand will need its own
        #   index. The payload blocks will be the data_capacity length as we will add parity_strands to each block.
        previous_blocks: List[List[str]] = []
        for index in range(0, len(previous_segments), self.MAX_BLOCK_STRANDS):
            previous_blocks.append(previous_segments[index:index + self.MAX_BLOCK_STRANDS])

        # Payload blocks are up to the payload_per_block length
        payload_blocks: List[List[str]] = []
        for index in range(0, len(payload_segments), self.payload_per_block):
            payload_blocks.append(payload_segments[index:index + self.payload_per_block])

        return previous_blocks, payload_blocks


    def _dna_to_field_symbol(self, dna_sequence: str, previous_base: str) -> int:
        """
        Converts one Goldman-encoded DNA column into a GF(3^6) symbol.
        """
        trits: List[int] = [
            int(trit)
            for trit in Goldman().dna_to_trit(dna_sequence, previous_base)
        ]
        trits.extend([0] * (self.COLUMN_LENGTH - len(trits)))
        return self.trits_to_field_symbol(trits)


    @staticmethod
    def _field_symbol_to_dna(symbol: int, length: int, previous_base: str) -> str:
        """
        Converts one GF(3^6) parity symbol back into Goldman-encoded DNA.

        Args:
            symbol: The GF(3^6) parity symbol to be converted back into DNA.
            length: The length of the DNA sequence to be returned. This is used to ensure that the DNA sequence is the 
                same length as the original column, which may be less than 6 bases.
            previous_base: The previous base to be used for the DNA sequence.

        Returns:
            The DNA sequence corresponding to the GF(3^6) parity symbol.

        Raises:
            ValueError: If length symbol is less than 0.
            EncodingError: If the Goldman mapping fails to convert trits to DNA bases - happens when there's a 
                homopolymer.
        """
        trits: str = GoldmanIndexer.decimal_to_trit(
            decimal=symbol,
            width=RSOuter.COLUMN_LENGTH
        )
        dna_sequence: str = Goldman().trit_to_dna(trits, previous_base)[:length]

        return dna_sequence


    def _retrieve_payload_length(self, payload_segments: List[str] | List[str | None]) -> int:
        """
        Retrieves the length of the payload segments.

        Args:
            payload_segments: The payload segments to be put RS parity on.

        Returns:
            The length of the payload segments.

        Raises:
            ValueError: If payload_segments is empty/None.
            PayloadLengthError: If the payload length is 0.
        """
        if not payload_segments:
            raise ValueError("Payload segments cannot be empty.")

        # Get the next payload sequence that isn't none, if we don't find one, then the payload length is 0 and we 
        #   should raise an error, regardless if we allowing None's or not - we need them to have a length
        payload_length: int = next(
            (len(segment) for segment in payload_segments if segment is not None), 
            0
        )
        if payload_length == 0:
            raise PayloadLengthError(
                message="Payload segments must have a positive length",
                payload_length=payload_length
            )

        return payload_length


    def _retrieve_columns(self,
                          previous_segments: List[str], 
                          payload_segments: List[str] | List[str | None],
                          error_on_none: bool = False) -> list[list[galois.FieldArray]]:
        """
        Retrieves the columns of bases from the previous bases and payload segments.

        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put RS parity on.
            error_on_none: If True, raises an error if any of the payload segments are None. If False, skips any None

        Returns:
            A list of columns, where each column is a list of lists containing the previous base and the corresponding 
            payload bases.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
            EncodingError: If the Goldman mapping fails to convert DNA bases to trits - likely to happen when not 
                accounting for the specific spliced indices used for indexing.
            PayloadLengthError: If the payload length is 0.
        """
        self._validate_segments(previous_segments, payload_segments)

        # Need to know payload legnth
        payload_length: int = self._retrieve_payload_length(payload_segments)

        columns: List[List[galois.FieldArray]] = []

        # Each six-base DNA slice becomes one GF(3^6) symbol for RS encoding.
        for index in range(0, payload_length, RSOuter.COLUMN_LENGTH):
            column: List[galois.FieldArray] = []

            # Collect the same six-base position from every data strand.
            for segment_index, segment in enumerate(payload_segments):
                # Get our previous segment for current payload message
                previous_segment: str = previous_segments[segment_index]

                # If the segment is None, we either skip it or raise an error based on the error_on_none flag
                if segment is None and not error_on_none:
                    column.append(self.GF(0))
                    continue
                # Only reach if error_on_none is True and segment is None, so we raise an error
                if segment is None:
                    raise OuterCodeError(
                        message="Payload segment is None, but error_on_none is True",
                    )

                # The first slice starts after the strand's index. Later slices continue from the preceding DNA base in 
                #   the same strand
                previous_base: str
                if index == 0:
                    previous_base = previous_segment[-1]
                else:
                    previous_base = segment[index - 1]

                column_bases: str = segment[index:index + RSOuter.COLUMN_LENGTH]
                
                # The concrete outer-code variant converts the DNA column to the field representation used by its 
                #   Reed-Solomon code
                column.append(self.GF(self._dna_to_field_symbol(column_bases, previous_base)))

            columns.append(column)

        return columns


    def add_redundancy(self, previous_sequences: List[str], payload_sequences: List[str]) -> List[str]:
        """
        Adds Reed-Solomon parity strands into each block. Basically we take the payload strands and add parity strands 
        to them column by column. Each column is 6 bases long and gives us a GF(3^6) symbol (6 digits in base 3). We use
        a GF(3^6) in order to perform Goldman's rotation scheme to avoid homopolymers. Unfornately, this means we both 
        cannot use reedsolo library and cannot use our payload encoding scheme for the parity strands. We have to ensure
        that they remain the same length as the payload strands.

        Args:
            previous_sequences: The previous bases to be used for the columns.
            payload_sequences: The payload segments to be put RS parity on.

        Returns:
            A list of the payload sequences with Reed-Solomon parity added.

        Raises:
            ValueError: If previous_sequences or payload_sequences is empty/None.
            NotEnoughPreviousSegments: If there are not enough previous sequences for the payload and parity strands.
        """
        self._validate_segments(previous_sequences, payload_sequences)

        # Parity symbols need whole columns, otherwise we would have to truncate them
        payload_length: int = self._retrieve_payload_length(payload_sequences)
        if payload_length % self.COLUMN_LENGTH != 0:
            raise PayloadLengthError(
                message=f"Payload length must be a multiple of {self.COLUMN_LENGTH} for Reed-Solomon outer codes",
                payload_length=payload_length
            )

        # Every data and parity strand needs its own reserved index. Different from previous sequences being smaller,
        #   we need to know that we have enough sequences for all pairty strands. Different from the check of parity 
        #   strands being less than the data strands - this is do we have enough for adding parity?
        if len(previous_sequences) < self.calculate_total_strands(len(payload_sequences)):
            raise NotEnoughPreviousSegments(
                message="Not enough previous sequences for the payload and parity strands"
            )

        parity_strands: List[str] = []

        previous_blocks, payload_blocks = self._segment_to_blocks(previous_sequences, payload_sequences)
        for previous_block, payload_block in zip(previous_blocks, payload_blocks):
            # The last block may not be full, so it gets parity matching the redundancy percentage for that block
            payload_per_block: int = len(payload_block)
            parity_per_block: int = self._parity_for_block(payload_per_block)
            rs: galois.ReedSolomon = self._get_rs(parity_per_block)

            # Get our columns for this block, which is a all the GF symbols - if we got none we got a problem
            columns: List[List[galois.FieldArray]] = self._retrieve_columns(
                previous_segments=previous_block, 
                payload_segments=payload_block,
                error_on_none=True
            )

            # Each output entry is one parity strand assembled column by column, so we need to initalize them all first
            #   in order to ensure we can append to them in correct order
            parity_block: List[str] = [""] * parity_per_block

            # We need to keep track of the last base - currently its all the indexes (we assume)
            parity_previous_bases: List[str] = []
            for index in range(parity_per_block):
                parity_previous_bases.append(previous_block[payload_per_block + index][-1])
        
            for column_index, column in enumerate(columns):
                # Preserve the actual payload length when the final column is shorter than six bases
                column_length: int = min(
                    RSOuter.COLUMN_LENGTH,
                    len(payload_block[0]) - column_index * RSOuter.COLUMN_LENGTH
                )

                # RS returns one field symbol for each parity strand for this column
                parity_column: galois.FieldArray = rs.encode(self.GF(column), output="parity")

                # We need to convert each parity symbol back into DNA bases and add them to corresponding parity strand
                for parity_index in range(parity_per_block):
                    # Convert to a usable int based on the field
                    parity_sequence: str = self._field_symbol_to_dna(
                        symbol=int(parity_column[parity_index]),
                        previous_base=parity_previous_bases[parity_index],
                        length=column_length
                    )

                    parity_block[parity_index] += parity_sequence

                    # Goldman needs the last emitted base for the next column. Direct DNA conversion does not use this
                    if parity_sequence:
                        parity_previous_bases[parity_index] = parity_sequence[-1]

            # Add the parity block to the list of all parity strands - keeps our order of blocks correct too
            parity_strands.extend(parity_block)

        return parity_strands


    def _get_erasure_mask(self, payload_block: List[str | None], ) -> List[bool]:
        """
        Creates an erasure mask for the given sequences. True indicates an erasure (None), and False indicates a valid 
        sequence.

        Args:
            payload_block: The list of payload segments.

        Returns:
            A list of booleans representing the erasure mask.
        """
        # Get the erasure mask - everywhere they are are 
        erasure_mask: List[bool] = []
        for sequence in payload_block:
            if sequence is None:
                erasure_mask.append(True)
                continue

            erasure_mask.append(False)

        return erasure_mask


    def perform_correction(self, 
                           previous_sequences: List[str], 
                           encoded_sequences: List[str | None] ) -> List[str | None]:
        """
        Performs Reed-Solomon error correction on the encoded sequences.

        Args:
            previous_sequences: The complete list of index sequences. These provide the preceding base needed to
                convert the first six-base column back from a field symbol. Assumed to be sorted. We regenerate any 
                missing one, so no worries about them being none from erasure
            encoded_sequences: The list of encoded sequences we retrieve after decoding. We assume that sequences that
                are None are the ones marked for erasure. The sequences that are not None are assumed to be correct and 
                will be used to correct the others. Also assumes that the sequences are in the same order as they were 
                originally encoded (that is sorted by index), so we can use the previous

        Returns:
            The list of corrected sequences without the parity strands. Sequences that could not be corrected = None.
        
        Raises:
            ValueError: If encoded_sequences or previous_sequences is empty/None.
            PayloadLengthError: If the payload length is 0.
            MistmatchSegments: If there are not enough previous sequences for the payload and parity strands. Or if the 
                RS block does not contain enough payload and parity strands to perform correction.
        """
        self._validate_segments(previous_sequences, encoded_sequences)

        # Must be same length in this scenario
        if len(previous_sequences) != len(encoded_sequences):
            raise MismatchSegments(
                message="Previous and encoded sequences must be the same length for performing correction"
            )

        # Some sequences we may not be able to correct, unlikely. Returning None means those sequences are forever lost
        corrected_sequences: List[str | None] = []

        # Should remain true for all blocks. Use the most common length so one damaged strand can't set it
        self._retrieve_payload_length(encoded_sequences)
        payload_length: int = Counter(
            len(segment) for segment in encoded_sequences if segment is not None
        ).most_common(1)[0][0]

        # Go through each block and correct one at at time, ending block (likely not full) will be handled the same
        for block_start in range(0, len(encoded_sequences), self.MAX_BLOCK_STRANDS):
            # Get the block of previous sequences and encoded sequences for this RS block
            previous_block: List[str] = previous_sequences[block_start:block_start + self.MAX_BLOCK_STRANDS]
            payload_block: List[str | None] = encoded_sequences[block_start:block_start + self.MAX_BLOCK_STRANDS]

            # Strands that are the wrong length or cannot be converted to field symbols (Goldman strands with a 
            #   homopolymer) are unusable, so we mark them for erasure rather than failing the whole block
            payload_block = [
                segment
                if segment is not None
                    and len(segment) == payload_length
                    and self._is_valid_segment(segment, previous_block[segment_index][-1])
                else None
                for segment_index, segment in enumerate(payload_block)
            ]

            # The last block may not be full, so recover its payload/parity split the same way add_redundancy made it
            data_count: int = self._payload_for_block(len(payload_block))
            rs: galois.ReedSolomon = self._get_rs(len(payload_block) - data_count)

            # Nothing usable in this block, so there is nothing to correct from
            if all(segment is None for segment in payload_block):
                corrected_sequences.extend([None] * data_count)
                continue

            erasure_mask: List[bool] = self._get_erasure_mask(payload_block)

            # Get the columns for this block, which is a all the GF symbols
            columns: List[List[galois.FieldArray]] = self._retrieve_columns(
                previous_segments=previous_block, 
                payload_segments=payload_block,
                error_on_none=False
            )

            # output="message" returns only the data symbols, so account for it
            corrected_block: List[str] = [""] * data_count
            block_failed: bool = False

            for column_index, column in enumerate(columns):
                # Preserve the actual payload length when the final column is shorter than six bases
                column_length: int = min(
                    RSOuter.COLUMN_LENGTH,
                    payload_length - column_index * RSOuter.COLUMN_LENGTH
                )

                # RS returns one field symbol for each parity strand for this column
                corrected_column: galois.FieldArray
                errors: NDArray[np.int_] | int
                corrected_column, errors = rs.decode(
                    codeword=self.GF(column),
                    erasures=np.array(erasure_mask),
                    output="message",
                    errors=True
                )

                # Galois reports an undecodable column with errors == -1. Goldman decoding is stateful across columns, 
                #   so no later column can make a complete DNA strand valid after this point
                if isinstance(errors, int) and errors == -1:
                    block_failed = True
                    break

                # We need to convert each corrected symbol back into DNA bases and add them to corresponding payload 
                #   strand. The last strand may not be uniform with the data capacity.
                for payload_index in range(data_count):
                    # Get our previous base for this strand
                    if column_index == 0:
                        previous_base: str = previous_block[payload_index][-1]
                    else:
                        previous_base: str = corrected_block[payload_index][column_index * RSOuter.COLUMN_LENGTH - 1]

                    # Convert corrected symbol back to DNA bases
                    corrected_sequence: str = self._field_symbol_to_dna(
                        symbol=int(corrected_column[payload_index]),
                        previous_base=previous_base,
                        length=column_length
                    )

                    corrected_block[payload_index] += corrected_sequence

            # A failed column prevents recovery of erased strands, but intact data strands remain usable. Do not return 
            #   partially reconstructed strands because Goldman columns depend on their predecessors.
            if block_failed:
                corrected_sequences.extend(payload_block[:data_count])
                continue

            # Add the corrected block to the list of corrected sequences
            corrected_sequences.extend(corrected_block)
            
        return corrected_sequences


    def calculate_total_strands(self, payload_strands: int) -> int:
        """
        Return full RS codeword strands for the payload strand count.

        Args:
            payload_strands: The number of payload strands to be encoded with Reed-Solomon parity
        
        Returns:
            The total number of strands after adding Reed-Solomon parity.
        """
        if payload_strands <= 0:
            raise ValueError("Payload strand count must be positive.")

        # Update the number of blocks based on the payload strands and payload per block. This is used to calculate the 
        #   total number of strands after adding parity.
        blocks: int = (payload_strands + self.payload_per_block - 1) // self.payload_per_block
        self.num_blocks = blocks

        # Every block but the last is full. The last block holds whatever payload is left, plus its own parity
        full_blocks: int = blocks - 1
        last_payload: int = payload_strands - full_blocks * self.payload_per_block

        return full_blocks * self.MAX_BLOCK_STRANDS + last_payload + self._parity_for_block(last_payload)


    def select_payload_indices(self, indices: List[str]) -> List[str]:
        """
        Remove parity indices from each RS block before payload encoding.

        The returned indices retain their original block order, so each payload segment is encoded with the same 
        predecessor it will have in  the final framed data.

        Args:
            indices: The complete list of indices, including both payload and parity indices.

        Returns:
            A list of indices that are used for payload encoding, with parity indices removed.

        Raises:
            ValueError: If indices is empty or None.
        """
        if not indices:
            raise ValueError("Indices cannot be empty.")

        payload_indices: List[str] = []
        for block_start in range(0, len(indices), self.MAX_BLOCK_STRANDS):
            block: List[str] = indices[block_start:block_start + self.MAX_BLOCK_STRANDS]

            # Add the first payload_per_block indices from the block to the payload_indices list
            payload_indices += block[:self.payload_per_block]
        
        return payload_indices


class DirectDNARSOuter(RSOuter):
    """
    Reed-Solomon outer code for encodings that allow homopolymers. We cannot convert "GGG" from Goldman trits and back
    as the rotation scheme doesnt allow for it. This DirectDNARSOuter class allows for direct DNA to GF(4^6) conversion
    without using Goldman's rotation scheme. Where we basically assigning each digit a base-4 value and converting the
    them with six nt's (too keep consitient with the RSOuter class) into a GF(4^6) symbol.
    """

    GF_ORDER: int = 4**6
    GF: type[galois.FieldArray] = galois.GF(GF_ORDER)

    # Use the maximum Reed-Solomon codeword length supported by GF(4096)
    MAX_BLOCK_STRANDS: int = GF_ORDER - 1

    DNA_TO_DIGIT: Final[dict[str, int]] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }
    DIGIT_TO_DNA: Final[dict[int, str]] = {v: k for k, v in DNA_TO_DIGIT.items()}

    def __init__(self, redundancy_percentage: float = 0.1):
        """
        Initializes the DirectDNARSOuter with a specified redundancy percentage.

        Args:
            redundancy_percentage: The percentage of redundancy to be used in the outer code.
        """
        super().__init__(redundancy_percentage=redundancy_percentage)


    @override
    def _is_valid_segment(self, segment: str, *args, **kwargs) -> bool:
        """
        Checks whether a segment can be converted into field symbols - any string of A, C, G, and T can be.

        Args:
            segment: The DNA segment to check.
            *args and **kwargs, used to maintain compatibility with the RSOuter class, but not used here.

        Returns:
            True if the segment only contains DNA bases, False otherwise.
        """
        return all(base in self.DNA_TO_DIGIT for base in segment)


    @override
    def _dna_to_field_symbol(self, dna_sequence: str, *args, **kwargs) -> int:
        """
        Converts up to six arbitrary DNA bases into one GF(4^6) symbol.

        Args:
            dna_sequence: The DNA bases to be converted into a GF(4^6) symbol.
            *args and **kwargs, used to maintain compatibility with the RSOuter class, but not used here.

        Returns:
            The integer representation of the GF(4^6) symbol.
        """
        symbol: int = 0
        for base in dna_sequence:
            symbol = symbol * 4 + self.DNA_TO_DIGIT[base]

        for _ in range(self.COLUMN_LENGTH - len(dna_sequence)):
            symbol *= 4

        return symbol


    @override
    def _field_symbol_to_dna(self, symbol: int, length: int, *args, **kwargs) -> str:
        """
        Converts one GF(4^6) data symbol back into DNA bases. Essientaly 'quits' or qubits? to decimal. Only used for
        data strands, parity strands go through _symbols_to_parity_dna.

        Args:
            symbol: The GF(4^6) symbol to be converted back into DNA.
            length: The length of the DNA sequence to be returned. This is used to ensure that the DNA sequence is the
                same length as the original column.
            *args and **kwargs, used to maintain compatibility

        Returns:
            The DNA sequence corresponding to the GF(4^6) symbol.
        """
        bases: str = ""
        for _ in range(self.COLUMN_LENGTH):
            digit: int = symbol % 4
            symbol //= 4
            # Prepend each least-significant base so the final sequence stays in its original left-to-right order.
            bases = self.DIGIT_TO_DNA[digit] + bases

        return bases[:length]