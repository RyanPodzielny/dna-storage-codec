"""
Module provides outer error correction codes for DNA encoding. Outer codes are applied to the payload segments after
inner error correction codes have been applied. They are used to correct errors that occur during the synthesis
and sequencing of the DNA.
"""


from abc import abstractmethod
from typing import ClassVar, Final, override
from numpy.typing import NDArray

# Can't use reedsolo because it doesn't support GF(3^6) and has a max of 255 symbols. We need to encode each column of
#   the oligos as 6-bit symbols, which gives us 3^6 = 729 symbols. This way we can retain the same fixed-length oligo
#   and use Goldman's rotation scheme to avoid homopolymers. It is much slower than reedsolo, but it the best option.
import galois
import numpy as np
import math

from Encode.Helpers.Errors import EncodingError, MismatchSegments, NotEnoughPreviousSegments, OuterCodeError, \
    PayloadLengthError
from Encode.Indexer import GoldmanIndexer
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import Goldman, Huffman


class OuterCodes():
    """
    Serves as a base class for outer error correction codes. 
    """
    def __init__(self):
        self.payload_per_block: int = 0
        self.parity_per_block: int = 0
        # Full block length (payload + parity) - each strand in the block gets its own index
        self.block_strands: int = 0
        self.num_blocks: int = 0


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    DNA_TO_DIGIT: Final[dict[str, int]] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }
    DIGIT_TO_DNA: Final[dict[int, str]] = {v: k for k, v in DNA_TO_DIGIT.items()}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _validate_segments(self,
                           previous_segments: list[str] | list[str | None], 
                           payload_segments: list[str] | list[str | None]) -> None:
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
                           previous_segments: list[str],
                           payload_segments: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        """
        Segments are split into blocks of strands - basically each segment into its own block.

        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put parity on.

        Returns:
            A tuple containing the previous blocks and the payload blocks.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
        """
        self._validate_segments(previous_segments, payload_segments)

        # Our previous blocks (likely our indices) need to be the entire block length as each strand will need its own
        #   index. The payload blocks will be the block strand length as we will add parity_strands to each block.
        previous_blocks: list[list[str]] = []
        for index in range(0, len(previous_segments), self.block_strands):
            previous_blocks.append(previous_segments[index:index + self.block_strands])

        # Payload blocks are up to the payload_per_block length
        payload_blocks: list[list[str]] = []
        for index in range(0, len(payload_segments), self.payload_per_block):
            payload_blocks.append(payload_segments[index:index + self.payload_per_block])

        return previous_blocks, payload_blocks


    def _retrieve_payload_length(self, payload_segments: list[str] | list[str | None]) -> int:
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


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def align_payload_length(payload_length: int) -> int:
        raise NotImplementedError()


    @abstractmethod
    def add_redundancy(self, previous_sequences: list[str], payload_sequences: list[str]) -> list[str]:
        raise NotImplementedError()


    @abstractmethod
    def perform_correction(self, previous_sequences: list[str], encoded_sequences: list[str | None]) -> list[str | None]:
        raise NotImplementedError()


    @abstractmethod
    def calculate_total_strands(self, payload_strands: int) -> int:
        raise NotImplementedError()


    def select_payload_indices(self, indices: list[str]) -> list[str]:
        """
        Remove parity indices from each block before payload encoding.

        The returned indices retain their original block order, so each payload segment is encoded with the same
        predecessor it will have in the final framed data.

        Args:
            indices: The complete list of indices, including both payload and parity indices.

        Returns:
            A list of indices that are used for payload encoding, with parity indices removed. The last block is usually
            shorter than a full block, so the list may end with indices past the payload count - these are ignored as
            they sit where the next payload strand would go in a full block.

        Raises:
            ValueError: If indices is empty or None.
        """
        if not indices:
            raise ValueError("Indices cannot be empty.")

        payload_indices: list[str] = []
        for block_start in range(0, len(indices), self.block_strands):
            block: list[str] = indices[block_start:block_start + self.block_strands]

            # Add the first payload_per_block indices from the block to the payload_indices list
            payload_indices += block[:self.payload_per_block]

        return payload_indices


    def arrange_strands(self, payload_segments: list[str], parity_segments: list[str]) -> list[str]:
        """
        Arranges the payload and parity strands in the order they will be framed. Each block of payload strands is
        followed by its own parity strands, so they all stay in order with the indices.

        Args:
            payload_segments: The payload segments from the encoder.
            parity_segments: The parity segments from add_redundancy.

        Returns:
            The payload and parity segments in the order they will be framed.
        """
        # Split on number of parity segments
        blocked_parity_segments: list[list[str]] = []
        for index in range(0, len(parity_segments), self.parity_per_block):
            blocked_parity_segments.append(parity_segments[index:index + self.parity_per_block])

        # Now travel through each block of payload segments and add the parity segments to the end of each block,
        #   so that they all stay in order
        arranged_segments: list[str] = []
        parity_block_index: int = 0
        for index in range(0, len(payload_segments), self.payload_per_block):
            payload_block: list[str] = payload_segments[index:index + self.payload_per_block]
            parity_block: list[str] = blocked_parity_segments[parity_block_index]

            # Add the parity segments to the end of the payload segments for this block
            arranged_segments.extend(payload_block + parity_block)
            parity_block_index += 1

        return arranged_segments



class GoldmanRedundancy(OuterCodes):
    """
    Creates four quarter shifted copies of a payload stream, based on Goldman. All the payload strands are joined into 
    one trit stream and split into quarters. Each framed strand (window) is four quarters long and starts one quarter 
    after the previous one, so every quarter shows up in four different windows. Every odd window is reversed to give 
    each copy a different sequence (adds bit more diversity for recovery). We pad the stream with three quarters of 0
    trits at each end so the first and last quarters still get all four copies.

    Unlike XOR and RS, the payload strands themselves are never framed - only the windows are. Each window is rotated
    from its own index base so it never makes a homopolymer with its index. Requires homopolymer-free payloads as we
    use Goldman's rotation scheme to get the trits.
    """

    def __init__(self):
        """
        Initializes the GoldmanRedundancy. Each payload strand ends up as roughly four windows.
        """
        super().__init__()
        # Not a true block code, but we can treat each payload strand as a block of one strand, and each window as a 
        #   block of four strands.
        self.payload_per_block: int = 1
        self.parity_per_block: int = 3
        self.block_strands: int = self.payload_per_block + self.parity_per_block


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # How many copies of each quarter we make, and how many quarters are in a window
    LAYERS: Final[int] = 4


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    @staticmethod
    def align_payload_length(payload_length: int) -> int:
        """
        Aligns the payload length to a multiple of four, so each window splits into four whole quarters.

        Args:
            payload_length: The number of bases available for the payload.

        Returns:
            The largest multiple of four that fits in the payload length.

        Raises:
            PayloadLengthError: If the aligned payload length is 0.
        """
        aligned_length: int = payload_length - (payload_length % GoldmanRedundancy.LAYERS)

        if aligned_length <= 0:
            raise PayloadLengthError(
                message="Payload length must fit at least one base per quarter",
                payload_length=payload_length
            )

        return aligned_length


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _to_digits(self, dna_sequence: str, previous_base: str) -> str:
        """
        Converts DNA to the digits we build the stream from. Goldman trits, so it needs the previous base. Overridde
        in DirectDNAGoldmanRedundancy to use quats instead of trits.

        Args:
            dna_sequence: The DNA bases to be converted.
            previous_base: The base before the sequence (last base of its index).

        Returns:
            The trit string of the DNA bases.

        Raises:
            EncodingError: If the DNA sequence contains invalid bases or cannot be converted to trits.
        """
        return Goldman().dna_to_trit(dna_sequence, previous_base)


    def _to_dna(self, digits: str, previous_base: str) -> str:
        """
        Converts the digits back to DNA, rotating from the previous base so there are no homopolymers. Overridde in 
        DirectDNAGoldmanRedundancy to use quats instead of trits.
 
        Args:
            digits: The trit string to be converted.
            previous_base: The base before the sequence (last base of its index).

        Returns:
            The DNA bases of the trit string.

        Raises:
            ValueError: If the trit string contains invalid trits.
            EncodingError: If the trit string cannot be converted to DNA.
        """
        return Goldman().trit_to_dna(digits, previous_base)


    def _reverse_window(self, digits: str) -> str:
        """
        Reverses an odd window. Reversing the trits instead of taking the reverse complement of the DNA keeps the window
        rotating from its own index base. Reversing twice gives back the original, so this also undoes it.

        Args:
            digits: The trit string of the window.

        Returns:
            The reversed trit string.
        """
        return digits[::-1]


    def _choose_quarter(self, quarter_copies: list[str]) -> str | None:
        """
        Picks the quarter from all its copies that survived. Takes the most common copy, so a copy with an error that
        got past the checksum is outvoted by the others. Ties go to the first copy seen.

        Args:
            quarter_copies: Every surviving copy of the quarter.

        Returns:
            The most common copy, or None if no copies survived (the quarter is lost).
        """
        if not quarter_copies:
            return None

        # Count each distinct copy
        counts: dict[str, int] = {}
        for quarter_copy in quarter_copies:
            counts[quarter_copy] = counts.get(quarter_copy, 0) + 1

        # Find the most common copy. Only replace on a strictly higher count, so ties stay with the first copy seen
        best_copy: str = quarter_copies[0]
        for quarter_copy, count in counts.items():
            if count > counts[best_copy]:
                best_copy = quarter_copy

        return best_copy


    def _validate_quarter_length(self, payload_length: int) -> None:
        """
        Ensures each window splits into LAYERS whole quarters. Otherwise the quarters drift away from the payload strands
        a little more with every strand, and we would silently rebuild them from the wrong part of the stream.

        Args:
            payload_length: The length of the payload strands (and windows).

        Raises:
            PayloadLengthError: If the payload length is not a multiple of LAYERS.
        """
        if payload_length % self.LAYERS != 0:
            raise PayloadLengthError(
                message=f"Payload length must be a multiple of {self.LAYERS}",
                payload_length=payload_length
            )


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def add_redundancy(self, previous_sequences: list[str], payload_sequences: list[str]) -> list[str]:
        """
        Creates the overlapping windows from the payload strands. The payload strands are joined into one trit stream,
        padded, and split into windows of four quarters, each starting one quarter after the previous one.

        Args:
            previous_sequences: The complete list of index sequences. Payload strand k was encoded from index k, and
                window i is framed with index i.
            payload_sequences: The payload sequences to make the windows from.

        Returns:
            A list of windows, which are the only strands that get framed.

        Raises:
            ValueError: If previous_sequences or payload_sequences is empty/None.
            MismatchSegments: If there are not enough previous sequences for the payload strands.
            PayloadLengthError: If the payload length is 0 or not a multiple of LAYERS.
            NotEnoughPreviousSegments: If there are not enough previous sequences for all the windows.
        """
        # Guard clauses
        self._validate_segments(previous_sequences, payload_sequences)
        payload_length: int = self._retrieve_payload_length(payload_sequences)
        self._validate_quarter_length(payload_length)

        # Every window needs its own reserved index - ensure we have enough of them
        num_windows: int = self.calculate_total_strands(len(payload_sequences))
        if len(previous_sequences) < num_windows:
            raise NotEnoughPreviousSegments(
                message="Not enough previous sequences for all the windows"
            )

        # Join all the payload strands into one continuous stream. Each payload strand was encoded from the last base of
        #   its index, so convert it to trits from that same base
        stream: str = ""
        for index, payload_sequence in enumerate(payload_sequences):
            previous_base: str = previous_sequences[index][-1]
            stream += self._to_digits(payload_sequence, previous_base)

        # Pad both ends with three quarters of 0 trits, so the first and last quarters are in four windows too. We 
        #   ensure that our payload length is a multiple of LAYERS, so the quarters are all the same length 
        quarter_length: int = payload_length // self.LAYERS
        padding: str = "0" * (quarter_length * (self.LAYERS - 1))
        stream = padding + stream + padding

        windows: list[str] = []
        for window_index in range(num_windows):
            # Each window is four quarters, starting one quarter after the previous window
            window_start: int = window_index * quarter_length
            window_digits: str = stream[window_start:window_start + payload_length]

            # Every odd window gets reversed, so the copies of a quarter don't all look the same. It's similar to how
            #   Goldman did it with reverse complement. It's real job is to make the windows different so if there is 
            #   is a common error, it doesn't affect all four copies of a quarter. 
            if window_index % 2 == 1:
                window_digits = self._reverse_window(window_digits)

            # Rotate from the window's own index base, not the payload's, as the window is what gets framed
            previous_base: str = previous_sequences[window_index][-1]
            windows.append(self._to_dna(window_digits, previous_base))

        return windows


    @override
    def perform_correction(self,
                           previous_sequences: list[str],
                           encoded_sequences: list[str | None]) -> list[str | None]:
        """
        Rebuilds the payload strands from the windows we have left. Every window gives a copy of its four quarters, and
        each quarter is picked from its copies. A payload strand is lost only if one of its quarters lost all four
        copies.

        Args:
            previous_sequences: The complete list of index sequences. Assumed to be sorted. We regenerate any missing
                one, so no worries about them being none from erasure
            encoded_sequences: The list of windows we retrieve after decoding. We assume that sequences that are None
                are the ones marked for erasure. Also assumes that the sequences are in the same order as they were
                originally encoded (that is sorted by index), as a window's position tells us which quarters it holds.

        Returns:
            The list of rebuilt payload strands. A payload strand with a lost quarter stays None.

        Raises:
            ValueError: If previous_sequences or encoded_sequences is empty/None.
            PayloadLengthError: If the payload length is 0 or not a multiple of LAYERS.
            MistmatchSegments: If previous_sequences and encoded_sequences are not the same length, or the number of
                windows doesn't match any number of payload strands.
        """
        # Guard clauses
        self._validate_segments(previous_sequences, encoded_sequences)
        payload_length: int = self._retrieve_payload_length(encoded_sequences)
        self._validate_quarter_length(payload_length)

        # Must be same length in this scenario - ensures we have the same number of oligos we encodeded with
        if len(previous_sequences) != len(encoded_sequences):
            raise MismatchSegments(
                message="Previous and encoded sequences must be the same length for performing correction"
            )

        # Inverse of calculate_total_strands - there are LAYERS windows per payload strand, plus the padding windows
        num_windows: int = len(encoded_sequences)
        if num_windows < self.LAYERS or (num_windows - (self.LAYERS - 1)) % self.LAYERS != 0:
            raise MismatchSegments(
                message=f"{num_windows} windows doesn't match any number of payload strands"
            )
        num_payload_strands: int = (num_windows - (self.LAYERS - 1)) // self.LAYERS

        # Every quarter of the padded stream, with all the copies of it we find. Need to take into account the padding
        #   quarters at the start and end, which are not part of any payload strand.
        quarter_length: int = payload_length // self.LAYERS
        num_quarters: int = num_payload_strands * self.LAYERS + 2 * (self.LAYERS - 1)
        quarter_copies: list[list[str]] = []
        for _ in range(num_quarters):
            quarter_copies.append([])

        # Every window we have left gives a copy of the four quarters it holds - we need to rebuild the quarters from
        #   their copies, and then rebuild the payload strands from the quarters.
        for window_index, window in enumerate(encoded_sequences):
            if window is None:
                continue

            # Undo add_redundancy: trits from the window's own index base, and unreverse the odd windows
            previous_base: str = previous_sequences[window_index][-1]
            window_digits: str = self._to_digits(window, previous_base)
            if window_index % 2 == 1:
                window_digits = self._reverse_window(window_digits)

            # Add all the quarters from this window to the list of copies for each quarter
            for layer in range(self.LAYERS):
                quarter_start: int = layer * quarter_length
                quarter_copies[window_index + layer].append(window_digits[quarter_start:quarter_start + quarter_length])

        # Some sequences we may not be able to correct. Returning None means those sequences are forever lost
        corrected_sequences: list[str | None] = []

        for payload_index in range(num_payload_strands):
            # Our real quarters start after the padding quarters
            first_quarter: int = (self.LAYERS - 1) + payload_index * self.LAYERS

            payload_digits: str = ""
            payload_lost: bool = False
            for quarter_index in range(first_quarter, first_quarter + self.LAYERS):
                quarter: str | None = self._choose_quarter(quarter_copies[quarter_index])

                # Lost all four copies of this quarter, so we can't rebuild the payload strand
                if quarter is None:
                    payload_lost = True
                    break

                payload_digits += quarter

            # Segment couldn't be rebuilt, so we return None for it
            if payload_lost:
                corrected_sequences.append(None)
                continue

            # Payload strand k was encoded from the last base of index k, so rotate it back from there
            previous_base: str = previous_sequences[payload_index][-1]
            corrected_sequences.append(self._to_dna(payload_digits, previous_base))

        return corrected_sequences


    @override
    def calculate_total_strands(self, payload_strands: int) -> int:
        """
        Return the number of windows for the payload strand count.

        Args:
            payload_strands: The number of payload strands to make windows from.

        Returns:
            The total number of windows. The stream has LAYERS quarters per payload strand plus the padding quarters at
            each end, and there is one window starting at every quarter that still has four quarters after it.

        Raises:
            ValueError: If payload_strands is not positive.
        """
        if payload_strands <= 0:
            raise ValueError("Payload strand count must be positive.")

        return payload_strands * self.LAYERS + (self.LAYERS - 1)


    @override
    def select_payload_indices(self, indices: list[str]) -> list[str]:
        """
        Every index is used. Payload strands are never framed, so payload strand k just uses the last base of index k
        as the starting base to encode from (and decode back to). Window i is framed with index i.

        Args:
            indices: The complete list of indices.

        Returns:
            The same list of indices. There are more indices than payload strands, the extra ones are ignored.

        Raises:
            ValueError: If indices is empty or None.
        """
        if not indices:
            raise ValueError("Indices cannot be empty.")

        return indices


    @override
    def arrange_strands(self, payload_segments: list[str], parity_segments: list[str]) -> list[str]:
        """
        Only the windows are framed - each piece of the payload strands is already in four of them.

        Args:
            payload_segments: The payload segments from the encoder, not framed. Used to keep polymorphism.
            parity_segments: The windows from add_redundancy.

        Returns:
            The windows, in the order they will be framed.
        """
        return parity_segments



class DirectDNAGoldmanRedundancy(GoldmanRedundancy):
    """
    Goldman's four layer redundancy for encodings that allow homopolymers. We cannot convert "GGG" from Goldman trits
    and back as the rotation scheme doesnt allow for it. Instead each base is directly a base-4 digit (quat), and odd
    windows are the true reverse complement like in Goldman's paper. The windows, padding and correction all work the
    same as GoldmanRedundancy - previous bases are not needed.
    """

    def __init__(self):
        """
        Initializes the DirectDNAGoldmanRedundancy, same windows as GoldmanRedundancy.
        """
        super().__init__()


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def _to_digits(self, dna_sequence: str, previous_base: str) -> str:
        """
        Converts DNA bases directly into a quat string, each base being its own base-4 digit.

        Args:
            dna_sequence: The DNA bases to be converted.
            previous_base: Used to maintain compatibility with the GoldmanRedundancy class, but not used here.

        Returns:
            The quat string of the DNA bases.
        """
        quats: str = ""
        for base in dna_sequence:
            quats += str(OuterCodes.DNA_TO_DIGIT[base])

        return quats


    @override
    def _to_dna(self, digits: str, previous_base: str) -> str:
        """
        Converts a quat string directly back into DNA bases.

        Args:
            digits: The quat string to be converted.
            previous_base: Used to maintain compatibility with the GoldmanRedundancy class, but not used here.

        Returns:
            The DNA bases of the quat string.
        """
        dna_sequence: str = ""
        for quat in digits:
            dna_sequence += OuterCodes.DIGIT_TO_DNA[int(quat)]

        return dna_sequence


    @override
    def _reverse_window(self, digits: str) -> str:
        """
        Takes the reverse complement of an odd window, like in Goldman's paper. With A=0, C=1, G=2, T=3 the complement
        of a base is 3 - its digit. Taking it twice gives back the original, so this also undoes it.

        Args:
            digits: The quat string of the window.

        Returns:
            The quat string of the reverse complement.
        """
        reverse_complement: str = ""
        for quat in reversed(digits):
            reverse_complement += str(3 - int(quat))

        return reverse_complement



class XOR(OuterCodes):
    """
    Implements a simple XOR-based error correction scheme on the DNA itself. Requires the strands to be homopolymer-free
    so we can use Goldman's rotation scheme to encode the strands. Each parity strand is the XOR of two payload 
    strands (mod 3). If one of the payload strands is missing, we can recover it using the parity strand and the other
    payload strand. Will fail if more than one payload strand is missing in a block, since we only have one parity 
    strand per block.
    """

    def __init__(self):
        super().__init__()
        self.payload_per_block: int = 2
        self.parity_per_block: int = 1
        self.block_strands: int = self.payload_per_block + self.parity_per_block


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    @staticmethod
    def align_payload_length(payload_length: int) -> int:
        """
        Aligns the payload to what is is. No aligment needed for XOR, just need the strands to same length.

        Args:
            payload_length: The number of bases available for the payload.

        Returns:
            The payload length.

        Raises:
            PayloadLengthError: If the payload length is 0.
        """
        if payload_length <= 0:
            raise PayloadLengthError(
                message="Payload length must be greater than 0",
                payload_length=payload_length
            )

        return payload_length


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _XOR_mod(self, dit_strand_1: str, dit_strand_2: str, mod: int = 3) -> str:
        """
        Performs XOR on two dit strings. The XOR is performed mod 3, so the result is a trit string of the same length.
        The sum is negated so that strand 1 + strand 2 + parity = 0 (mod 3), the same as A ^ B ^ P = 0 in binary XOR.
        This way any missing strand is recovered the same way: -(sum of the other two) mod 3.

        Args:
            dit_strand_1: The first dit string.
            dit_strand_2: The second dit string.
            mod: The modulus to use for the XOR operation. Default is 3 for trits.

        Returns:
            The result of the XOR operation as a dit (unit of measure, either trit or quat) string.

        Raises:
            ValueError: If the dit strings are not of the same length or contain invalid dit values.
        """
        # Check that the trit strings are of the same length
        if len(dit_strand_1) != len(dit_strand_2):
            raise ValueError("Trit strings must be of the same length.")

        result: str = ""

        # Perform XOR on each pair of dits - each position is its own sum, nothing carries over to the next one
        for dit_1, dit_2 in zip(dit_strand_1, dit_strand_2):
            # Ensure we are dealing with actual valid numbers
            if not dit_1.isdigit() or not dit_2.isdigit() or int(dit_1) >= mod or int(dit_2) >= mod:
                raise ValueError(f"Invalid dit values: {dit_1}, {dit_2}. Trits must be in the range [0, {mod - 1}].")

            # We always negate the sum so that strand 1 + strand 2 + parity = 0 (mod 3), the same as A ^ B ^ P = 0 in
            #   binary XOR. This way any missing strand is recovered the same way: -(sum of the other two) mod 3.
            result += str(-(int(dit_1) + int(dit_2)) % mod)

        return result


    def _get_parity_strand(self,
                           payload_strand_1: str,
                           payload_strand_2: str | None,
                           previous_bases: list[str]) -> str:
        """
        Gets the parity strand for two payload strands. Since strand 1 + strand 2 + parity = 0 (mod 3), this is also how
        we recover any missing strand: pass in the two strands we have and the previous base of the missing one last.

        Args:
            payload_strand_1: The first payload strand.
            payload_strand_2: The second payload strand. None if there isn't one (odd number of payload strands), we
              then XOR with a virtual strand of all 0 trits.
            previous_bases: The previous bases to use for the columns. In lock step: previous_base_1, previous_base_2,
              previous_base_parity. Without a second payload strand: previous_base_1, previous_base_parity.

        Returns:
            The parity strand as a DNA string.
        """
        # Get our previous bases - parity is always the last one in the block
        previous_base_1: str = previous_bases[0]
        previous_base_parity: str = previous_bases[-1]

        # Convert to trits
        trit_1: str = Goldman().dna_to_trit(payload_strand_1, previous_base_1)

        # The virtual strand has to be all 0 trits, not all of one base - that would be a homopolymer in DNA
        trit_2: str = "0" * len(trit_1)
        if payload_strand_2 is not None:
            trit_2 = Goldman().dna_to_trit(payload_strand_2, previous_bases[1])

        # Perform our XOR (-(strand 1 + strand 2) mod 3) and convert back to DNA
        parity_trit: str = self._XOR_mod(trit_1, trit_2, 3)
        parity_strand: str = Goldman().trit_to_dna(parity_trit, previous_base_parity)

        return parity_strand


    def _payloads_complete(self, missing_indices: list[int], num_payload_strands: int) -> bool:
        """
        Checks if all the payload strands in a block are present. Covers nothing missing and only the parity strand
        missing, since parity holds no data of its own.

        Args:
            missing_indices: The indices of the missing strands in the block.
            num_payload_strands: The number of payload strands in the block. 2 for a full block, 1 for the last odd one.

        Returns:
            True if no payload strand is missing, False otherwise.
        """
        for index in missing_indices:
            # Payload strands come first in the block, so any index before the parity is a missing payload strand
            if index < num_payload_strands:
                return False

        return True


    def _too_many_missing(self, missing_indices: list[int]) -> bool:
        """
        Checks if a block has more missing strands than we can recover. Each parity strand can only fill in one missing
        strand, so more missing than parity strands (2 of the 3, or both in the last odd block) can't be recovered.

        Args:
            missing_indices: The indices of the missing strands in the block.

        Returns:
            True if there are more missing strands than parity strands, False otherwise.
        """
        return len(missing_indices) > self.parity_per_block


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def add_redundancy(self, previous_sequences: list[str], payload_sequences: list[str]) -> list[str]:
        """
        Adds XOR parity strands to the payload sequences. Each parity strand is the XOR of two payload strands. We are
        doing XOR with mod 3 rather than mod 2 because we are using Goldman encoding to preserve the homopolymer-free
        property. 

        Args:
            previous_sequences: The previous bases to be used for the columns.
            payload_sequences: The payload sequences to be put XOR parity on.

        Returns:
            A list of parity strands, where each parity strand is the XOR of two payload strands.
        
        Raises:
            ValueError: If previous_sequences or payload_sequences is empty/None.
            OuterCodeError: If there are not enough payload sequences to create parity strands.
            MistmatchSegments: If there are not enough previous sequences for the payload and parity strands.
            PayloadLengthError: If the payload length is 0.
            NotEnoughPreviousSegments: If there are not enough previous sequences to create the required number of 
                parity strands.
        """
        self._validate_segments(previous_sequences, payload_sequences)
        # Ensure we have a positive payload length, otherwise we can't create parity strands
        self._retrieve_payload_length(payload_sequences)

        # Every data and parity strand needs its own reserved index. Parity strand will have its own base from our 
        #   assumed indicies, so ensure we have enough of them to create the parity strand with the previous base.
        if len(previous_sequences) < self.calculate_total_strands(len(payload_sequences)):
            raise NotEnoughPreviousSegments(
                message="Not enough previous sequences for the payload and parity strands"
            )

        parity_strands: list[str] = []

        # Go through each pair of payload strands and create a parity strand for them. Previous sequences hold every
        #   index laid out in blocks of [payload, payload, parity], so the blocks line up with each pair
        for previous_block, payload_block in zip(*self._segment_to_blocks(previous_sequences, payload_sequences)):
            # Get our first payload strand and its previous base for the column
            payload_strand_1: str = payload_block[0]
            # If we dont have a second payload strand, we will XOR with a virtual strand of all 0 trit
            payload_strand_2: str | None = None
            if len(payload_block) == self.payload_per_block:
                payload_strand_2 = payload_block[1]
            
            # Take all the previous bases from our (presumed) indicies
            previous_bases: list[str] = []
            for sequence in previous_block[:len(payload_block) + self.parity_per_block]:
                previous_bases.append(sequence[-1])
    
            # A missing second strand is a virtual strand of all 0 trits. It doesn't show up in our actual payload
            #   strands, but the last payload still gets a parity strand for it
            parity_strand: str = self._get_parity_strand(
                payload_strand_1=payload_strand_1,
                payload_strand_2=payload_strand_2,
                previous_bases=previous_bases
            )

            parity_strands.append(parity_strand)

        return parity_strands


    @override
    def perform_correction(self, 
                           previous_sequences: list[str], 
                           encoded_sequences: list[str | None]) -> list[str | None]:
        """
        Performs XOR error correction on the encoded sequences. Each parity strand is the XOR of two payload strands. If
        one of the payload strands is missing, we can recover it using the parity strand and the other payload strand.

        Args:
            previous_sequences: The complete list of index sequences. These provide the preceding base needed to
                convert each strand to and from trits. Assumed to be sorted. We regenerate any missing one, so no
                worries about them being none from erasure
            encoded_sequences: The list of encoded sequences we retrieve after decoding, including the parity strands.
                We assume that sequences that are None are the ones marked for erasure. The sequences that are not None
                are assumed to be correct and will be used to correct the others. Also assumes that the sequences are in
                the same order as they were originally encoded (that is sorted by index).

        Returns:
            The list of corrected sequences without the parity strands. A block with more than one missing strand can't
            be recovered, so its missing payload strands stay None.

        Raises:
            ValueError: If previous_sequences or encoded_sequences is empty/None.
            PayloadLengthError: If the payload length is 0.
            MistmatchSegments: If previous_sequences and encoded_sequences are not the same length.
        """
        self._validate_segments(previous_sequences, encoded_sequences)
        self._retrieve_payload_length(encoded_sequences)

        # Must be same length in this scenario
        if len(previous_sequences) != len(encoded_sequences):
            raise MismatchSegments(
                message="Previous and encoded sequences must be the same length for performing correction"
            )

        # Some sequences we may not be able to correct. Returning None means those sequences are forever lost
        corrected_sequences: list[str | None] = []

        # Go through each block of [payload, payload, parity]. With an odd number of payload strands the last block is
        #   only [payload, parity], the second payload being the virtual strand of all 0 trits
        for block_start in range(0, len(encoded_sequences), self.block_strands):
            # Get our blocks
            previous_block: list[str] = previous_sequences[block_start:block_start + self.block_strands]
            payload_block: list[str | None] = encoded_sequences[block_start:block_start + self.block_strands]

            # Parity is always the last strand in the block, everything before it is payload
            num_payload_strands: int = len(payload_block) - self.parity_per_block
            corrected_block: list[str | None] = payload_block[:num_payload_strands]

            # Find the indices of any missing sequences
            missing_indices: list[int] = []
            for index, payload_sequence in enumerate(payload_block):
                if payload_sequence is None:
                    missing_indices.append(index)

            # Nothing to recover if all our payload strands are here, and we can't recover with more missing than parity
            #   strands (the missing payload strands stay None). Either way we keep the payload strands as they are
            if self._payloads_complete(missing_indices, num_payload_strands) or self._too_many_missing(missing_indices):
                corrected_sequences.extend(corrected_block)
                continue

            # We have exactly one missing sequence, and it is a payload strand - the one we want to recover
            missing_index: int = missing_indices[0]

            # Get the strands we do have and their previous bases, in block order. In the last odd block this is only
            #   the parity strand, so the virtual strand fills in as the second strand
            present_strands: list[str] = []
            previous_bases: list[str] = []
            for previous_sequence, payload_sequence in zip(previous_block, payload_block):
                if payload_sequence is not None:
                    present_strands.append(payload_sequence)
                    previous_bases.append(previous_sequence[-1])

            # The missing strand's previous base goes last (our calculated new parity)
            previous_bases.append(previous_block[missing_index][-1])

            # Get our present payload strands, the second one may be None if we are in the last odd block
            payload_strand_1: str = present_strands[0]
            payload_strand_2: str | None = None
            if len(present_strands) == self.payload_per_block:
                payload_strand_2 = present_strands[1]

            # Same as making the parity strand: -(sum of the other two) mod 3 gives back whichever strand is missing.
            #   The missing strand must be a payload strand, so we put it in the correct block
            corrected_block[missing_index] = self._get_parity_strand(
                payload_strand_1=payload_strand_1,
                payload_strand_2=payload_strand_2,
                previous_bases=previous_bases
            )

            corrected_sequences.extend(corrected_block)

        return corrected_sequences

    
    @override
    def calculate_total_strands(self, payload_strands: int) -> int:
        """
        Return the payload and parity strands for the payload strand count.

        Args:
            payload_strands: The number of payload strands to be encoded with XOR parity.

        Returns:
            The total number of strands after adding XOR parity.

        Raises:
            ValueError: If payload_strands is not positive.
        """
        if payload_strands <= 0:
            raise ValueError("Payload strand count must be positive.")

        # Every 2 payload strands get 1 parity strand. An odd last payload strand is paired with the virtual strand of
        #   all 0 trits, so it still gets its own parity strand - rounding the parity count up
        self.num_blocks = (payload_strands + 1) // 2

        return payload_strands + self.num_blocks



class DirectDNAXOR(XOR):
    """
    Implements a simple XOR-based error correction scheme on the DNA itself. For encodings that allow homopolymers we
    cannot convert "GGG" from Goldman trits and back as the rotation scheme doesnt allow for it. Instead each base is
    directly a base-4 digit (quat) and each parity strand is the XOR of two payload strands (mod 4). The blocks,
    correction and odd last block all work the same as XOR, only the conversion between DNA and digits changes -
    previous bases are not needed. The parity strands can have homopolymers.
    """

    def __init__(self):
        """
        Initializes the DirectDNAXOR, same blocks as XOR: 2 payload strands and 1 parity strand.
        """
        super().__init__()


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _dna_to_quats(self, dna_sequence: str) -> str:
        """
        Converts DNA bases directly into a quat string, each base being its own base-4 digit.

        Args:
            dna_sequence: The DNA bases to be converted.

        Returns:
            The quat string of the DNA bases.
        """
        quats: str = ""
        for base in dna_sequence:
            quats += str(OuterCodes.DNA_TO_DIGIT[base])

        return quats


    def _quats_to_dna(self, quats: str) -> str:
        """
        Converts a quat string directly back into DNA bases.

        Args:
            quats: The quat string to be converted.

        Returns:
            The DNA bases of the quat string.
        """
        dna_sequence: str = ""
        for quat in quats:
            dna_sequence += OuterCodes.DIGIT_TO_DNA[int(quat)]

        return dna_sequence


    @override
    def _get_parity_strand(self,
                           payload_strand_1: str,
                           payload_strand_2: str | None,
                           *args, **kwargs) -> str:
        """
        Gets the parity strand for two payload strands. Since strand 1 + strand 2 + parity = 0 (mod 4), this is also how
        we recover any missing strand: pass in the two strands we have.

        Args:
            payload_strand_1: The first payload strand.
            payload_strand_2: The second payload strand. None if there isn't one (odd number of payload strands), we
              then XOR with a virtual strand of all 0 quats.
            *args and **kwargs, used to maintain polymorphism with the XOR class, but not used in this method.

        Returns:
            The parity strand as a DNA string.
        """
        quat_1: str = self._dna_to_quats(payload_strand_1)

        # The virtual strand is all 0 quats (all As) - homopolymers are fine here
        quat_2: str = "0" * len(quat_1)
        if payload_strand_2 is not None:
            quat_2 = self._dna_to_quats(payload_strand_2)

        # Perform our XOR (-(strand 1 + strand 2) mod 4) and convert back to DNA
        parity_quat: str = self._XOR_mod(quat_1, quat_2, 4)

        return self._quats_to_dna(parity_quat)



class RSOuter(OuterCodes):
    """
    Reed-Solomon outer code for Goldman-encoded DNA.

    Goldman encoding represents each DNA base as a transition from the
    previous base, so each six-base column is represented by six trits.
    """


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

        super().__init__()

        self.redundancy_percentage: float = redundancy_percentage

        # Redundancy is measured relative to data strands, not the complete codeword.
        self.payload_per_block: int = int(self.MAX_BLOCK_STRANDS / (1 + redundancy_percentage))
        self.parity_per_block: int = self.MAX_BLOCK_STRANDS - self.payload_per_block
        self.block_strands: int = self.MAX_BLOCK_STRANDS
        self.rs: galois.ReedSolomon = self._get_rs(self.payload_per_block, self.parity_per_block)

        self.num_blocks: int = 0


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # How many bases we define to be a column
    COLUMN_LENGTH: Final[int] = 6

    GF_ORDER: int = 3**6
    GF: type[galois.FieldArray] = galois.GF(GF_ORDER)
    MAX_BLOCK_STRANDS: int = GF_ORDER - 1

    # Cache for RS codes when we have the same parity_strands value. Takes a while to initialize.
    _RS_CACHE: ClassVar[dict[tuple[int, int], galois.ReedSolomon]] = {}


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    @staticmethod
    def align_payload_length(payload_length: int) -> int:
        """
        Rounds the payload length down to a multiple of COLUMN_LENGTH. Every column must be COLUMN_LENGTH bases, as
        each parity symbol needs all COLUMN_LENGTH bases and a parity strand only has as many bases as the payload.

        Args:
            payload_length: The number of bases available for the payload.

        Returns:
            The payload length rounded down to a multiple of COLUMN_LENGTH.
        """
        return payload_length - (payload_length % RSOuter.COLUMN_LENGTH)


    @staticmethod
    def trits_to_field_symbol(trits: list[int]) -> int:
        """
        Converts six trits into one GF(3^6) symbol (basically trit to decimal with a list)

        Args:
            trits: The trits that make up a field symbol.

        Returns:
            The integer representation of the field symbol.
        """

        return int("".join(str(trit) for trit in trits), 3)


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#
    

    def _get_rs(self, payload_count: int, parity_count: int) -> galois.ReedSolomon:
        """
        Returns the Reed-Solomon code for a block with the given number of parity strands. Full blocks use
        (MAX_BLOCK_STRANDS, payload_per_block), the last block may be smaller with its own reduced parity count.

        Galois can only build codes where n divides GF_ORDER - 1, so a smaller block uses the full length code with
        n - k = parity_count, shortened. Galois shortens automatically when given fewer than k message symbols (or a
        codeword shorter than n), so payload_count only needs to fit.

        We cache the Reed-Solomon code for each parity count to avoid reinitializing it multiple times. It takes so
        long (usually seconds) so we are just going to lazy load it when we need it.

        Args:
            payload_count: The number of payload strands in the block.
            parity_count: The number of parity strands (n - k) in the block.

        Returns:
            The Reed-Solomon code for the block.

        Raises:
            OuterCodeError: If the payload strands do not fit in the block with the given parity.
        """
        # Can't have more payload strands than the maximum block size minus the parity strands
        k: int = self.MAX_BLOCK_STRANDS - parity_count
        if payload_count > k:
            raise OuterCodeError(
                message=f"{payload_count} payload strands do not fit an RS block with {parity_count} parity strands"
            )

        # Add code to cache if doesn't exist yet
        cache_key: tuple[int, int] = (self.GF_ORDER, parity_count)
        if cache_key not in self._RS_CACHE:
            self._RS_CACHE[cache_key] = galois.ReedSolomon(
                n=self.MAX_BLOCK_STRANDS, 
                k=k, 
                field=self.GF
            )

        return self._RS_CACHE[cache_key]


    def _parity_for_block(self, payload_count: int) -> int:
        """
        Returns the number of parity strands for a block with the given number of payload strands. Full blocks get
        parity_per_block, a smaller (last) block gets parity based on the redundancy percentage.

        Args:
            payload_count: The number of payload strands in the block.

        Returns:
            The number of parity strands for the block.
        """
        # When payload is at least our max payload per block, use full parity
        if payload_count >= self.payload_per_block:
            return self.parity_per_block

        # Otherwise calculate parity based on the redundancy and round up to ensure we have enough parity strands
        return math.ceil(payload_count * self.redundancy_percentage)


    def _payload_for_block(self, block_strands: int) -> int:
        """
        Inverse of _parity_for_block. Given the total strands in a block (payload + parity), returns how many of them
        are payload strands. Payload strands are always first in the block, followed by the parity strands.

        Args:
            block_strands: The total number of strands (payload + parity) in the block.

        Returns:
            The number of payload strands in the block.

        Raises:
            OuterCodeError: If no payload count produces the given block size.
        """
        if block_strands >= self.MAX_BLOCK_STRANDS:
            return self.payload_per_block

        # The payload + parity is strictly increasing in payload, so there is at most one match
        for payload_count in range(block_strands, 0, -1):
            if payload_count + self._parity_for_block(payload_count) == block_strands:
                return payload_count

        raise OuterCodeError(
            message=f"No payload count produces an RS block of {block_strands} strands"
        )


    def _dna_to_field_symbol(self, dna_sequence: str, previous_base: str) -> int:
        """
        Converts one Goldman-encoded DNA column into a GF(3^6) symbol.

        Args:
            dna_sequence: The DNA sequence to be converted into a field symbol.
            previous_base: The previous base to be used for the DNA sequence.

        Returns:
            The GF(3^6) symbol corresponding to the DNA sequence.
        """
        trits: list[int] = []
        for trit in Goldman().dna_to_trit(dna_sequence, previous_base):
            trits.append(int(trit))

        return self.trits_to_field_symbol(trits)


    def _field_symbol_to_dna(self, symbol: int, previous_base: str) -> str:
        """
        Converts one GF(3^6) parity symbol back into Goldman-encoded DNA.

        Args:
            symbol: The GF(3^6) parity symbol to be converted back into DNA.
            previous_base: The previous base to be used for the DNA sequence.

        Returns:
            The DNA sequence corresponding to the GF(3^6) parity symbol.

        Raises:
            EncodingError: If the Goldman mapping fails to convert trits to DNA bases - happens when there's a 
                homopolymer.
        """
        trits: str = GoldmanIndexer.decimal_to_trit(
            decimal=symbol,
            width=RSOuter.COLUMN_LENGTH
        )
        dna_sequence: str = Goldman().trit_to_dna(trits, previous_base)

        return dna_sequence


    def _retrieve_columns(self,
                          previous_segments: list[str], 
                          payload_segments: list[str] | list[str | None],
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

        columns: list[list[galois.FieldArray]] = []

        # Each six-base DNA slice becomes one GF(3^6) symbol for RS encoding.
        for index in range(0, payload_length, RSOuter.COLUMN_LENGTH):
            column: list[galois.FieldArray] = []

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


    def _get_erasure_mask(self, payload_block: list[str | None], ) -> list[bool]:
        """
        Creates an erasure mask for the given sequences. True indicates an erasure (None), and False indicates a valid 
        sequence.

        Args:
            payload_block: The list of payload segments.

        Returns:
            A list of booleans representing the erasure mask.
        """
        # Get the erasure mask - everywhere they are are 
        erasure_mask: list[bool] = []
        for sequence in payload_block:
            if sequence is None:
                erasure_mask.append(True)
                continue

            erasure_mask.append(False)

        return erasure_mask


    def _fill_erasures(self,
                       rs: galois.ReedSolomon,
                       codewords: galois.FieldArray,
                       erasure_mask: list[bool]) -> galois.FieldArray | None:
        """
        Fills in the erased symbols of every column in a block at once. Galois' decoder is very very slow whenever
        there is an erasure that is not known (all are known to us), but when we only have erasures (the other strands 
        passed the checksum) we can solve for the missing symbols directly.

        Args:
            rs: The Reed-Solomon code for the block.
            codewords: The (columns, block_strands) symbols of the block, erased symbols can be anything.
            erasure_mask: True for each strand that is erased.

        Returns:
            The codewords with the erased symbols filled in, or None if there are more erasures than parity strands.
        """
        # Positions (strands) of the erased and known symbols - needs to be numpy arrays for indexing
        erased: NDArray[np.int_] = np.flatnonzero(erasure_mask)
        known: NDArray[np.int_] = np.flatnonzero(np.logical_not(erasure_mask))

        # Nothing to fill
        if erased.size == 0:
            return codewords

        # Each parity strand gives one equation, so we can't solve for more erasures than we have parity
        if erased.size > rs.n - rs.k:
            return None

        # Galois computes H (the parity-check matrix) once per RS code. A shortened (thelast) block drops its leading 
        #   message positions, so it uses the last block_strands columns of H. Keeping one row/equation per erasure
        block_strands: int = codewords.shape[1]
        parity_check: galois.FieldArray = self.GF(rs.H[:erased.size, -block_strands:])

        # Each row of H is one check (equation) that a valid codeword passes: we multiply every strands symbol by that
        #   rows number for the strand, add them all up, and the total is will be 0. For example, with 4 strands where
        #   strand 2 is erased (x) and one check row [h0, h1, h2, h3]:
        #       h0 * s0 + h1 * s1 + h2 * x + h3 * s3 = 0
        #   We know s0, s1 and s3, so we can add up their part now and move it to the other side:
        #       h2*x = -(h0 * s0 + h1 * s1 + h3 * s3)
        #   This computes that righ hand side, the "-(h0 * s0 + h1 * s1 + h3 * s3)", for every check row and every 
        #   column at once. matmul does the multiply and add for us (in GF(3^6) math, not normal integer math). The
        #   codewords are transposed because matmul pairs each row of the first matrix with each column of the second
        #   matrix, so the strands of each codeword need to run down a column to line up with the strands in H's rows.
        #   The result is a (erasures, columns) table, one total per check per codeword column.
        right_side: galois.FieldArray = -np.matmul(parity_check[:, known], codewords[:, known].T)

        # Now we have one equation per check with only the erased symbols unknown, e.g. with 2 erasures (x, y):
        #       a * x + b * y = r0
        #       c * x + d * y = r1
        #   where a, b, c, d are H's numbers for the erased strands and r0, r1 come from right_side. Same number of
        #   equations as unknowns, so like solving a system of equations in algebra, there is exactly one answer, and
        #   that answer is the missing symbols. solve does this for every column at once, giving a (erasures, columns)
        #   table of the erased symbols. Wrapped in self.GF to make sure the result stays a field array.
        erased_symbols: galois.FieldArray = self.GF(np.linalg.solve(parity_check[:, erased], right_side))

        # Put the found symbols back into the erased positions (transposed back to one row per codeword column)
        filled: galois.FieldArray = codewords.copy()
        filled[:, erased] = erased_symbols.T

        return filled


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def add_redundancy(self, previous_sequences: list[str], payload_sequences: list[str]) -> list[str]:
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

        # Every data and parity strand needs its own reserved index. Different from previous sequences being smaller,
        #   we need to know that we have enough sequences for all pairty strands. Different from the check of parity 
        #   strands being less than the data strands - this is do we have enough for adding parity?
        if len(previous_sequences) < self.calculate_total_strands(len(payload_sequences)):
            raise NotEnoughPreviousSegments(
                message="Not enough previous sequences for the payload and parity strands"
            )

        parity_strands: list[str] = []

        previous_blocks, payload_blocks = self._segment_to_blocks(previous_sequences, payload_sequences)
        for previous_block, payload_block in zip(previous_blocks, payload_blocks):
            # The last block may not be full, so it gets parity based on the redundancy percentage and its own
            #   shortened RS code. Its parity strands sit directly after its payload strands, not at payload_per_block
            payload_count: int = len(payload_block)
            parity_per_block: int = self._parity_for_block(payload_count)
            rs: galois.ReedSolomon = self._get_rs(payload_count, parity_per_block)

            # Get our columns for this block, which is a all the GF symbols - if we got none we got a problem
            columns: list[list[galois.FieldArray]] = self._retrieve_columns(
                previous_segments=previous_block, 
                payload_segments=payload_block,
                error_on_none=True
            )

            # Each output entry is one parity strand assembled column by column, so we need to initalize them all first
            #   in order to ensure we can append to them in correct order
            parity_block: list[str] = [""] * parity_per_block

            # We need to keep track of the last base - currently its all the indexes (we assume)
            parity_previous_bases: list[str] = []
            for index in range(parity_per_block):
                parity_previous_bases.append(previous_block[payload_count + index][-1])
        
            for column in columns:
                # RS returns one field symbol for each parity strand for this column
                parity_column: galois.FieldArray = rs.encode(self.GF(column), output="parity")

                # We need to convert each parity symbol back into DNA bases and add them to corresponding parity strand
                for parity_index in range(parity_per_block):
                    # Convert to a usable int based on the field
                    parity_sequence: str = self._field_symbol_to_dna(
                        symbol=int(parity_column[parity_index]),
                        previous_base=parity_previous_bases[parity_index]
                    )

                    parity_block[parity_index] += parity_sequence

                    # Goldman needs the last emitted base for the next column. Direct DNA conversion does not use this
                    if parity_sequence:
                        parity_previous_bases[parity_index] = parity_sequence[-1]

            # Add the parity block to the list of all parity strands - keeps our order of blocks correct too
            parity_strands.extend(parity_block)

        return parity_strands


    def perform_correction(self,
                           previous_sequences: list[str], 
                           encoded_sequences: list[str | None] ) -> list[str | None]:
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
        corrected_sequences: list[str | None] = []

        # Go through each block and correct one at at time, ending block (likely not full) will be handled the same
        for block_start in range(0, len(encoded_sequences), self.MAX_BLOCK_STRANDS):
            # Get the block of previous sequences and encoded sequences for this RS block
            previous_block: list[str] = previous_sequences[block_start:block_start + self.MAX_BLOCK_STRANDS]
            payload_block: list[str | None] = encoded_sequences[block_start:block_start + self.MAX_BLOCK_STRANDS]

            erasure_mask: list[bool] = self._get_erasure_mask(payload_block)

            # Get the columns for this block, which is a all the GF symbols
            columns: list[list[galois.FieldArray]] = self._retrieve_columns(
                previous_segments=previous_block, 
                payload_segments=payload_block,
                error_on_none=False
            )

            # Our last block may not be full. The block here holds payload + parity, so recover how many are payload
            #   the same way add_redundancy sized it, and use the matching shortened RS code
            data_count: int = self._payload_for_block(len(payload_block))
            parity_per_block: int = len(payload_block) - data_count
            rs: galois.ReedSolomon = self._get_rs(data_count, parity_per_block)
            
            corrected_block: list[str] = [""] * data_count
            block_failed: bool = False

            # Fill the erasures of every column at once. The strands that aren't None passed the checksum, so this is
            #   almost always enough
            filled: galois.FieldArray | None = self._fill_erasures(
                rs=rs, 
                codewords=self.GF(np.array(columns)), 
                erasure_mask=erasure_mask
            )

            for column_index, column in enumerate(columns):
                # Take the filled column if it is a valid codeword
                corrected_column: galois.FieldArray
                if filled is not None and not rs.detect(filled[column_index]):
                    corrected_column = self.GF(filled[column_index][:data_count])

                # Couldn't fill it or the filled column is not a valid codeword, so we have to decode it. 
                #   This is very slow, but we have no choice
                else:
                    errors: NDArray[np.int_] | int
                    corrected_column, errors = rs.decode(
                        codeword=self.GF(column),
                        erasures=np.array(erasure_mask),
                        output="message",
                        errors=True
                    )

                    # Galois reports an undecodable column with errors == -1. Goldman decoding is stateful across
                    #   columns, so no later column can make a complete DNA strand valid after this point
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
                        previous_base=previous_base
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
        full_blocks: int = payload_strands // self.payload_per_block
        remainder: int = payload_strands % self.payload_per_block
        self.num_blocks = full_blocks
        if remainder > 0:
            self.num_blocks = full_blocks + 1
            
        # Blocks are packed back to back, so a smaller last block only takes its payload plus its reduced parity
        total_strands: int = full_blocks * self.MAX_BLOCK_STRANDS
        if remainder:
            total_strands += remainder + self._parity_for_block(remainder)

        return total_strands



class DirectDNARSOuter(RSOuter):
    """
    Reed-Solomon outer code for encodings that allow homopolymers. We cannot convert "GGG" from Goldman trits and back
    as the rotation scheme doesnt allow for it. This DirectDNARSOuter class allows for direct DNA to GF(4^6) conversion
    without using Goldman's rotation scheme. Where we basically assigning each digit a base-4 value and converting the
    them with six nt's (too keep consitient with the RSOuter class) into a GF(4^6) symbol.
    """


    def __init__(self, redundancy_percentage: float = 0.1):
        """
        Initializes the DirectDNARSOuter with a specified redundancy percentage.

        Args:
            redundancy_percentage: The percentage of redundancy to be used in the outer code.
        """
        super().__init__(redundancy_percentage=redundancy_percentage)


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Columns stays the same

    GF_ORDER: int = 4**6
    GF: type[galois.FieldArray] = galois.GF(GF_ORDER)

    # Use the maximum Reed-Solomon codeword length supported by GF(4096)
    MAX_BLOCK_STRANDS: int = GF_ORDER - 1


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def _dna_to_field_symbol(self, dna_sequence: str, *args, **kwargs) -> int:
        """
        Converts six arbitrary DNA bases into one GF(4^6) symbol.

        Args:
            dna_sequence: The DNA bases to be converted into a GF(4^6) symbol.
            *args and **kwargs, used to maintain compatibility with the RSOuter class, but not used here.

        Returns:
            The integer representation of the GF(4^6) symbol.
        """
        symbol: int = 0
        for base in dna_sequence:
            symbol = symbol * 4 + OuterCodes.DNA_TO_DIGIT[base]

        return symbol


    @override
    def _field_symbol_to_dna(self, symbol: int, *args, **kwargs) -> str:
        """
        Converts one GF(4^6) data symbol back into DNA bases. Essientaly 'quits' or qubits? to decimal. Only used for
        data strands, parity strands go through _symbols_to_parity_dna.

        Args:
            symbol: The GF(4^6) symbol to be converted back into DNA.
            *args and **kwargs, used to maintain compatibility

        Returns:
            The DNA sequence corresponding to the GF(4^6) symbol.
        """
        bases: str = ""
        for _ in range(self.COLUMN_LENGTH):
            digit: int = symbol % 4
            symbol //= 4
            # Prepend each least-significant base so the final sequence stays in its original left-to-right order.
            bases = OuterCodes.DIGIT_TO_DNA[digit] + bases

        return bases