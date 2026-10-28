from abc import abstractmethod
from typing import ClassVar, Final, List, override

# Can't use reedsolo because it doesn't support GF(3^6) and has a max of 255 symbols. We need to encode each column of
#   the oligos as 6-bit symbols, which gives us 3^6 = 729 symbols. This way we can retain the same fixed-length oligo
#   and use Goldman's rotation scheme to avoid homopolymers. It is much slower than reedsolo, but it the best option.
import galois

from Encode.Helpers.Errors import OuterCodeError
from Encode.Indexer import GoldmanIndexer
from Encode.Methods.HuffmanCoding import Goldman

class OuterCodes():
    def __init__(self):
        pass


    @abstractmethod
    def calculate_total_strands(self, payload_strands: int) -> int:
        raise NotImplementedError()


    @abstractmethod
    def select_payload_indices(self, indices: List[str]) -> List[str]:
        raise NotImplementedError()
    

    @abstractmethod
    def get_addtional_strands(self) -> int:
        raise NotImplementedError()


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

        # Redundancy is measured relative to data strands, not the complete codeword.
        self.data_capacity: int = int(self.MAX_BLOCK_STRANDS / (1 + redundancy_percentage))
        self.parity_per_block: int = self.MAX_BLOCK_STRANDS - self.data_capacity

        self.num_blocks: int = 0

        # We cache the Reed-Solomon code for each parity_strands value to avoid reinitializing it multiple times. It 
        #   takes so long (usually seconds) so we are just going to lazy load it when we need it. Imagine multiple runs
        #   of the same parity_strands value, we don't want to reinitialize it every time.
        cache_key: tuple[int, int] = (self.GF_ORDER, self.parity_per_block)
        self.rs: galois.ReedSolomon = self._RS_CACHE.get(
            cache_key,
            galois.ReedSolomon(
                n=self.MAX_BLOCK_STRANDS,
                k=self.data_capacity,
                field=self.GF
            )
        )
        if cache_key not in self._RS_CACHE:
            self._RS_CACHE[cache_key] = self.rs


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


    def _validate_segments(self, previous_segments: List[str], payload_segments: List[str]) -> None:
        """
        Validates the previous and payload segments for Reed-Solomon error correction.

        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put RS parity on.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
        """
        if not previous_segments:
            raise ValueError("Previous segments cannot be empty.")
        if not payload_segments:
            raise ValueError("Payload segments cannot be empty.")


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

        # Payload blocks are up to the data_capacity length
        payload_blocks: List[List[str]] = []
        for index in range(0, len(payload_segments), self.data_capacity):
            payload_blocks.append(payload_segments[index:index + self.data_capacity])

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


    def _retrieve_columns(self,
                          previous_segments: List[str], 
                          payload_segments: List[str]) -> List[List[galois.FieldArray]]:
        """
        Retrieves the columns of bases from the previous bases and payload segments.

        Args:
            previous_segments: The previous bases to be used for the columns.
            payload_segments: The payload segments to be put RS parity on.

        Returns:
            A list of columns, where each column is a list of lists containing the previous base and the corresponding 
            payload bases.

        Raises:
            ValueError: If previous_segments or payload_segments is empty/None.
            EncodingError: If the Goldman mapping fails to convert DNA bases to trits - likely to happen when not 
                accounting for the specific spliced indices used for indexing.
        """
        self._validate_segments(previous_segments, payload_segments)

        if len(previous_segments) < len(payload_segments):
            raise ValueError("There must be at least one previous sequence for each payload sequence.")

        payload_length: int = len(payload_segments[0])
        if payload_length == 0:
            raise ValueError("Payload sequences cannot contain empty sequences.")
        if any(len(segment) != payload_length for segment in payload_segments):
            raise ValueError("Payload sequences must all have the same length.")

        columns: List[List[galois.FieldArray]] = []

        # Each six-base DNA slice becomes one GF(3^6) symbol for RS encoding.
        for index in range(0, payload_length, RSOuter.COLUMN_LENGTH):
            column: List[galois.FieldArray] = []

            # Ensure we add proper virtual zero trits for the final column if it is shorter than six bases
            column_length: int = min(RSOuter.COLUMN_LENGTH, payload_length - index)

            # Collect the same six-base position from every data strand.
            for segment_index, segment in enumerate(payload_segments):
                # The first slice starts after the strand's index. Later slices continue from the preceding DNA base in 
                #   the same strand
                previous_base: str
                if index == 0:
                    previous_base = previous_segments[segment_index][-1]
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
            OuterCodeError: If there are not enough previous sequences for the payload and parity strands.
        """
        self._validate_segments(previous_sequences, payload_sequences)

        # Every data and parity strand needs its own reserved index.
        if len(previous_sequences) < self.calculate_total_strands(len(payload_sequences)):
            raise OuterCodeError(
                message="Not enough previous sequences for the payload and parity strands."
            )

        parity_strands: List[str] = []

        previous_blocks, payload_blocks = self._segment_to_blocks(previous_sequences, payload_sequences)
        for index_block, payload_block in zip(previous_blocks, payload_blocks):
            # Get our columns for this block, which is a all the GF symbols
            columns: List[List[galois.FieldArray]] = self._retrieve_columns(index_block, payload_block)

            # Each output entry is one parity strand assembled column by column, so we need to initalize them all first
            #   in order to ensure we can append to them in correct order
            parity_block: List[str] = [""] * self.parity_per_block

            # We need to keep track of the last base - currently its all the indexes (we assume)
            parity_previous_bases: List[str] = []
            for index in range(self.parity_per_block):
                parity_previous_bases.append(index_block[self.data_capacity + index][-1])
        
            for column_index, column in enumerate(columns):
                # Preserve the actual payload length when the final column is shorter than six bases
                column_length: int = min(
                    RSOuter.COLUMN_LENGTH,
                    len(payload_block[0]) - column_index * RSOuter.COLUMN_LENGTH
                )

                # RS returns one field symbol for each parity strand for this column
                parity_column: galois.FieldArray = self.rs.encode(self.GF(column), output="parity")

                # Still uses same parity per block regardlesss of block size
                for parity_index in range(self.parity_per_block):
                    # The concrete outer-code variant converts the parity
                    # field symbol back into DNA.
                    parity_sequence: str = self._field_symbol_to_dna(
                        symbol=int(parity_column[parity_index]),
                        previous_base=parity_previous_bases[parity_index],
                        length=column_length
                    )

                    parity_block[parity_index] += parity_sequence

                    # Goldman needs the last emitted base for the next
                    # column. Direct DNA conversion does not use this state.
                    if parity_sequence:
                        parity_previous_bases[parity_index] = parity_sequence[-1]

            # Add the parity block to the list of all parity strands - keeps our order of blocks correct too
            parity_strands.extend(parity_block)

        return parity_strands

    def preform_correction(self, encoded_sequecnes: List[str | None]) -> List[str | None]:
        """
        Performs Reed-Solomon error correction on the encoded sequences.

        Args:
            encoded_sequences: The list of encoded sequences, we retrieve after decoded. We assume that sequences that 
                are None are the ones marked for erasure. The sequences that are not None are assumed to be correct and 
                will be used to correct the others.

        Returns:
            The list of corrected sequences without the parity strands. Sequences that could not be corrected = None.
        """


        # Placeholder for the actual error correction logic
        return encoded_sequecnes


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

        # Update the number of blocks based on the payload strands and data capacity. This is used to calculate the 
        #   total number of strands after adding parity.
        blocks: int = (payload_strands + self.data_capacity - 1) // self.data_capacity
        self.num_blocks = blocks

        return blocks * self.MAX_BLOCK_STRANDS


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

            # Add the first data_capacity indices from the block to the payload_indices list
            payload_indices += block[:self.data_capacity]
        
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
        Converts one GF(4^6) parity symbol back into DNA bases. Essientaly 'quits'? to decimal

        Args:
            symbol: The GF(4^6) parity symbol to be converted back into DNA.
            length: The length of the DNA sequence to be returned. This is used to ensure that
                the DNA sequence is the same length as the original column.
            *args and **kwargs, used to maintain compatibility

        Returns:
            The DNA sequence corresponding to the GF(4^6) parity symbol.
        """
        bases: str = ""
        for _ in range(self.COLUMN_LENGTH):
            digit: int = symbol % 4
            symbol //= 4
            
            # Prepend each least-significant base so the final sequence stays in its original left-to-right order.
            bases = self.DIGIT_TO_DNA[digit] + bases

        return bases[:length]
