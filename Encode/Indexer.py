"""
Module for enumerative indexing of DNA sequences. This module provides classes and methods to perform indexing on 
oligos, which is needed to identify them for decoding. 
"""

import math
from abc import abstractmethod
from functools import lru_cache
from typing import Any, Final, override

from Encode.Helpers.Errors import EncodingError, MaxRunLengthError, RankError, UnrankError
from Encode.Methods.HuffmanCoding import Goldman, Huffman
from Encode.Methods.Encoding import Encoding


class Indexer:
    """
    Abstract base class for indexing DNA sequences.

    This class provides a framework for creating different types of indexers for DNA sequences.
    """
    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the Indexer the width of the index sequence.

        Args:
            width: The width of the index sequence.
            Args and kwargs are included for polymorphism, but unused
        """
        # Width can be set to -1 if we set it at a later date, but it cannot be negative when used for ranking 
        self.width: int = width

        # Default value, can be overridden in subclasses
        self.max_run_length: int = 1  


    @staticmethod
    def determine_optimal_width(required_capacity: int, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError()


    @abstractmethod
    def calculate_total_valid_indices(self) -> int:
        raise NotImplementedError()


    @abstractmethod
    def rank(self, index_sequence: str, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError()


    @abstractmethod
    def unrank(self, index: int, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError()



class DNAMapIndexer(Indexer):
    """
    Standard DNA mapping indexer that uses a simple 2-bit representation for each base (A, C, G, T). This indexer does 
    not have a maximum run length constraint, meaning it can generate index sequences with homopolymer runs of any 
    length.
    """

    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the DNAMapIndexer with the width of the index sequence.
        
        Args:
            num_oligos: The number of oligos to index.
            width: The width of the index sequence. 
        """
        super().__init__(width)

        # The DNAMapIndexer does not have a maximum run length constraint
        self.max_run_length = -1


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    @staticmethod
    def determine_optimal_width(required_capacity: int, *args, **kwargs) -> int:
        """
        Determines the minimum width of the index sequence required to uniquely identify a given number of oligos.

        Parameters:
            required_capacity: The number of oligos to index (or just just numbers to support in general).
            args and kwargs are included for polymorphism, but unused

        Returns:
            The optimal width of the index sequence.

        Raises:
            ValueError: If the required capacity is not a positive integer.
        """
        # Capture error if required_capacity is less than or equal to 0
        if required_capacity <= 0:
            raise ValueError("Required capacity must be a positive integer.")

        # Each 2 bits represents 1 base, so we need to calculate the number of bits required to represent the number of oligos
        bits_required: int = math.ceil(math.log2(required_capacity))

        # Since each base is represented by 2 bits, we need to round up to the nearest even number of bits
        if bits_required % 2 != 0:
            bits_required += 1

        # The width is the number of bases, which is the number of bits divided by 2
        width: int = bits_required // 2

        # Ensure at least a width of 1 for a single base
        if width == 0:
            width = 1  

        return width


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def unrank(self, index: int, *args: Any, **kwargs: Any) -> str:
        """
        Generates the index sequence corresponding to a given rank based on the DNA mapping indexing scheme.

        Args:
            index: The rank of the index sequence to generate.
            Args and kwargs are included for polymorphism, but unused
            
        Returns:
            The index sequence corresponding to the given rank.

        Raises:
            ValueError: If the index is negative or exceeds the maximum value for the given width.
            UnrankError: If the generated index sequence length does not match the expected width.
        """
        if self.width <= 0:
            raise ValueError("Width must be a positive integer.")

        # Convert the decimal index to a binary string with correct leading zeros (need x2 as each base is 2 bits)
        binary_string: str = f"{index:0{self.width * 2}b}"
        if len(binary_string) > self.width * 2:
            raise UnrankError(
                message="Could not unrank index sequence!",      
                index=index, 
                capacity=self.calculate_total_valid_indices()
            )

        # Map the binary string to DNA bases using the DNAMap scheme
        unrank: str = ""
        for index in range(0, len(binary_string), 2):
            unrank += Encoding.DNA_MAP[binary_string[index:index + 2]]

        return unrank
    

    @override
    def rank(self, index_sequence: str, *args: Any, **kwargs: Any) -> int:
        """
        Calculates the rank of a given index sequence based on the DNA mapping indexing scheme.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            Args and kwargs are included for polymorphism, but unused

        Returns:
            The rank (index) of the index sequence.

        Raises:
            RankError: If the index sequence length does not match the expected width.
            KeyError: If the index sequence contains invalid characters not in the DNA mapping.
        """
        if len(index_sequence) != self.width:
            raise RankError(
                message="Could not rank index sequence",
                index_sequence=index_sequence,
                width=self.width
            )

        # Convert the index sequence to a binary string
        binary_string: str = ""
        for base in index_sequence:
            binary_string += Encoding.REVERSE_DNA_MAP[base]


        if binary_string == "" or not all(bit in Encoding.BINARY_SYMBOLS for bit in binary_string):
            raise RankError(
                message="Could not rank index sequence",
                index_sequence=index_sequence,
                width=self.width
            )

        # Convert to decimal
        rank: int = int(binary_string, 2)

        return rank


    @override
    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences for the DNAMap indexing scheme.

        Returns:
            The total number of valid index sequences.
        """
        return 4 ** self.width



class GoldmanIndexer(Indexer):
    """
    An indexer that uses the Goldman indexing scheme. This scheme is designed to avoid homopolymer runs in DNA 
    sequences, which can cause issues in sequencing and synthesis.
    """
    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the GoldmanIndexer with the number of oligos and the width of the index sequence.
        
        Args:
            num_oligos: The number of oligos to index.
            width: The width of the index sequence. If None, it will be determined based on the number of oligos.
        """
        super().__init__(width)
        # Goldman indexing does not allow homopolymer runs longer than 1
        self.max_run_length: int = 1  


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#
    

    @staticmethod
    def decimal_to_trit(decimal: int, width: int) -> str:
        """
        Converts a decimal to a trit representation.

        Args:
            decimal: The decimal to convert.

        Returns:
            The trit representation of the decimal.

        Raises:
            ValueError: If the decimal is negative.
        """
        if decimal < 0:
            raise ValueError("Index must be a non-negative integer.")

        trit: str = ""
        while decimal > 0:
            trit = str(decimal % 3) + trit
            decimal //= 3

        zeros: str = ""
        if len(trit) < width:
            # Add leading zeros to make representation same as width
            zeros = "0" * (width - len(trit))

        return zeros + trit


    @staticmethod
    def trit_to_decimal(trit: str) -> int:
        """
        Converts a trit representation to a decimal index.

        Args:
            trit: The trit representation to convert.

        Returns:
            The decimal index corresponding to the trit representation.

        Raises:
            ValueError: If a digit in trit cannot be caust to an int.
        """
        decimal: int = 0
        for digit in trit:
            decimal = decimal * 3 + int(digit)
        return decimal

    
    @override
    @staticmethod
    def determine_optimal_width(required_capacity: int, *args, **kwargs) -> int:
        """
        Determines the minimum width of the index sequence required to uniquely identify a given number of oligos.

        Parameters:
            required_capacity: The number of oligos to index (or just just numbers to support in general).
            args and kwargs are included for polymorphism, but unused

        Returns:
            The optimal width of the index sequence.

        Raises:
            ValueError: If the required capacity is not a positive integer.
        """
        if required_capacity <= 0:
            raise ValueError("Required capacity must be a positive integer.")

        # We can easily calculate the optimal width for the Goldman indexer by using the formula:
        #   width = ceil(log3(required_capacity))
        width: int = math.ceil(math.log(required_capacity, 3))
        return width


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences that can be generated given the before and after 
        sequences, using worst case scenario for homopolymer runs.

        Returns:
            The total number of valid index sequences.

        Raises:
            ValueError: If the width is not a positive integer.
        """
        if self.width <= 0:
            raise ValueError("Width must be a positive integer.")

        # For Goldman indexing, the total number of valid indices is simply 3^width, since each position in the index
        #   sequence can be one of three bases (A, C, G) and we are not considering homopolymer runs in this calculation.
        total_valid_indices: int = 3 ** self.width
        return total_valid_indices


    @override
    def unrank(self, index: int, before_sequence: str, *args: Any, **kwargs: Any) -> str:
        """
        Generates the index sequence corresponding to a given rank based on the Goldman indexing scheme.

        Args:
            index: The rank of the index sequence to generate.
            before_sequence: The before sequence used in the oligo.
            Args and kwargs are included for polymorphism, but unused

        Returns:
            The index sequence corresponding to the given rank.

        Raises:
            ValueError: If the index is negative or exceeds the maximum value for the given width.
            UnrankError: If the generated index sequence length does not match the expected width.
        """
        # Default to start base if before is not provided
        before_end: str = Huffman.START_BASE  
        if before_sequence:
            before_end = before_sequence[-1] 

        # Convert decimal index to trit representation, then use Goldman mapping to convert to DNA
        trit_representation: str = GoldmanIndexer.decimal_to_trit(index, self.width)

        # Want to that this is actually an indexing unrank error rather than encoding (helps locate)
        try:
            unrank: str = Goldman().trit_to_dna(trit_representation, start_base=before_end)
            return unrank
        except EncodingError as _:
            raise UnrankError(
                message="Could not unrank index sequence",
                index=index,
                capacity=self.calculate_total_valid_indices()
            )


    @override
    def rank(self, index_sequence: str, before_sequence: str, *args: Any, **kwargs: Any) -> int:
        """
        Calculates the rank of a given index sequence based on the Goldman indexing scheme.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            before_sequence: The before sequence used in the oligo.
            after_sequence: The after sequence used in the oligo.
            *args and **kwargs are included for polymorphism, but unused

        Returns:
            The rank of the index sequence.

        Raises:
            ValueError: If cannot convert trit to decimal.
            RankError: If the index sequence cannot be ranked due to encoding issues.
        """
        # Default to start base if before is not provided
        before_end: str = Huffman.START_BASE  
        if before_sequence:
            before_end: str = before_sequence[-1] 

        # Need to know that it is actually an indexing rank error rather than encoding (helps locate)
        try:
            # Convert DNA bases to trits using Goldman mapping and back to decimal index
            trit_representation: str = Goldman().dna_to_trit(index_sequence, start_base=before_end)
            rank: int = GoldmanIndexer.trit_to_decimal(trit_representation)

            return rank
        except EncodingError as _:
            raise RankError(
                message="Could not rank index sequence",
                index_sequence=index_sequence,
                width=self.width
            )


class EnumerativeIndexer(Indexer):
    """
    An indexer that uses enumerative indexing to avoid homopolymer runs in DNA sequences. This scheme is designed to
    avoid homopolymer runs in DNA sequences, which can cause issues in sequencing and synthesis. The enumerative
    indexing scheme is more complex than the Goldman scheme, but it allows for more flexibility in the design of the
    index sequences - where both previous and after sequences can be considered for homopolymer runs. 
    """

    def __init__(self, width: int, max_run_length: int = 1):
        """
        Initializes the EnumerativeIndexer with the number of oligos, width of the index sequence, and maximum allowed 
        homopolymer run length.

        Note: cannot be combined with state aware encoding methods as the indexer needs to know the after as a DNA
        sequenece, but the after requires the previous base (which would be the index base that cannot be encoded).
        This circular dependency is fine, just mean Pgrams, Complementary Tree and Goldman cannot be state aware - no
        homopolymers will occur as the indexer still solves for that, just cannot be state aware together. The 
        recommended solution is to encode the after first as it is the main purpose of the encoding, then use the 
        indexer.
        """
        super().__init__(width)
        self.max_run_length: int = max_run_length


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    ORDERED_ALPHABET: Final[tuple[str, ...]] = ('A', 'C', 'G', 'T')
    VALID_NUM: Final[int] = len(ORDERED_ALPHABET)

    # Arbitrary limit to prevent infinite loops in determine_optimal_width
    REASONABLE_WIDTH_LIMIT: Final[int] = 50  


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @override
    @staticmethod
    def determine_optimal_width(required_capacity: int, *, max_run_length: int = 1) -> int:
        """
        Determines the minimum width of the index sequence required to uniquely identify a given number of oligos, 
        considering the maximum allowed homopolymer run length.
        
        Args:
            required_capacity: The number of oligos to index (or just just numbers to support in general).
            max_run_length: The maximum allowed homopolymer run length.

        Returns:
            The optimal width of the index sequence.

        Raises:
            ValueError: If the required capacity is not a positive integer.
            RuntimeError: If the optimal width cannot be determined within reasonable bounds (width > 50).
        """
        if required_capacity <= 0:
            raise ValueError("Required capacity must be a positive integer.")

        # Start with a width of 1 and increment until we find a width that can accommodate the number of oligos
        width: int = 1
        indexer = EnumerativeIndexer(width, max_run_length=max_run_length)
        while True:
            total_valid_indices: int = indexer.calculate_total_valid_indices()

            # Found a width that can accommodate the number of oligos, return it
            if total_valid_indices >= required_capacity:
                return width
            width += 1
            indexer.width = width

            # Prevent infinite loop
            if width > EnumerativeIndexer.REASONABLE_WIDTH_LIMIT:  
                raise RuntimeError("Unable to determine optimal width within reasonable bounds.")


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _run_length(self, sequence: str, reverse: bool) -> int:
        """
        Calculates the homopolymer run length of a given sequences, either start or end

        Args:
            sequence: The sequence to calculate the run length for.
            reverse: A boolean indicating whether to calculate the run length from the end or start

        Returns:
            The homopolymer run length of the sequence.
        """
        
        run_length: int = 1

        # Reversing the sequence takes a bunch of time espiecally if we have a lot of oligos, so we just iterate 
        #   backwards instead of reversing the string

        # Start from the end of the sequence if reverse is True
        if reverse:
            for index in range(len(sequence) - 1, 0, -1):
                if sequence[index] == sequence[index - 1]:
                    run_length += 1
                else:
                    break
        # Start from the beginning of the sequence if reverse is False
        else:
            for index in range(1, len(sequence)):
                if sequence[index] == sequence[index - 1]:
                    run_length += 1
                else:
                    break

        return run_length

    def _sequence_gaurd(self, before_sequence: str, after_sequence: str, ignore_run_length: bool = False) -> tuple[int, int]:
        """
        Validates the before and after sequences to ensure they are not empty and do not exceed the maximum run length.

        Args:
            before_sequence: The before sequence used in the oligo.
            after_sequence: The after sequence used in the oligo.
            ignore_run_length: A boolean indicating whether to ignore the maximum run length check.

        Returns:
            A tuple containing the homopolymer run lengths of the before and after sequences.

        """
        before_run: int = self._run_length(before_sequence, reverse=True)
        after_run: int = self._run_length(after_sequence, reverse=False)

        # If we're ignoring the run length, we'll cap the run lengths at the maximum run length
        if ignore_run_length and self.max_run_length >= 0:
            before_run = min(before_run, self.max_run_length)
            after_run = min(after_run, self.max_run_length)

        return before_run, after_run


    @lru_cache(maxsize=None)    # Makes this way faster, tradeoff with memory but a good idea for speed
    def _num_valid_placements(self, bases_left: int, same_after_base: bool, 
                                    index_run: int, after_run: int) -> int:
        """
        Calculates the number of valid ways there are to place the next n bases left in the index sequence.

        Args:
            bases_left: The number of bases left to place in the index sequence.
            same_after_base: A boolean indicating whether the next base is the same as the prev base in the after.
            index_run: The current homopolymer run length of the same base in the index sequence.
            after_run: The current homopolymer run length of the same base in the after sequence.

        Returns:
            The number of valid ways to place the next n bases.
        """

        # Have no more nucleotides to place, meaning index is complete.
        if bases_left == 0:
            # If not same base as after and smaller than max run length, then we have 1 valid placement
            if (not same_after_base) or (index_run + after_run <= self.max_run_length):
                return 1
            # Same base as after or runlength exceeds max run length, then 0 valid placements (nothing added)
            else:
                return 0

        # First option: switch to a different base (3 choices, each starts new run of 1)
        valids: int = 0

        # Previous base is same as after base, so none of the bases can be the same as the after base
        if same_after_base:
            valids = self._num_valid_placements(
                bases_left - 1,      # One less base to place
                False,               # Next base not same as after nt
                1,                   # New run of 1
                after_run            # Stays the same
            ) * (self.VALID_NUM - 1) # Multiply by 3 as 3 valid bases to choose from
            
        # Previous base is different from after base
        else:
            # The next base can be the same as the after base (1 choice, starts new run of 1)
            valids += self._num_valid_placements(
                bases_left - 1, # One less base to place
                True,           # Next base is same as after nt
                1,              # New run of 1
                after_run       # Stays the same
            ) 

            # The next base can be any of the other bases (2 choices, starts new run of 1)
            valids += self._num_valid_placements(
                bases_left - 1,      # One less base to place
                False,               # Next base not same as after nt
                1,                   # New run of 1
                after_run            # Stays the same
            ) * (self.VALID_NUM - 2) # Multiply by 2 as 2 valid bases to choose from

        # Second option: continue the same base (1 choice, increases run length by 1). This creates homopolymer runs,
        #   which are generally undesirable in DNA sequences, but we allow it for an option for the user.
        if index_run < self.max_run_length:
            valids += self._num_valid_placements(
                bases_left - 1,    # One less base to place                         
                same_after_base, # Next base is same as prev base in index
                index_run + 1,     # Increases run length by 1
                after_run        # Stays the same
            )

        return valids 


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#
    

    def rank(self, index_sequence: str, *, before_sequence: str, after_sequence: str, ignore_run_length: bool = False) -> int:
        """
        Calculates the rank of a given index sequence based on the enumerative indexing scheme. Basically gives the
        index based on the sequence.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            before_sequence: The before sequence used in the oligo.
            after_sequence: The sequence after the data in the oligo.
        
        Returns:
            The rank of the index sequence.

        Raises:
            ValueError: If width is not a positive integer or if the before/after sequence exceeds maximum run length.
            RankError: If the index sequence length does not match the expected width.
        """
        if self.width <= 0:
            raise ValueError("Width must be a positive integer.")

        # Check if the index sequence length matches the expected width
        if len(index_sequence) != self.width:
            raise RankError(
                message="Could not rank index sequence",      
                index_sequence=index_sequence, 
                width=self.width
            )

        # Gaurd to ensure that the before and after sequences do not exceed the maximum run lengths
        before_run: int 
        after_run: int
        before_run, after_run = self._sequence_gaurd(before_sequence, after_sequence, ignore_run_length)

        # Ensure we always have a previous start or end base (if no primers this can happen)
        previous_base: str = before_sequence[-1] if len(before_sequence) > 0 else Huffman.START_BASE
        ending_base: str = after_sequence[0] if len(after_sequence) > 0 else Huffman.START_BASE
    
        index_run: int = before_run
        new_run: int = 0

        rank: int = 0

        for position, actual_base in enumerate(index_sequence):
            # Reset found flag for each position in the index sequence
            found: bool = False
            
            for candidate_base in self.ORDERED_ALPHABET:
                if candidate_base == previous_base:
                    # Skip this base if it would exceed the max run length
                    if index_run >= self.max_run_length:
                        continue

                    # If we are allowing homopolymer runs, then we can continue the run of the same base
                    new_run = index_run + 1
            
                # Reset run length for a new base
                else:
                    new_run = 1

                # Count up each base that is lexicographically smaller than the actual base, and add to rank
                if candidate_base != actual_base:
                    rank += self._num_valid_placements(
                        self.width - position - 1,     
                        candidate_base == ending_base,  
                        new_run,                        
                        after_run                      
                    )

                # We found the actual base, so we can break out of the ordered alphabet loop and move to the next p
                #   osition in the index sequence
                else:
                    # Update previous base and run length for next iteration
                    previous_base = candidate_base
                    index_run = new_run

                    # Set found flag to True and break out of the loop
                    found = True
                    break

            # Means the actual base was not found in the ordered alphabet, which should never happen unless error
            if not found:
                raise RankError(
                    message="Could not rank index sequence",
                    index_sequence=index_sequence,
                    width=self.width
                )

        # Check the boundary between index and after. Should never be a problem.
        if (not ignore_run_length and previous_base == ending_base and index_run + after_run > self.max_run_length):
            raise MaxRunLengthError(
                message="Index sequence and after exceed the maximum run length",
                sequence=index_sequence,
                max_run_length=self.max_run_length
            )

        return rank
        

    def unrank(self, index: int, *, before_sequence: str, after_sequence: str, ignore_run_length: bool = False) -> str:
        """
        Generates the index sequence corresponding to a given rank based on the enumerative indexing scheme.

        Args:
            index: The rank of the index sequence to generate.
            before_sequence: The before sequence used in the oligo.
            after_sequence: The after sequence used in the oligo.

        Returns:
            The index sequence corresponding to the given rank.

        Raises:
            ValueError: If the index or width is negative, or if the before or after sequences are empty
        """
        if self.width <= 0:
            raise ValueError("Width must be a positive integer.")
    
        if index < 0:
            raise ValueError("Index must be a non-negative integer.")

        before_run: int
        after_run: int 
        before_run, after_run = self._sequence_gaurd(before_sequence, after_sequence, ignore_run_length)
        previous_base: str = before_sequence[-1] if len(before_sequence) > 0 else Huffman.START_BASE
        ending_base: str = after_sequence[0] if len(after_sequence) > 0 else Huffman.START_BASE

        index_run: int = before_run
        new_run: int = 0

        # Calculate the total number of valid indices first to determine if the given rank is valid
        total: int = self._num_valid_placements(
            self.width,                    
            previous_base == ending_base,
            index_run,
            after_run,
        )
        if index >= total:
            raise UnrankError(
                message="Index is not in bounds of capacity",
                index=index,
                capacity=total
            )

        index_sequence: str = ""
        # Go through each position in the index sequence and determine which base corresponds to the current rank
        for position in range(self.width):
            # Go through each base in the ordered alphabet to find the one that corresponds to the current rank
            for candidate_base in self.ORDERED_ALPHABET:
                # Determine run length if this base is selected
                if candidate_base == previous_base:
                    # Skip this base if it would exceed the max run length
                    if index_run >= self.max_run_length:
                        continue
                    new_run = index_run + 1

                # Reset run length for a new base
                else:
                    new_run = 1

                # Count all sequences beginning with this candidate.
                block = self._num_valid_placements(
                    self.width - position - 1,
                    candidate_base == ending_base,
                    new_run,
                    after_run,
                )

                # Index lies in this block: commit to this base and move to the next position in the index sequence
                if index < block:
                    index_sequence += candidate_base

                    # Update previous base and run length for next iteration
                    previous_base = candidate_base
                    index_run = new_run
                    break

                # Desired sequence occurs later
                index -= block

        return index_sequence


    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences that can be generated given the before and after 
        sequences, using worst case scenario for homopolymer runs.

        Returns:
            The total number of valid index sequences.
        """
        before_run: int = self.max_run_length
        after_run: int = self.max_run_length

        # Calculate the total number of valid indices for the current width, assumes before and after are of 
        #   length 1, which is the worst case scenario for homopolymer runs
        total_valid_indices: int = self._num_valid_placements(
            self.width,                    
            True,
            before_run,                    
            after_run                    
        )

        return total_valid_indices