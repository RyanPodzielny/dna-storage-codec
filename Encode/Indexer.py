"""
Module for enumerative indexing of DNA sequences. This module provides classes and methods to perform indexing on 
oligos, which is needed to identify them for decoding. 
"""

import math
from abc import abstractmethod
from functools import lru_cache
from typing import Any, Final, override
from Encode.Methods.HuffmanCoding import Goldman
from Encode.Methods.Encoding import Encoding

class Indexer:
    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the Indexer the width of the index sequence.

        Args:
            width: The width of the index sequence.
            Args and kwargs are included for polymorphism, but unused
        """
        self.width: int = width


    @abstractmethod
    def calculate_total_valid_indices(self) -> int:
        raise NotImplementedError("Subclasses must implement this method.")


    @staticmethod
    def determine_optimal_width(num_oligos: int, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError("Subclasses must implement this method.")


    @abstractmethod
    def rank(self, index_sequence: str, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError("Subclasses must implement this method.")


    @abstractmethod
    def unrank(self, index: int, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class DNAMapIndexer(Indexer):
    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the DNAMapIndexer with the width of the index sequence.
        
        Args:
            num_oligos: The number of oligos to index.
            width: The width of the index sequence. 
        """
        super().__init__(width)


    @override
    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences for the DNAMap indexing scheme.

        Returns:
            The total number of valid index sequences.
        """
        return 4 ** self.width


    @override
    @staticmethod
    def determine_optimal_width(required_capacity: int) -> int:
        """
        Determines the minimum width of the index sequence required to uniquely identify a given number of oligos.

        Returns:
            The optimal width of the index sequence.
        """
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


    @override
    def rank(self, index_sequence: str, *args: Any, **kwargs: Any) -> int:
        """
        Calculates the rank of a given index sequence based on the DNA mapping indexing scheme.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            Args and kwargs are included for polymorphism, but unused

        Returns:
            The rank of the index sequence.

        Raises:
            ValueError: If the index sequence length does not match the expected width.
        """
        if len(index_sequence) != self.width:
            raise ValueError(f"Index sequence length {len(index_sequence)} does not match expected width {self.width}.")

        # Convert the index sequence to a binary string
        binary_string: str = ""
        for base in index_sequence:
            binary_string += Encoding.REVERSE_DNA_MAP[base]

        # Convert to decimal
        rank: int = int(binary_string, 2)

        return rank


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
        """
        # Convert the decimal index to a binary string with correct leading zeros (need x2 as each base is 2 bits)
        binary_string: str = f"{index:0{self.width * 2}b}"
        if len(binary_string) > self.width * 2:
            raise ValueError(f"Index {index} exceeds the maximum value for width {self.width}.")

        # Map the binary string to DNA bases using the DNAMap scheme
        unrank: str = ""
        for index in range(0, len(binary_string), 2):
            unrank += Encoding.DNA_MAP[binary_string[index:index + 2]]

        return unrank


class GoldmanIndexer(Indexer):
    def __init__(self, width: int, *args: Any, **kwargs: Any):
        """
        Initializes the GoldmanIndexer with the number of oligos and the width of the index sequence.
        
        Args:
            num_oligos: The number of oligos to index.
            width: The width of the index sequence. If None, it will be determined based on the number of oligos.
        """
        super().__init__(width)


    def _decimal_to_trit(self, index: int) -> str:
        """
        Converts a decimal index to a trit representation.

        Args:
            index: The decimal index to convert.

        Returns:
            The trit representation of the decimal index.
        """
        trit: str = ""
        while index > 0:
            trit = str(index % 3) + trit
            index //= 3

        # Add leading zeros to make representation same as width
        zeros: str = "0" * (self.width - len(trit))

        return zeros + trit


    def _trit_to_decimal(self, trit: str) -> int:
        """
        Converts a trit representation to a decimal index.

        Args:
            trit: The trit representation to convert.

        Returns:
            The decimal index corresponding to the trit representation.
        """
        decimal: int = 0
        for digit in trit:
            decimal = decimal * 3 + int(digit)
        return decimal


    @override
    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences that can be generated given the primer and payload 
        sequences, using worst case scenario for homopolymer runs.

        Returns:
            The total number of valid index sequences.
        """
        # For Goldman indexing, the total number of valid indices is simply 3^width, since each position in the index
        #   sequence can be one of three bases (A, C, G) and we are not considering homopolymer runs in this calculation.
        total_valid_indices: int = 3 ** self.width
        return total_valid_indices
    

    @override
    @staticmethod
    def determine_optimal_width(required_capacity: int) -> int:
        """
        Determines the minimum width of the index sequence required to uniquely identify a given number of oligos.

        Returns:
            The optimal width of the index sequence.
        """
        # We can easily calculate the optimal width for the Goldman indexer by using the formula:
        #   width = ceil(log3(required_capacity))
        width: int = math.ceil(math.log(required_capacity, 3))
        return width


    @override
    def unrank(self, index: int, primer: str, *args: Any, **kwargs: Any) -> str:
        """
        Generates the index sequence corresponding to a given rank based on the Goldman indexing scheme.

        Args:
            index: The rank of the index sequence to generate.
            primer: The primer sequence used in the oligo.
            Args and kwargs are included for polymorphism, but unused

        Returns:
            The index sequence corresponding to the given rank.
        """
        primer_end: str = primer[-1]

        # Convert decimal index to trit representation, then use Goldman mapping to convert to DNA
        trit_representation: str = self._decimal_to_trit(index)
        unrank: str = Goldman().trit_to_dna(trit_representation, start_base=primer_end)
        return unrank


    @override
    def rank(self, index_sequence: str, primer: str, *args: Any, **kwargs: Any) -> int:
        """
        Calculates the rank of a given index sequence based on the Goldman indexing scheme.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            primer: The primer sequence used in the oligo.
            _payload: Unused, included for polymorphism.

        Returns:
            The rank of the index sequence.
        """
        primer_base: str = primer[-1]

        # Convert DNA bases to trits using Goldman mapping and back to decimal index
        trit_representation: str = Goldman().dna_to_trit(index_sequence, start_base=primer_base)
        rank: int = self._trit_to_decimal(trit_representation)

        return rank


class EnumerativeIndexer(Indexer):
    ORDERED_ALPHABET: Final[tuple[str, ...]] = ('A', 'C', 'G', 'T')
    VALID_NUM: Final[int] = len(ORDERED_ALPHABET)
    def __init__(self, width: int, max_run_length: int = 1):
        """
        Initializes the EnumerativeIndexer with the number of oligos, width of the index sequence, and maximum allowed 
        homopolymer run length.

        Note: cannot be combined with state aware encoding methods as the indexer needs to know the payload as a DNA
        sequenece, but the payload requires the previous base (which would be the index base that cannot be encoded).
        This circular dependency is fine, just mean Pgrams, Complementary Tree and Goldman cannot be state aware - no
        homopolymers will occur as the indexer still solves for that, just cannot be state aware together. The 
        recommended solution is to encode the payload first as it is the main purpose of the encoding, then use the 
        indexer.
        """
        self.max_run_length: int = max_run_length
        super().__init__(width)


    def _run_length(self, sequence: str, reverse: bool) -> int:
        """
        Calculates the homopolymer run length of a given sequences, either start or end

        Args:
            sequence: The sequence to calculate the run length for.
            reverse: A boolean indicating whether to calculate the run length from the end or start

        Returns:
            The homopolymer run length of the sequence.

        Raises:
            ValueError: If the sequence is empty.
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty.")

        run_length: int = 1

        # Reversing the sequence takes a bunch of time espiecally if we have a lot of oligos, so we just iterate 
        #   backwards instead of reversing the string

        # Start from the end of the sequence if reverse is True
        if reverse:
            for i in range(len(sequence) - 1, 0, -1):
                if sequence[i] == sequence[i - 1]:
                    run_length += 1
                else:
                    break
        # Start from the beginning of the sequence if reverse is False
        else:
            for i in range(1, len(sequence)):
                if sequence[i] == sequence[i - 1]:
                    run_length += 1
                else:
                    break

        return run_length

    def _sequence_gaurd(self, primer: str, payload: str) -> tuple[int, int]:
        """
        Validates the primer and payload sequences to ensure they are not empty and do not exceed the maximum run length.

        Args:
            primer: The primer sequence used in the oligo.
            payload: The payload sequence used in the oligo.

        Raises:
            ValueError: If the primer or payload sequences are empty, or if they exceed the maximum
            run length.
        """
        if not primer or not payload:
            raise ValueError("Primer and payload sequences cannot be empty.")

        primer_run: int = self._run_length(primer, reverse=True)
        payload_run: int = self._run_length(payload, reverse=False)

        # Check if the primer or payload exceeds the maximum run length, which would make it impossible to generate 
        #   valid index sequences with the max run length constraint.
        if primer_run > self.max_run_length or payload_run > self.max_run_length:
            raise ValueError("Primer or payload exceeds the maximum run length.")

        return primer_run, payload_run


    @lru_cache(maxsize=None)    # Makes this way faster, tradeoff with memory but a good idea for speed
    def _num_valid_placements(self, bases_left: int, same_payload_base: bool, 
                                    index_run: int, payload_run: int) -> int:
        """
        Calculates the number of valid ways there are to place the next n bases left in the index sequence.

        Args:
            bases_left: The number of bases left to place in the index sequence.
            same_payload_base: A boolean indicating whether the next base is the same as the prev base in the payload.
            index_run: The current homopolymer run length of the same base in the index sequence.
            payload_run: The current homopolymer run length of the same base in the payload sequence.

        Returns:
            The number of valid ways to place the next n bases.
        """

        # Have no more nucleotides to place, meaning index is complete.
        if bases_left == 0:
            # If not same base as payload and smaller than max run length, then we have 1 valid placement
            if (not same_payload_base) or (index_run + payload_run <= self.max_run_length):
                return 1
            # Same base as payload or runlength exceeds max run length, then 0 valid placements (nothing added)
            else:
                return 0

        # First option: switch to a different base (3 choices, each starts new run of 1)
        valids: int = 0

        # Previous base is same as payload base, so none of the bases can be the same as the payload base
        if same_payload_base:
            valids = self._num_valid_placements(
                bases_left - 1,      # One less base to place
                False,               # Next base not same as payload nt
                1,                   # New run of 1
                payload_run          # Stays the same
            ) * (self.VALID_NUM - 1) # Multiply by 3 as 3 valid bases to choose from
            
        # Previous base is different from payload base
        else:
            # The next base can be the same as the payload base (1 choice, starts new run of 1)
            valids += self._num_valid_placements(
                bases_left - 1, # One less base to place
                True,           # Next base is same as payload nt
                1,              # New run of 1
                payload_run     # Stays the same
            ) 

            # The next base can be any of the other bases (2 choices, starts new run of 1)
            valids += self._num_valid_placements(
                bases_left - 1,      # One less base to place
                False,               # Next base not same as payload nt
                1,                   # New run of 1
                payload_run          # Stays the same
            ) * (self.VALID_NUM - 2) # Multiply by 2 as 2 valid bases to choose from

        # Second option: continue the same base (1 choice, increases run length by 1). This creates homopolymer runs,
        #   which are generally undesirable in DNA sequences, but we allow it for an option for the user.
        if index_run < self.max_run_length:
            valids += self._num_valid_placements(
                bases_left - 1,    # One less base to place                         
                same_payload_base, # Next base is same as prev base in index
                index_run + 1,     # Increases run length by 1
                payload_run        # Stays the same
            )

        return valids 


    def calculate_total_valid_indices(self) -> int:
        """
        Calculates the total number of valid index sequences that can be generated given the primer and payload 
        sequences, using worst case scenario for homopolymer runs.

        Returns:
            The total number of valid index sequences.
        """
        primer_run: int = self.max_run_length
        payload_run: int = self.max_run_length

        # Calculate the total number of valid indices for the current width, assumes primer and payload are of 
        #   length 1, which is the worst case scenario for homopolymer runs
        total_valid_indices: int = self._num_valid_placements(
            self.width,                    
            True,
            primer_run,                    
            payload_run                    
        )

        return total_valid_indices
    

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
        """

        # Start with a width of 1 and increment until we find a width that can accommodate the number of oligos
        width: int = 1
        
        while True:
            indexer = EnumerativeIndexer(width, max_run_length=max_run_length)

            total_valid_indices: int = indexer.calculate_total_valid_indices()

            # Found a width that can accommodate the number of oligos, return it
            if total_valid_indices >= required_capacity:
                return width
            width += 1

            # Prevent infinite loop
            if width > 50:  
                raise ValueError("Unable to determine optimal width within reasonable bounds.")


    def rank(self, index_sequence: str, *, primer: str, payload: str) -> int:
        """
        Calculates the rank of a given index sequence based on the enumerative indexing scheme. Basically gives the
        index based on the sequence.

        Args:
            index_sequence: The index sequence to calculate the rank for.
            primer: The primer sequence used in the oligo.
            payload: The payload sequence used in the oligo.
        
        Returns:
            The rank of the index sequence.

        Raises:
            ValueError: If the index sequence length does not match the expected width, or if the
                primer or payload sequences are empty, or if the primer or payload exceeds the maximum run length.
        """

        if len(index_sequence) != self.width:
            raise ValueError(f"Index sequence length {len(index_sequence)} does not match expected width {self.width}.")

        primer_run: int 
        payload_run: int
        primer_run, payload_run = self._sequence_gaurd(primer, payload)
        previous_base: str = primer[-1]
        ending_base: str = payload[0]

        index_run: int = primer_run
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
                        payload_run                      
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

            # Means the actual base was not found in the ordered alphabet, which should never happen
            if not found:
                raise ValueError(f"Invalid index sequence '{index_sequence}'.")

        # Check the boundary between index and payload.
        if (previous_base == ending_base) and (index_run + payload_run > self.max_run_length):
            raise ValueError("Index sequence and payload exceed the maximum run length.")

        return rank
        

    def unrank(self, index: int, *, primer: str, payload: str) -> str:
        """
        Generates the index sequence corresponding to a given rank based on the enumerative indexing scheme.

        Args:
            index: The rank of the index sequence to generate.
            primer: The primer sequence used in the oligo.
            payload: The payload sequence used in the oligo.

        Returns:
            The index sequence corresponding to the given rank.

        Raises:
            ValueError: If the index is negative, or if the primer or payload sequences are empty
        """

        if index < 0:
            raise ValueError("Rank cannot be negative.")

        primer_run: int
        payload_run: int 
        primer_run, payload_run = self._sequence_gaurd(primer, payload)
        previous_base: str = primer[-1]
        ending_base: str = payload[0]

        index_run: int = primer_run
        new_run: int = 0

        # Calculate the total number of valid indices first to determine if the given rank is valid
        total: int = self._num_valid_placements(
            self.width,                    
            previous_base == ending_base,
            index_run,
            payload_run,
        )
        if index >= total:
            raise ValueError(f"Rank {index} exceeds the total number of valid indices {total}.")

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
                    payload_run,
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