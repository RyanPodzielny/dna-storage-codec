"""
Module provides inner error correction codes for DNA encoding. Inner codes are applied to the data before it is 
converted to DNA sequences. They are used to correct errors that occur during the synthesis and sequencing of the
DNA. The inner codes are designed to work with the indexing and encoding methods provided in the Encode package, and can
be used in conjunction with outer error correction codes for additional error correction
capabilities. The inner codes are implemented using the reedsolo library for Reed-Solomon error
"""


from abc import abstractmethod

import reedsolo 
from fastcrc import crc16, crc32
from typing import Any, Callable, Final, override
from Encode.Indexer import Indexer
from Encode.Methods.Encoding import Base, Encoding
from Helpers.Errors import InnerCodeError, RankError


class InnerCodes():
    """
    Abstract base class for inner error correction codes. This class defines the interface for encoding and decoding
    data to and from DNA sequences. Subclasses must implement the encode_to_dna and decode_from_dna methods to provide 
    specific error correction functionality.
    """
    def __init__(self, indexer: type[Indexer], *args, **kwargs):
        self.indexer = indexer


    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this inner code when loading a saved framer.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {}


    @abstractmethod
    def encode_to_dna(self, before_sequence: str, after_sequence: str, *args, **kwargs) -> tuple[str, str]:
        raise NotImplementedError()


    @abstractmethod
    def decode_from_dna(self, 
                        encoded_data: str, 
                        before_sequence: str, after_sequence: str,
                        *args, **kwargs) -> tuple[str, str]:
        raise NotImplementedError()



class CheckSum(InnerCodes):
    """
    A class for calculating and verifying checksums. Uses Cyclic Redundancy Check (CRC) for error detection. In our
    specific case we will use CRC-32 on each oligio, if we have an inner ECC code, e.g. RS, we will use CRC-16 to save
    space as the inner code will already provide some error correction.
    """

    def __init__(self, indexer: type[Indexer], *, inner_code: bool = False, max_run_length: int | None = None):
        """
        Initializes the CheckSum instance.

        Args:
            inner_code (bool): A flag indicating whether to use inner code. 
        """

        self.inner_code: bool = inner_code

        # Default to CRC-32 for checksum calculation if we have an inner_code, use CRC-16 instead
        #   Everything comes from: https://fastcrc.readthedocs.io/en/latest/
        self.width: int = 0
        self.crc_func: Callable
        if self.inner_code:
            self.crc_func = crc16.xmodem
            self.width = 16
        else:
            self.crc_func = crc32.iso_hdlc
            self.width = 32

        # Bit shift left to get 2^required_width, which is number of unique values we can have
        self.indexer: Indexer = indexer(
            width=indexer.determine_optimal_width(1 << self.width),
            max_run_length=max_run_length
        )


    @override
    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this instance when loading a saved framer.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {
            "indexer": type(self.indexer),
            "inner_code": self.inner_code,
            "max_run_length": self.indexer.max_run_length
        }


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def get_dna_bases(self) -> int:
        """
        Gets the length of bits used for the checksum based on the configuration,

        Returns:
            int: The length of the checksum in bases
        """

        return self.indexer.width
    

    def get_checksum_length(self) -> int:
        """
        Gets the length of bits used for the checksum based on the configuration,

        Returns:
            int: The length of the checksum in bases
        """

        return self.width


    def calculate_checksum(self, encoded_data: str) -> int:
        """
        Calculates the checksum for the given data.

        Args:
            data: The index + after data for which the checksum is to be calculated.

        Returns:
            int: The calculated checksum.

        Raises:
            ValueError: If the data is empty or None.
        """
        if not encoded_data:
            raise ValueError("Data for checksum calculation cannot be empty or None.")

        # Convert data into bytes
        byte_stream: bytes = bytes(encoded_data, 'utf-8')

        return self.crc_func(byte_stream)


    def create_checksum_segments(self, encoded_data: list[str]) -> list[int]:
        """
        Calculates the checksum for each string in the provided list.

        Args:
            encoded_data: A list of strings for which checksums will be calculated - these are DNA sequences.

        Returns:
            list[int]: A new list of integers with checksums appended.

        Raises:
            ValueError: If the unencoded_data list is empty or None.
        """
        checksums: list[int] = []

        for item in encoded_data:
            if not item:
                raise ValueError("Data for checksum calculation cannot be empty or None.")

            checksum: int = self.calculate_checksum(item)
            checksums.append(checksum)

        return checksums


    def verify_checksum(self, encoded_data: str, checksum: int) -> bool:
        """
        Verifies the checksum of the provided data.

        Args:
            encoded_data: The data string with the appended checksum.
            checksum: The checksum to verify against.

        Returns:
            bool: True if the checksum is valid, False otherwise.

        Raises:
            ValueError: If the data is empty or None.
        """
        if not encoded_data:
            raise ValueError("Data for checksum verification cannot be empty or None.")

        # Calculate the checksum for the original data
        calculated_checksum: int = self.calculate_checksum(encoded_data)

        # Compare the calculated checksum with the provided checksum
        return calculated_checksum == checksum


    def encode_to_dna(self, before_sequence: str, after_sequence: str, *, checksum: int) -> str:
        """
        Converts the checksum integer to a DNA sequence.

        Args:
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            checksum: The checksum integer to be converted.

        Returns:
            str: The corresponding DNA sequence.
        
        Raises:
            UnrankError: If unable to unrank the checksum integer to a DNA sequence.
        """
        # Convert the checksum integer to a DNA sequence using the indexer
        dna_sequence: str = self.indexer.unrank(
            index=checksum, 
            before_sequence=before_sequence, 
            after_sequence=after_sequence
        )

        return dna_sequence


    def decode_from_dna(self, encoded_data: str, before_sequence: str, after_sequence: str, *args, **kwargs) -> int:
        """
        Converts a DNA sequence back to the corresponding checksum integer.

        Args:
            encoded_data: The DNA sequence to be converted.
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            *args and **kwargs there to maintain polymorphisms - unused

        Returns:
            int: The corresponding checksum integer.

        Raises:
            RankError: If unable to rank the DNA sequence back to a checksum integer.
        """
        # Convert the DNA sequence back to the checksum integer using the indexer
        checksum: int = self.indexer.rank(
            index_sequence=encoded_data, 
            before_sequence=before_sequence, 
            after_sequence=after_sequence,
            ignore_run_length=True
        )

        # Verify that the checksum can be unranked back to the same DNA sequence - if it can't it means the checksum is
        #   likely corrupted (for whatever reason) and we should not trust it. 
        canonical_sequence: str = self.indexer.unrank(
            index=checksum,
            before_sequence=before_sequence,
            after_sequence=after_sequence,
            ignore_run_length=True
        )
        if canonical_sequence != encoded_data:
            raise RankError(
                message="Could not rank checksum sequence",
                index_sequence=encoded_data,
                width=self.indexer.width
            )

        return checksum



class RSInner(InnerCodes):
    """
    Class to implement Reed-Solomon error correction for inner codes. This class provides methods to encode and decode 
    data into dna sequences with Reed-Solomon error correction/parity. Uses the `reedsolo` library for encoding and 
    decoding as its lightweight and fast. Only corrects substitution errors, not insertions or deletions.
    """


    def __init__(self, indexer: type[Indexer], *, parity_bytes: int, max_run_length: int | None = None):
        """
        Initializes the ReedSolomon instance.

        Args:
            parity_bytes: The number of Reed-Solomon parity bytes to append.
            indexer: The type of indexer to use for the conversion.
            max_run_length: The maximum allowed run length in the DNA sequence.

        Raises:
            ValueError: If parity_bytes is not between 2 and 16.
        """
        # Need a hard cap on the capacity. We need to represent them in a fixed width scheme for our dna, and we are 
        #   using a byte representation combination to achieve this. Becomes computationally expensive and impractical.
        # Need at least 2 parity bytes to correct 1 error, and 16 is just the practical limit.
        if not 2 <= parity_bytes <= 16:
            raise ValueError("parity_bytes must be between 2 and 16.")

        self.parity_bytes: int = parity_bytes
        self.RS: reedsolo.RSCodec = reedsolo.RSCodec(parity_bytes)

        # Going to encode each base as the byte we have - so maxium of 256 numbers per byte
        self.indexer: Indexer = indexer(
            width=indexer.determine_optimal_width(1 << 8),
            max_run_length=max_run_length
        )


    @override
    def get_spec_args(self) -> dict[str, Any]:
        """
        Gets the constructor arguments needed to rebuild this instance when loading a saved framer.

        Returns:
            A dictionary of constructor argument names to their values.
        """
        return {
            "indexer": type(self.indexer),
            "parity_bytes": self.parity_bytes,
            "max_run_length": self.indexer.max_run_length
        }


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC CONSTANTS                                                       #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Padding base for Reed-Solomon encoding
    RS_PADDING: Final[str] = Base.A.value 


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPERS METHODS                                                #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _to_bytes(self, encoded_data: str) -> bytearray:
        """
        Converts a DNA sequence to bytes with virtual padding.

        Args:
            encoded_data: The DNA sequence to be converted.

        Returns:
            bytes: The corresponding byte representation.
        """
        # We will treat each base as 2 bits, so we can pack 4 bases into a single byte. We are encoding over the DNA -
        #   we do not care about the actual information it is storing, we just want to be able to correct errors in the 
        #   DNA sequence. Padding is necessary, but its virtual and will not actually show up into the DNA sequence. We
        #   use Reed-Solomon with a GF(256) field, which means we must store the data in the same size as a byte. 
        #   Techinally could just use each base as a byte, but it limits oligo length and is very wasteful. This way we
        #   can extend how long each oligo can be and still have error correction (e.g. 200 bases = 50 bytes, instead of 
        #   200 bytes).

        length: int = len(encoded_data)

        # Pad the data to a multiple of 4 bases - this is just virtual, nothing is added to sequence
        if length % 4 != 0:
            padding_length: int = 4 - (length % 4)
            encoded_data += self.RS_PADDING * padding_length
            length += padding_length

        # Convert the DNA sequence to bytes - our Encoding class has this but in str form, so we will just use that
        bit_stream: str = "".join(Encoding.REVERSE_DNA_MAP[base] for base in encoded_data)

        # Convert the bit stream to bytes
        byte_stream: bytearray = bytearray()
        for index in range(0, len(bit_stream), 8):
            byte: str = bit_stream[index:index+8]
            try:
                byte_stream += bytes([int(byte, 2)])
            except ValueError as _:
                raise InnerCodeError(
                    message="Invalid DNA sequence for byte conversion", 
                    sequence=bit_stream[index:index+8]
                )

        return byte_stream


    def _encode(self, encoded_data: str) -> list[int]:
        """
        Encodes the given data using Reed-Solomon error correction.

        Args:
            encoded_data: The data to be encoded with parity. We assume its the [index | payload | checksum] sequence.

        Returns:
            The list of Reed-Solomon parity bytes.
        """
       
        byte_stream: bytearray = self._to_bytes(encoded_data)

        # Return only the parity bytes
        return list(self.RS.encode(byte_stream)[-self.parity_bytes:])


    def _decode(self, 
                encoded_data: str, 
                parity_bytes: bytearray, 
                sequence_length: int, 
                erasures: list[int]) -> tuple[str, bytearray]:
        """
        Decodes the given data using Reed-Solomon error correction.

        Args:
            encoded_data: The data to be decoded with parity. We assume its the [index | payload | checksum] sequence.
            parity_bytes: The Reed-Solomon parity bytes to be used for error correction.
            sequence_length: The length of the original sequence [index | payload | checksum] without the parity bytes.
            erasures: The positions of the erasures in the parity sequence, i.e. known to be incorrect.

        Returns:
            A tuple containing the decoded data without parity and the corrected parity bytes.

        Raises:
            ReedSolomonError: If the data cannot be corrected.
            KeyError: If the bits are not in the DNA_MAP, which should not happen if the data is valid.
        """
        byte_stream: bytearray = self._to_bytes(encoded_data)
        stream_length: int = len(byte_stream)

        # Record the positions of the erasures in the byte stream
        erase_positions: list[int] = [stream_length + erasure for erasure in erasures]

        # Add rest of the parity bytes
        byte_stream += parity_bytes

        # Keep the corrected full codeword so corrected parity is available and seperate them
        corrected_codeword: bytearray = bytearray(self.RS.decode(byte_stream, erase_pos=erase_positions)[1])
        corrected_data = corrected_codeword[:stream_length]
        corrected_parity = corrected_codeword[stream_length:]

        # Convert the decoded bytes back to a DNA sequence
        bit_stream: str = "".join(f"{byte:08b}" for byte in corrected_data)
        decoded_sequence: str = ""
        for index in range(0, len(bit_stream), 2):
            bits: str = bit_stream[index:index+2]
            decoded_sequence += Encoding.DNA_MAP[bits]
    
        # Return only the original sequence length, discard any padding and parity bytes corrected
        return decoded_sequence[:sequence_length], corrected_parity


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def encode_to_dna(self, before_sequence: str, after_sequence: str, *, encoded_data: str,) -> str:
        """
        Converts the Reed-Solomon parity bytes integer to a DNA sequence.

        Args:
            before_sequence: The sequence before the parity bytes in the oligo.
            after_sequence: The sequence after the parity bytes in the oligo.
            encoded_data: The data to be encoded with parity. We assume its the [index | payload | checksum] sequence.
    
        Returns:
            The corresponding DNA sequence to the Reed-Solomon parity bytes.
        
        Raises:
            UnrankError: If unable to unrank the Reed-Solomon parity bytes integer to a DNA sequence.
        """
        parity_bytes: list[int] = self._encode(encoded_data)
        dna_sequence: str = ""

        # Unrank each parity byte individually to a DNA sequence using the indexer
        for byte in parity_bytes:
            dna_sequence += self.indexer.unrank(
                index=byte, 
                before_sequence=before_sequence,
                after_sequence=after_sequence
            )

            # Update the before_sequence for the next parity byte to be encoded
            before_sequence = dna_sequence 

        return dna_sequence


    def decode_from_dna(self, 
                        encoded_data: str, 
                        before_sequence: str, 
                        after_sequence: str,
                        *,
                        parity_dna_sequence: str, ) -> tuple[str, str]:
        """
        Attempts to decode the given DNA sequence using Reed-Solomon error correction.

        Args:
            encodeded_data: The DNA sequence without the RS parity bytes, i.e. [index | payload | checksum] sequence.
            before_sequence: The sequence before the parity bytes in the oligo.
            after_sequence: The sequence after the parity bytes in the oligo.
            parity_dna_sequence: The DNA sequence representing the Reed-Solomon parity bytes.

        Returns:
            A tuple containing the corrected data DNA and corrected RS parity DNA.

        Raises:
            RankError: If unable to rank the DNA sequence back to a Reed-Solomon parity bytes integer.
            ReedSolomonError: If the data cannot be corrected.
            KeyError: If the bits are not in the DNA_MAP, which should not happen if the data is valid.
        """
        parity_bytes: bytearray = bytearray()

        # Incase we have unrecoverable parity errors (cannot rank them), mark that location for reedsolo decode
        erasures: list[int] = []

        # Each parity byte is represented by a DNA sequence of length equal to the indexer's width
        for index in range(self.parity_bytes):
            # Get the exact DNA sequence for the current parity byte
            start: int = index * self.indexer.width
            end: int = start + self.indexer.width
            current_sequence: str = parity_dna_sequence[start:end]

            try:
                # Rank the current DNA sequence to get the corresponding parity byte
                parity_byte: int = self.indexer.rank(
                    index_sequence=current_sequence, 
                    before_sequence=before_sequence, 
                    after_sequence=after_sequence,
                    ignore_run_length=True
                )
                
                # Can't decode, treat as an erasure
                if not (0 <= parity_byte < 256):
                    parity_byte = 0
                    erasures.append(index)
            # If we cannot unrank, treat as ersasure - we will just use a 0 for the parity byte, but mark it as an erasure for the RS decoder
            except RankError as _:
                parity_byte: int = 0
                erasures.append(index)
        
            # Update new start sequence for the next parity byte to be decoded
            before_sequence = current_sequence

            parity_bytes.append(parity_byte)

        # Decode the data using the Reed-Solomon decoder
        decoded_sequence, corrected_parity_bytes = self._decode(
            encoded_data=encoded_data,
            parity_bytes=parity_bytes, 
            sequence_length=len(encoded_data),
            erasures=erasures
        )

        # Use the corrected checksum as the context for the corrected parity DNA
        corrected_before: str = decoded_sequence[-len(before_sequence):] if before_sequence else ""
        corrected_parity_dna: str = ""

        # Unrank the corrected parity bytes back to DNA sequences
        for byte in corrected_parity_bytes:
            parity_sequence = self.indexer.unrank(
                index=byte,
                before_sequence=corrected_before,
                after_sequence=after_sequence
            )
            corrected_parity_dna += parity_sequence
            corrected_before = corrected_parity_dna

        # Return the decoded sequence and the corrected parity DNA sequence
        return decoded_sequence, corrected_parity_dna


    def get_dna_bases(self) -> int:
        """
        Gets the length of the Reed-Solomon parity bytes in bases.

        Returns:
            int: The length of the Reed-Solomon parity bytes in bases.
        """
        return self.indexer.width * self.parity_bytes