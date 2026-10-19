import reedsolo
from fastcrc import crc16, crc32
from typing import Callable
from Encode.Indexer import Indexer
from Encode.Methods.Encoding import Encoding


class CheckSum:
    """
    A class for calculating and verifying checksums. Uses Cyclic Redundancy Check (CRC) for error detection. In our
    specific case we will use CRC-32 on each oligio, if we have an inner ECC code, e.g. RS, we will use CRC-16 to save
    space as the inner code will already provide some error correction.
    """

    def __init__(self, indexer: type[Indexer], inner_code: bool = False, max_run_length: int | None = None):
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
            width=indexer.determine_optimal_width(2**self.width),
            max_run_length=max_run_length
        )


    def get_checksum_bases(self) -> int:
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


    def calculate_checksum(self, data: str) -> int:
        """
        Calculates the checksum for the given data.

        Args:
            data: The index + after data for which the checksum is to be calculated.

        Returns:
            int: The calculated checksum.
        """
        # Convert data into bytes
        byte_stream: bytes = bytes(data, 'utf-8')

        return self.crc_func(byte_stream)


    def create_checksum_segments(self, unencoded_data: list[str]) -> list[int]:
        """
        Calculates the checksum for each string in the provided list.

        Args:
            unencoded_data: A list of strings for which checksums will be calculated - not DNA bases.

        Returns:
            list[int]: A new list of integers with checksums appended.
        """
        checksums: list[int] = []

        for item in unencoded_data:
            checksum: int = self.calculate_checksum(item)
            checksums.append(checksum)

        return checksums


    def verify_checksum(self, data: str, checksum: int) -> bool:
        """
        Verifies the checksum of the provided data.

        Args:
            data: The data string with the appended checksum.
            checksum: The checksum to verify against.

        Returns:
            bool: True if the checksum is valid, False otherwise.
        """

        # Calculate the checksum for the original data
        calculated_checksum: int = self.calculate_checksum(data)

        # Compare the calculated checksum with the provided checksum
        return calculated_checksum == checksum



    def encode_to_dna(self, checksum: int, before_sequence: str, after_sequence: str) -> str:
        """
        Converts the checksum integer to a DNA sequence.

        Args:
            checksum: The checksum integer to be converted.
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            indexer: The indexer to use for the conversion.

        Returns:
            str: The corresponding DNA sequence.
        """
        # Convert the checksum integer to a DNA sequence using the indexer
        dna_sequence: str = self.indexer.unrank(checksum, 
                                                before=before_sequence, 
                                                after=after_sequence)

        return dna_sequence



    def decode_from_dna(self, dna_sequence: str, before_sequence: str, after_sequence: str) -> int:
        """
        Converts a DNA sequence back to the corresponding checksum integer.

        Args:
            dna_sequence: The DNA sequence to be converted.
            before_sequence: The sequence before the checksum in the oligo.
            after_sequence: The sequence after the checksum in the oligo.
            indexer: The indexer to use for the conversion.

        Returns:
            int: The corresponding checksum integer.
        """
        # Convert the DNA sequence back to the checksum integer using the indexer
        checksum: int = self.indexer.rank(dna_sequence, 
                                          before=before_sequence, 
                                          after=after_sequence)

        return checksum


class ReedSolomon:
    """
    Reed-Solomon error correction for DNA strands.
    """
    # Padding base for Reed-Solomon encoding
    RS_PADDING: str = 'A' 

    def __init__(self, parity_bytes: int, indexer: type[Indexer] = Indexer, max_run_length: int | None = None):
        """
        Initializes the ReedSolomon instance.

        Args:
            parity_bytes: The number of Reed-Solomon parity bytes to append.
        """
        # Need a hard cap on the capacity. We need to represent them in a fixed width scheme for our dna, and we are 
        #   using a byte representation combination to achieve this. Becomes computationally expensive and impractical.
        if not 1 <= parity_bytes <= 8:
            raise ValueError("parity_bytes must be between 1 and 8.")

        self.parity_bytes: int = parity_bytes
        self.RSCodec: reedsolo.RSCodec = reedsolo.RSCodec(parity_bytes)

        # Going to encode each base as the byte we have - so maxium of 256 numbers per byte
        self.indexer: Indexer = indexer(
            width=indexer.determine_optimal_width(2**8),
            max_run_length=max_run_length
        )


    def _to_bytes(self, encoded_data: str) -> bytes:
        """
        Converts a DNA sequence to bytes with virtual padding.

        Args:
            data: The DNA sequence to be converted.

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
        byte_stream: bytes = bytes()
        for index in range(0, len(bit_stream), 8):
            byte: str = bit_stream[index:index+8]
            byte_stream += bytes([int(byte, 2)])

        return byte_stream


    def _encode(self, encoded_data: str) -> list[int]:
        """
        Encodes the given data using Reed-Solomon error correction.

        Args:
            encoded_data: The data to be encoded with parity. We assume its the [index | payload | checksum] sequence.

        Returns:
            The list of Reed-Solomon parity bytes.
        """
       
        byte_stream: bytes = self._to_bytes(encoded_data)

        # Return only the parity bytes
        return list(self.RSCodec.encode(byte_stream)[-self.parity_bytes:])


    def encode_to_dna(self, encoded_data: str, before_sequence: str, after_sequence: str) -> str:
        """
        Converts the Reed-Solomon parity bytes integer to a DNA sequence.

        Args:
            encoded_data: The data to be encoded with parity. We assume its the [index | payload | checksum] sequence.
            before_sequence: The sequence before the parity bytes in the oligo.
            after_sequence: The sequence after the parity bytes in the oligo.

        Returns:
            The corresponding DNA sequence to the .
        """
        parity_bytes: list[int] = self._encode(encoded_data)
        dna_sequence: str = ""

        for byte in parity_bytes:
            dna_sequence += self.indexer.unrank(byte, 
                                                before=before_sequence, 
                                                after=after_sequence)
            # Update the before_sequence for the next byte
            before_sequence = dna_sequence 

        return dna_sequence


    def _decode(self, encoded_data: str, parity_bytes: bytes) -> str:
        """
        Decodes the given data using Reed-Solomon error correction.

        Args:
            encoded_data: The data to be decoded with parity. We assume its the [index | payload | checksum] sequence.

        Returns:
            The decoded fixed data as a string.

        Raises:
            ReedSolomonError: If the data cannot be corrected.
        """
        byte_stream: bytes = self._to_bytes(encoded_data) + parity_bytes

        items: list[list[int]] = []
        for item in self.RSCodec.decode(byte_stream):
            items.append(list(item))

        decoded_bytes: bytes = bytes(items[0])

        # Convert the decoded bytes back to a DNA sequence
        bit_stream: str = "".join(f"{byte:08b}" for byte in decoded_bytes)
        decoded_sequence: str = ""
        for index in range(0, len(bit_stream), 2):
            bits: str = bit_stream[index:index+2]
            decoded_sequence += Encoding.DNA_MAP[bits]

        return decoded_sequence


    def decode_from_dna(self, 
                        encodeded_data: str, 
                        parity_dna_sequence: str, 
                        before_sequence: str, after_sequence: str) -> str:
        """
        Converts a DNA sequence back to the corresponding Reed-Solomon parity bytes integer.

        Args:
            encodeded_data: The DNA sequence to be converted back to Reed-Solomon parity bytes.
            parity_dna_sequence: The DNA sequence representing the Reed-Solomon parity bytes.
            before_sequence: The sequence before the parity bytes in the oligo.
            after_sequence: The sequence after the parity bytes in the oligo.

        Returns:
            The corresponding Reed-Solomon parity bytes integer.
        """
        parity_bytes: bytes = bytes()

        # Each parity byte is represented by a DNA sequence of length equal to the indexer's width
        for index in range(self.parity_bytes):
            # Get the exact DNA sequence for the current parity byte
            start: int = index * self.indexer.width
            end: int = start + self.indexer.width
            current_sequence: str = parity_dna_sequence[start:end]

            parity_byte: int = self.indexer.rank(current_sequence, 
                                                 before=before_sequence, 
                                                 after=after_sequence)

            # Update new start sequence for the next parity byte to be decoded
            before_sequence = current_sequence
            
            parity_bytes += bytes([parity_byte])

        decoded_sequence: str = self._decode(encodeded_data, parity_bytes)
    
        return decoded_sequence


    def get_dna_bases(self) -> int:
        """
        Gets the length of the Reed-Solomon parity bytes in bases.

        Returns:
            int: The length of the Reed-Solomon parity bytes in bases.
        """
        return self.indexer.width


class Hedges:
    pass
