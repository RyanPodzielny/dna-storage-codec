from crc import Calculator, Configuration, Crc16, Crc32
import reedsolo
from Encode.Indexer import Indexer


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

        # Default to CRC-32 for checksum calculation if we have an inner_code, use CRC-16 instead. The config values are
        #   from https://pypi.org/project/crc/
        self.config: Configuration = Crc32.CRC32.value
        if self.inner_code:
            self.config = Crc16.XMODEM.value

        self.crc: Calculator = Calculator(self.config)

        # Bit shift left to get 2^required_width, which is number of unique values we can have
        self.indexer: Indexer = indexer(
            width=indexer.determine_optimal_width(1 << self.config.width),
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

        return self.config.width


    def calculate_checksum(self, data: str) -> int:
        """
        Calculates the checksum for the given data.

        Args:
            data: The index + payload data for which the checksum is to be calculated.

        Returns:
            int: The calculated checksum.
        """
        # Convert data into bytes
        byte_stream: bytes = data.encode('utf-8')

        return self.crc.checksum(byte_stream)


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


    def verify_checksum(self, data: str) -> bool:
        """
        Verifies the checksum of the provided data.

        Args:
            data: The data string with the appended checksum.

        Returns:
            bool: True if the checksum is valid, False otherwise.
        """
        # Extract the original data and the provided checksum
        provided_checksum: str = data[-self.get_checksum_length():]
        original_data: str = data[:-self.get_checksum_length()]

        # Calculate the checksum for the original data
        calculated_checksum: int = self.calculate_checksum(original_data)

        # Compare the calculated checksum with the provided checksum
        return str(calculated_checksum) == provided_checksum


    def encode_to_dna(self, checksum: int, primer: str, payload: str) -> str:
        """
        Converts the checksum integer to a DNA sequence.

        Args:
            checksum: The checksum integer to be converted.
            primer: The primer sequence.
            payload: The payload sequence.
            indexer: The indexer to use for the conversion.

        Returns:
            str: The corresponding DNA sequence.
        """
        # Convert the checksum integer to a DNA sequence using the indexer
        dna_sequence: str = self.indexer.unrank(checksum, primer=primer, payload=payload)

        return dna_sequence


    def decode_from_dna(self, dna_sequence: str, primer: str, payload: str) -> int:
        """
        Converts a DNA sequence back to the corresponding checksum integer.

        Args:
            dna_sequence: The DNA sequence to be converted.
            primer: The primer sequence.
            payload: The payload sequence.
            indexer: The indexer to use for the conversion.

        Returns:
            int: The corresponding checksum integer.
        """
        # Convert the DNA sequence back to the checksum integer using the indexer
        checksum: int = self.indexer.rank(dna_sequence, primer=primer, payload=payload)

        return checksum


class ReedSolomon:
    """
    A class for Reed-Solomon error correction. This is a placeholder for the actual implementation.
    """

    def __init__(self, symbols: int, information: int):
        """
        Initializes the ReedSolomon instance.

        Args:
            symbols: The total number of symbols in the codeword.
            information: The number of information symbols in the codeword.
        """
        self.symbols: int = symbols
        self.information: int = information


        self.RSCodec: reedsolo.RSCodec = reedsolo.RSCodec(nsym=self.symbols - self.information)