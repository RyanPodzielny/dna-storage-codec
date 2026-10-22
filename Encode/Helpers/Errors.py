class Error(Exception):
    """
    Base class to encompass all of custom errors"
    """
    def __init__(self, message: str = "Encoding error occurred"):
        super().__init__(message)


class JSONError(Error):
    """
    Error raised when a JSON file is invalid or cannot be read from its attributes.
    """
    def __init__(self, 
                 *,
                 message: str = "JSON error occurred",
                 json_path: str):
        self.json_path: str = json_path
        super().__init__(f"{message}: {json_path}")


class EncodingError(Error):
    """
    Error raised when an encoding or decoding error occurs.
    """
    def __init__(self, 
                 *,
                 message: str = "Encoding error occurred",
                 encoding_method: str):
        self.encoding_method: str = encoding_method
        super().__init__(message)


class IndexingError(Error):
    """
    Error raised when an indexing error occurs.
    """
    def __init__(self, 
                 *,
                 message: str = "Indexing error occurred",
                 index: int):
        self.message: str = message
        self.index: int = index
        super().__init__(f"{message}: Index {index} is not in bounds of what is needed.")


class PayloadLengthError(Error):
    """
    Error raised when the payload length is not valid (e.g., less than or equal to 0).
    """
    def __init__(self, 
                 *,
                 message: str = "Payload must be greater than 0",
                 payload_length: int):
        self.payload_length: int = payload_length
        super().__init__(f"{message}: Payload length {payload_length} is not valid.")


class UnrankError(Error):
    """
    Error raised when the index is not in bounds of the capacity.
    """
    def __init__(self, 
                 *,
                 message: str = "Index is not in bounds of capacity",
                 index: int,
                 capacity: int):
        self.index: int = index
        self.capacity: int = capacity
        super().__init__(f"{message}: Index {index} is not in bounds of capacity {capacity}.")


class RankError(Error):
    """
    Error raised when the index sequence length does not match the expected width.
    """
    def __init__(self, 
                 *,
                 message: str = "Index sequence length does not match expected width",
                 index_sequence: str,
                 width: int):
        self.index_sequence: str = index_sequence
        self.width: int = width
        super().__init__(f"{message}: {index_sequence} (length: {len(index_sequence)})"
                         f" does not match expected width {width}.")


class MaxRunLengthError(Error):
    """
    Error raised when a sequence exceeds the maximum run length.
    """
    def __init__(self, 
                 *,
                 message: str = "Sequence exceeds maximum run length",
                 sequence: str,
                 max_run_length: int):
        self.sequence: str = sequence
        self.max_run_length: int = max_run_length
        super().__init__(f"{message}: {sequence} exceeds maximum run length of {max_run_length}.")


class ExpectedLengthError(Error):
    """
    Error raised when a sequence length does not match the expected length.
    """
    def __init__(self, 
                 *,
                 message: str = "Sequence length does not match expected length",
                 sequence: str,
                 expected_length: int):
        self.sequence: str = sequence
        self.expected_length: int = expected_length
        super().__init__(f"{message}: {sequence} (length: {len(sequence)})"
                         f" does not match expected length {expected_length}.")


class InnerCodeError(Error):
    """
    Error raised when inner error correction fails to correct a sequence.
    """
    def __init__(self, 
                 *,
                 message: str = "Inner error correction failed",
                 sequence: str):
        self.sequence: str = sequence
        super().__init__(f"{message}: {sequence} could not be corrected by inner error correction.")