"""
This module tests framing and deframing files with every combination of encoder, indexer, inner code, and outer code.
Each combination is framed with a FramerConfig, saved with the Serializer, then decoded from only the saved file to
check that it can be recovered.
"""


import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import Goldman, BitScanQuaternary, Pgram, Quaternary, Ternary
from Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Encode.Indexer import DNAMapIndexer, EnumerativeIndexer, GoldmanIndexer, Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, RSInner
from Encode.ErrorCorrection.OuterCodes import DirectDNAGoldmanRedundancy, DirectDNARSOuter, DirectDNAXOR, \
    GoldmanRedundancy, OuterCodes, RSOuter, XOR
from Encode.OligoFramer import FramerConfig, OligoDeframer, OligoFramer
from Serialize.FileSerializer import Serializer


#--------------------------------------------------------------------------#
#                                                                          #
#   SETTINGS                                                               #
#                                                                          #
#--------------------------------------------------------------------------#


# Our testing directories 
TESTING_DIR: Path = Path("Testing")
ORIGINAL_DIR: Path = TESTING_DIR / "Original"
ENCODED_DIR: Path = TESTING_DIR / "Encoded"
DECODED_DIR: Path = TESTING_DIR / "Decoded"
RESULTS_DIR: Path = TESTING_DIR / "Results"

# Same oligo settings as test.py
OLIGO_LENGTH: int = 200
OVERHEAD_MARGIN: float = 0.1
PRIMER: str = "ACGTACGT"
MAX_RUN_LENGTH: int = 1

# Same inner and outer code settings as test.py
INNER_PARITY_BYTES: int = 4
OUTER_REDUNDANCY: float = 0.2

# Only one m for Pgram, and the optimal k is found per file for BitScanQuaternary
PGRAM_M: int = 8

# Runs of at least this length are counted when printing info
RUN_LENGTH_K: int = 3

# The saved framer files take up most of the space (gigabytes over every combination). When False, a combination's
#   saved file is deleted once it decodes 100%, so only the ones that failed are kept to look into
KEEP_PASSING_ENCODED: bool = False

# Encoders whose payloads are homopolymer free, these can use the normal outer codes. Every other encoder can produce
#   homopolymers, so it needs the DirectDNA outer codes
HOMOPOLYMER_FREE: tuple[type, ...] = (Goldman, Pgram)

# Indexers by name, each takes the width (set by the framer) and the max run length. Using lambda to avoid creating the 
#   indexers until they are needed, as some need to find the optimal k for the file
INDEXERS: dict[str, Callable[[], Indexer]] = {
    "Enumerative": lambda: EnumerativeIndexer(width=-1, max_run_length=MAX_RUN_LENGTH),
    "Goldman": lambda: GoldmanIndexer(width=-1),
    "DNAMap": lambda: DNAMapIndexer(width=-1),
}

# Outer codes by name, as (normal, DirectDNA). None means no outer code. Again lambda to avoid creating until needed.
OUTER_CODES: dict[str, tuple[Callable[[], OuterCodes], Callable[[], OuterCodes]] | None] = {
    "NoOuter": None,
    "GoldmanRedundancy": (GoldmanRedundancy, DirectDNAGoldmanRedundancy),
    "XOR": (XOR, DirectDNAXOR),
    "RSOuter": (lambda: RSOuter(redundancy=OUTER_REDUNDANCY), lambda: DirectDNARSOuter(redundancy=OUTER_REDUNDANCY)),
}


#--------------------------------------------------------------------------#
#                                                                          #
#   RESULTS                                                                #
#                                                                          #
#--------------------------------------------------------------------------#


@dataclass
class Result:
    """
    The outcome of framing and deframing one file with one combination.
    """
    file_name: str
    combination: str
    matches: bool = False
    recovered: float = 0.0
    error: str = ""


#--------------------------------------------------------------------------#
#                                                                          #
#   HELPERS                                                                #
#                                                                          #
#--------------------------------------------------------------------------#


def get_encoders(origin: str) -> list[Encoding]:
    """
    Makes one of every encoder that outputs DNA. Plain Huffman outputs bits, not bases, so it can't be framed.

    Args:
        origin: The path to the original file, used to find the optimal k for BitScanQuaternary.

    Returns:
        A list of encoders to test.
    """
    optimal_k: int = BitScanQuaternary().find_optimal_k(origin)[0]

    # All the encoders we have (besides standard huffman)
    return [
        Ternary(), 
        Quaternary(), 
        BitScanQuaternary(k=optimal_k), 
        Pgram(PGRAM_M), 
        Goldman(), 
        DNAMap(),
        DNAMapHuffman()
    ]


def get_encoder_name(encoder: Encoding) -> str:
    """
    Gets the name of an encoder for the file name, including its k or m if it has one.

    Args:
        encoder: The encoder.

    Returns:
        The encoder's class name, with its k or m appended (e.g. BitScanQuaternary-k6, Pgram-m8).
    """
    # Name is basically Encoder-(param) if there is param (only saved if failed with KEEP_PASSING_ENCODED = False)
    name: str = type(encoder).__name__
    if isinstance(encoder, BitScanQuaternary):
        name += f"-k{encoder.K}"
    elif isinstance(encoder, Pgram):
        name += f"-m{encoder.M}"

    return name


def make_config(origin: str, 
                encoder: Encoding, 
                indexer_name: str, 
                use_inner: bool,
                outer_name: str) -> tuple[str, FramerConfig]:
    """
    Makes the framer config for one combination, and the name used for its saved files.

    Args:
        origin: The path to the original file.
        encoder: The encoder to frame with.
        indexer_name: The name of the indexer in INDEXERS.
        use_inner: Whether to use an inner RS code.
        outer_name: The name of the outer code in OUTER_CODES.

    Returns:
        The combination's name and its config.
    """
    # Get the indexer instance and its type for the inner code
    indexer: Indexer = INDEXERS[indexer_name]()
    indexer_type: type = type(indexer)

    outer_correction: OuterCodes | None = None
    outer_codes: tuple[Callable[[], OuterCodes], Callable[[], OuterCodes]] | None = OUTER_CODES[outer_name]
    # Homopolymer producing encoders need the DirectDNA version of the outer code
    if outer_codes is not None:
        normal_outer, direct_outer = outer_codes
        outer_correction = normal_outer() if isinstance(encoder, HOMOPOLYMER_FREE) else direct_outer()

    # Add inner if requested
    inner_correction: RSInner | None = None
    if use_inner:
        inner_correction = RSInner(indexer_type, parity_bytes=INNER_PARITY_BYTES, max_run_length=MAX_RUN_LENGTH)

    # Create the config
    config: FramerConfig = FramerConfig(
        in_path=origin,
        oligo_length=OLIGO_LENGTH,
        overhead_margin=OVERHEAD_MARGIN,
        start_primer=PRIMER,
        end_primer=PRIMER,
        encoder=encoder,
        indexer=indexer,
        checksum=CheckSum(indexer_type, inner_code=use_inner, max_run_length=MAX_RUN_LENGTH),
        inner_correction=inner_correction,
        outer_correction=outer_correction
    )

    # Example name is just Ternary_Enumerative_RSInner_DirectDNARSOuter
    inner_name: str = "RSInner" if use_inner else "NoInner"
    outer_label: str = type(outer_correction).__name__ if outer_correction else "NoOuter"
    combination: str = f"{get_encoder_name(encoder)}_{indexer_name}_{inner_name}_{outer_label}"

    return combination, config


#--------------------------------------------------------------------------#
#                                                                          #
#   TESTS                                                                  #
#                                                                          #
#--------------------------------------------------------------------------#


def test_combination(origin: str, encoded_dir: Path, decoded_dir: Path, combination: str, config: FramerConfig,
                     original_size: int) -> Result:
    """
    Frames a file with one combination, saves it, then decodes it from only the saved file. To save space, a
    combination that decodes 100% writes to one decoded file shared by every combination of the file (overwriting it),
    and its saved framer file is deleted unless KEEP_PASSING_ENCODED is set. A combination that fails keeps both, named
    after the combination.

    Args:
        origin: The path to the original file.
        encoded_dir: The directory to save the framed file to.
        decoded_dir: The directory to save the decoded file to.
        combination: The combination's name, used for the file names.
        config: The config to frame with.
        original_size: The size of the original file in bits.

    Returns:
        The result of the test. Errors are caught and recorded so one combination can't stop the rest.
    """
    file_name: str = Path(origin).name
    result: Result = Result(file_name=file_name, combination=combination)

    encoded_path: Path = encoded_dir / f"{combination}.json"

    print(combination)
    try:
        # Frame and save
        start: float = time.perf_counter()
        framer: OligoFramer = OligoFramer(config)
        framer.frame_all_oligos()
        Serializer.save_framer(framer, str(encoded_path))

        # Start the clock
        encode_time: float = time.perf_counter() - start
        start = time.perf_counter()

        # Decode from the saved file, and check if it matches the original and how many oligos were recovered
        deframer: OligoDeframer = Serializer.load_deframer(str(encoded_path))
        decoded_oligos: list[str | None] = deframer.correct_all_oligos()
        content: str = deframer.decode_to_string(decoded_oligos)

        # STOP THE CLOCK
        decode_time: float = time.perf_counter() - start

        # Check if the decoded content matches the original, and how many oligos were recovered
        result.matches = deframer.matches_original(content)
        result.recovered = deframer.calculate_recovered(decoded_oligos)
        passed: bool = result.matches and result.recovered == 100.0

        # A full decode is the same as every other full decode, so they all share (and overwrite) one file named
        #   after the original. A failed decode gets its own file so it can be looked into
        decoded_path: Path = decoded_dir / (file_name if passed else f"{combination}.txt")
        with open(decoded_path, 'w', encoding='utf-8') as decoded_file:
            decoded_file.write(content)

        # The saved framer file isn't needed once it decodes fully
        if passed and not KEEP_PASSING_ENCODED:
            encoded_path.unlink()

        # Show off the results and info
        framer.print_info(original_size, RUN_LENGTH_K)
        print(f"\tEncode time: {encode_time:.2f}s, decode time: {decode_time:.2f}s")
        print(f"\tMatches original hash: {result.matches}")
        print(f"\tOligos recovered: {result.recovered:.2f}%")

    # Record the failure and keep going with the other combinations - will save the failed framer 
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
        print(f"\tFAILED - {result.error}")

    return result


def test_file(original_dir: Path, encoded_dir: Path, decoded_dir: Path, file_name: str) -> list[Result]:
    """
    Tests framing and deframing a single file with every combination.

    Args:
        original_dir: The directory containing the original file.
        encoded_dir: The directory to save the framed files to.
        decoded_dir: The directory to save the decoded files to.
        file_name: The name of the file to test.

    Returns:
        The result of each combination.
    """
    # Make a folder for the file in the encoded and decoded directories. Named with the extension (e.g. cp_html), as
    #   files can share a stem (cp.html and cp.txt) and would overwrite each other
    folder_name: str = file_name.replace(".", "_")
    file_encoded_dir: Path = encoded_dir / folder_name
    file_encoded_dir.mkdir(parents=True, exist_ok=True)
    file_decoded_dir: Path = decoded_dir / folder_name
    file_decoded_dir.mkdir(parents=True, exist_ok=True)

    origin: str = str(original_dir / file_name)
    # Convert bytes to bits
    original_size: int = Path(origin).stat().st_size * 8
    print(f"File: {file_name} - Original Size: {original_size} bits")

    # Preform tests on all combinations
    results: list[Result] = []
    # Get all the encoder types
    for encoder in get_encoders(origin):
        # Get all the indexer types
        for indexer_name in INDEXERS:
            # See if we want to use an inner code or not
            for use_inner in (True, False):
                # Get all the outer code types
                for outer_name in OUTER_CODES:

                    combination: str
                    config: FramerConfig

                    combination, config = make_config(
                        origin, 
                        encoder, 
                        indexer_name, 
                        use_inner, 
                        outer_name
                    )

                    results.append(test_combination(
                        origin, 
                        file_encoded_dir, 
                        file_decoded_dir, 
                        combination, 
                        config,
                        original_size)
                    )

    print(); print()
    return results


def test_directory(dir: str) -> list[Result]:
    """
    Tests framing and deframing every file in a directory of Testing/Original with every combination.

    Args:
        dir: The name of the directory in Testing/Original.

    Returns:
        The result of each file and combination.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    # Get our directories and make sure the original exists
    original_dir: Path = ORIGINAL_DIR / dir
    if not original_dir.exists():
        raise FileNotFoundError(f"Directory {original_dir} does not exist.")

    encoded_dir: Path = ENCODED_DIR / dir
    decoded_dir: Path = DECODED_DIR / dir

    results: list[Result] = []
    for file_path in sorted(original_dir.iterdir()):
        # Skip hidden files (e.g. .DS_Store) and directories
        if file_path.name.startswith(".") or file_path.is_dir():
            continue

        # Add the results of every combination for the file to the list
        results.extend(test_file(
            original_dir, 
            encoded_dir, 
            decoded_dir, 
            file_path.name)
        )

    return results


def print_summary(results: list[Result]) -> None:
    """
    Prints how many combinations passed, and every one that did not.

    Args:
        results: The result of each file and combination.

    Returns:
        None
    """
    # Add up all the results that passed and failed, a combination passes if it matches the original and has no errors
    passed: list[Result] = [result for result in results if result.matches and not result.error]
    failed: list[Result] = [result for result in results if not result.matches or result.error]

    # Print out the summary of the results, and every failed combination with its reason
    print("Summary")
    print(f"\tPassed: {len(passed)} / {len(results)}")
    print(f"\tFailed: {len(failed)} / {len(results)}")
    for result in failed:
        reason: str = result.error or f"hash did not match ({result.recovered:.2f}% recovered)"
        print(f"\t\t{result.file_name} - {result.combination}: {reason}")


"""
Simple logging to both console and file. Very bare bones, meant for testing.
"""
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def main() -> None:
    """
    Main function to run tests on framing and deframing the files in the specified directories and saves the results
    to a timestamped text file in the Testing/Results directory. Clears the old encoded and decoded files first so
    they don't get mixed up with the new ones.

    Returns:
        None
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    date_format: str = "%Y-%m-%d_%H:%M:%S"

    # Make file name timestampped
    start: str = datetime.now().strftime(date_format)
    output_file: str = str(RESULTS_DIR / f"Results-{start}.txt")
    with open(output_file, 'w', encoding='utf-8') as result_file:
        # Print both to console and to the file
        sys.stdout = Tee(sys.stdout, result_file)

        try:
            results: list[Result] = test_directory("canterbury")
            print_summary(results)

            # Add time information to the output file
            end: str = datetime.now().strftime(date_format)
            print(f"Start time: {start}")
            print(f"End time: {end}")

            # Show total time in seconds
            total_time = (datetime.strptime(end, date_format) - datetime.strptime(start, date_format)).total_seconds()
            print(f"Total time (seconds): {total_time}")
        finally:
            # Reset stdout to default
            sys.stdout = sys.__stdout__
            print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()