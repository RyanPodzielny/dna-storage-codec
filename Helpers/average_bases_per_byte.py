"""Measure DNA bases per original byte for the Canterbury test files."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

# The script lives in Helpers/, so its parent directory is the project root. Need it to open methods.
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import BitScanQuaternary, ComplementaryTree, Goldman, Huffman, Pgram, Quaternary, Ternary
from Serialize.EncoderSerializer import Serializer


class EncodingRun:
    """
    Describe one encoder and its optional parameter.

    Args:
        name: The name written to the output for this encoder.
        parameter: The Pgram length or BitScanQuaternary scan length, if applicable.
        create_encoder: A factory that creates a fresh encoder instance.
        uses_binary_input: Whether the encoder expects the source as binary text.
    """


    def __init__(self, 
                 name: str, 
                 parameter: int | None, 
                 create_encoder: Callable[[], Encoding], 
                 uses_binary_input: bool = False):
        self.name: str = name
        self.parameter: int | None = parameter
        self.create_encoder: Callable[[], Encoding] = create_encoder
        self.uses_binary_input: bool = uses_binary_input



class Measurement:
    """
    Store one file's encoded size and its bases-per-byte rate.

    Args:
        file_name: The name of the original input file.
        method: The encoding method used.
        parameter: The method parameter, if one was used.
        original_bytes: The size of the original file in bytes.
        encoded_bases: The number of DNA bases in the encoded payload.
    """


    def __init__(self, file_name: str, method: str, parameter: int | None, original_bytes: int, encoded_bases: int):
        self.file_name: str = file_name
        self.method: str = method
        self.parameter: int | None = parameter
        self.original_bytes: int = original_bytes
        self.encoded_bases: int = encoded_bases


    def bases_per_byte(self) -> float:
        """
        Return the encoded bases divided by the original byte count.

        Returns:
            The number of encoded DNA bases per original byte.

        Raises:
            ZeroDivisionError: If the original file is empty.
        """
        return self.encoded_bases / self.original_bytes



class Summary:
    """
    Store cross-file statistics for one encoding configuration.

    Args:
        method: The encoding method being summarized.
        parameter: The method parameter, if one was used.
        average: The average bases-per-byte rate.
        minimum: The smallest bases-per-byte rate.
        maximum: The largest bases-per-byte rate.
    """


    def __init__(self, method: str, parameter: int | None, average: float, minimum: float, maximum: float):
        self.method: str = method
        self.parameter: int | None = parameter
        self.average: float = average
        self.minimum: float = minimum
        self.maximum: float = maximum



def create_encoding_runs() -> list[EncodingRun]:
    """
    Create every encoding configuration required by the test suite.

    Returns:
        A list containing the fixed encoders, Pgrams 2 through 16, and
        BitScanQuaternary values 4 through 52.
    """
    # These encoders do not require a parameter.
    runs: list[EncodingRun] = [
        EncodingRun(Huffman.__name__, None, Huffman),
        EncodingRun(Quaternary.__name__, None, Quaternary),
        EncodingRun(Ternary.__name__, None, Ternary),
        EncodingRun(ComplementaryTree.__name__, None, ComplementaryTree),
        EncodingRun(Goldman.__name__, None, Goldman),
        EncodingRun(DNAMap.__name__, None, DNAMap),
        EncodingRun(DNAMapHuffman.__name__, None, DNAMapHuffman),
    ]

    # Pgrams needs to be tested for lengths 2 through 16 (like in paper)
    runs.extend(
        EncodingRun(
            Pgram.__name__,
            pgram_length,
            lambda length=pgram_length: Pgram(length),
        )
        for pgram_length in range(2, 17)
    )
    # BitScanQuaternary needs to be tested for scan lengths 4 through 52 (like in paper)
    runs.extend(
        EncodingRun(
            BitScanQuaternary.__name__,
            scan_length,
            lambda length=scan_length: BitScanQuaternary(length),
            uses_binary_input=True,
        )
        for scan_length in range(4, 53)
    )
    return runs


def count_encoded_bases(encoded_payload: str | list[str]) -> int:
    """
    Count DNA bases represented by an encoded JSON payload.

    Binary payloads use two bits per DNA base. Payloads already containing DNA symbols use one character per base.

    Args:
        encoded_payload: An encoded string or a list of encoded segments.

    Returns:
        The number of DNA bases represented by the payload.

    Raises:
        ValueError: If the payload contains unsupported symbols.
    """
    # Segmented encoders store a list; joining it gives the complete payload.
    payload: str
    if isinstance(encoded_payload, str):
        payload = encoded_payload
    else:
        payload = "".join(encoded_payload)

    # If the payload is empty, return 0 bases
    if not payload:
        return 0

    # Count the number of DNA bases represented by the payload. Binary payloads use two bits per base, while DNA 
    #   payloads use one character per base
    if set(payload) <= Encoding.BINARY_SYMBOLS:
        return (len(payload) + 1) // 2
    if set(payload) <= Encoding.DNA_BASES:
        return len(payload)
    raise ValueError("Encoded payload contains symbols that cannot represent DNA bases.")


def measure_encoding(source_file: Path, run: EncodingRun, temporary_directory: Path) -> Measurement:
    """
    Encode one source file and count its encoded DNA bases.

    Args:
        source_file: The original file being measured.
        run: The encoding configuration to execute.
        temporary_directory: Directory for the temporary encoded JSON file.

    Returns:
        A measurement containing the encoded size and rate.

    Raises:
        FileNotFoundError: If the selected input file does not exist.
        ValueError: If the encoder writes an invalid payload.
    """
    encoder: Encoding = run.create_encoder()

    # Each run gets a separate file, but the temporary directory is removed automatically when all files have been measured.
    output_file: Path = temporary_directory / f"{run.name}_{run.parameter}.json"
    payload: str
    _, payload = Serializer.encode_as_single_segment(
        encoder=encoder,
        in_path=str(source_file),
        json_path=str(output_file),
        as_binary=run.uses_binary_input,
    )

    # Count the number of DNA bases represented by the payload and return a measurement
    return Measurement(
        file_name=source_file.name,
        method=run.name,
        parameter=run.parameter,
        original_bytes=source_file.stat().st_size,
        encoded_bases=count_encoded_bases(payload),
    )


def canterbury_files(directory: Path) -> list[Path]:
    """
    Return sorted, visible regular files in a test directory.

    Args:
        directory: The directory containing the source files.

    Returns:
        A sorted list of source file paths.
    """
    # Get each file in the director
    paths: list[Path] = []
    for path in directory.iterdir():
        if path.is_file() and not path.name.startswith("."):
            paths.append(path)

    # Sort the files by name so the output is consistent across runs
    paths = sorted(paths, key=lambda file: file.name)

    return paths


def create_summaries(measurements: list[Measurement]) -> dict[tuple[str, int | None], Summary]:
    """
    Calculate average, minimum, and maximum rates per configuration.

    Args:
        measurements: The individual file measurements to group.

    Returns:
        A dictionary keyed by method and parameter.
    """
    # Group the measurements by method and parameter, then calculate the average, minimum, and maximum bases-per-byte 
    grouped_values: dict[tuple[str, int | None], list[float]] = {}
    for measurement in measurements:
        configuration: tuple[str, int | None] = (measurement.method, measurement.parameter)

        # Add the bases-per-byte rate to the list for this configuration
        if configuration not in grouped_values:
            grouped_values[configuration] = []
        grouped_values[configuration].append(measurement.bases_per_byte())

    # Each group has at least one measurement because it was created by a measurement.
    configurations: dict[tuple[str, int | None], Summary] = {}
    for configuration, values in grouped_values.items():
        configurations[configuration] = Summary(
            method=configuration[0],
            parameter=configuration[1],
            average=sum(values) / len(values),
            minimum=min(values),
            maximum=max(values),
        )
        
    return configurations


def write_measurements(measurements: list[Measurement],
                       summaries: dict[tuple[str, int | None], Summary],
                       output_path: Path) -> None:
    """Write detailed measurements and repeated summary values as CSV.

    Args:
        measurements: The per-file results to write.
        summaries: Cross-file statistics keyed by method and parameter.
        output_path: The path where the CSV data will be saved.
    """
    # Write out each measurement with its summary values. The summary values are repeated for each file to make the CSV 
    #   easier to analyze in a spreadsheet.
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "file",
                "method",
                "parameter",
                "original_bytes",
                "encoded_bases",
                "bases_per_byte",
                "average_bases_per_byte",
                "min_bases_per_byte",
                "max_bases_per_byte",
            ]
        )
        # Print them all pretty
        for measurement in measurements:
            summary: Summary = summaries[(measurement.method, measurement.parameter)]
            writer.writerow(
                [
                    measurement.file_name,
                    measurement.method,
                    "" if measurement.parameter is None else measurement.parameter,
                    measurement.original_bytes,
                    measurement.encoded_bases,
                    f"{measurement.bases_per_byte():.6f}",
                    f"{summary.average:.6f}",
                    f"{summary.minimum:.6f}",
                    f"{summary.maximum:.6f}",
                ]
            )


def write_terminal_table(summaries: list[Summary], output_path: Path | None = None) -> None:
    """
    Write a readable summary table to the terminal.

    Args:
        summaries: The cross-file statistics to display.
        output_path: The path where the table will be saved. If omitted, the
            table is printed to ``sys.stdout``.
    """
    # Make rows look pretty
    rows: list[list[str]] = [
        [
            summary.method,
            "" if summary.parameter is None else str(summary.parameter),
            f"{summary.average:.6f}",
            f"{summary.minimum:.6f}",
            f"{summary.maximum:.6f}",
        ]
        for summary in summaries
    ]

    # Make the table look nice by aligning the columns
    headers: list[str] = ["Method", "Parameter", "Average", "Minimum", "Maximum"]

    # Get the maximum width of each column by checking the header and all rows
    column_widths: list[int] = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in rows:
            width = max(width, len(row[index]))
        column_widths.append(width)

    # Calculate column widths first, then pad each value so the separators align.
    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(column_widths[index])
            for index, value in enumerate(row)
        )

    # Make a separator line that matches the column widths. The "-+-" is a visual separator between columns
    separator: str = "-+-".join("-" * width for width in column_widths)
    table_lines: list[str] = [format_row(headers), separator]
    table_lines.extend(format_row(row) for row in rows)
    table: str = "\n".join(table_lines)

    # No path means the caller wants the table on the terminal.
    if output_path is None:
        print(table)
    else:
        output_path.write_text(table + "\n", encoding="utf-8")


def main() -> None:
    """
    Measure all requested encoders for every file in the selected directory.

    Raises:
        FileNotFoundError: If the selected source directory does not exist.
    """

    source_directory: Path = (PROJECT_ROOT / "Testing" / "Original" / "canterbury").resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_directory}"
        )

    measurements: list[Measurement] = []
    encoding_runs: list[EncodingRun] = create_encoding_runs()

    # Temporary files prevent this measurement script from overwriting the project's existing encoded and decoded tests
    with TemporaryDirectory() as temporary_directory_name:
        temporary_directory: Path = Path(temporary_directory_name)

        # For each source file, run every encoder and measure the bases-per-byte rate
        for source_file in canterbury_files(source_directory):
            # Perform each encoding with its mesaurement
            for run in encoding_runs:
                measurement: Measurement = measure_encoding(
                    source_file,
                    run,
                    temporary_directory,
                )

                measurements.append(measurement)

    # Create summaries for each encoder configuration and save the CSV output.
    summaries: dict[tuple[str, int | None], Summary] = create_summaries(measurements)
    # Save under data dir
    data_directory: Path = Path(__file__).parent / "Data"
    data_directory.mkdir(exist_ok=True)
    write_measurements(
        measurements,
        summaries,
        data_directory / "Measurements.csv",
    )

    # CSV is useful for later analysis. The terminal table is convenient for quickly comparing encoder configurations
    #   after a run completes.
    write_terminal_table(list(summaries.values()))

    # Write the table to a normal text file
    write_terminal_table(
        list(summaries.values()),
        data_directory / "measurement_summary.txt",
    )

if __name__ == "__main__":
    main()