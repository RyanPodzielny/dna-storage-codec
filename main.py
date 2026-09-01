"""
This module contains functions to test the encoding and decoding of files using Huffman coding and DNA mapping.
"""

from HuffmanCoding import Huffman, HuffmanQuaternary
from DNAMapping import DNAEncodedMap
import os

def binary_to_ascii(binary_string: str) -> str:
    """
    Converts a binary string to an ASCII string.

    Args:
        binary_string: The binary string to be converted.

    Returns:
        An ASCII string representation of the binary string.
    """
    ascii_string: str = ""
    for i in range(0, len(binary_string), 8):
        byte: str = binary_string[i:i+8]
        ascii_string += chr(int(byte, 2))
    return ascii_string

def ascii_to_binary(ascii_string: str) -> str:
    """
    Converts an ASCII string to a binary string.

    Args:
        ascii_string: The ASCII string to be converted.

    Returns:
        A binary string representation of the ASCII string.
    """
    return ''.join(format(ord(char), '08b') for char in ascii_string)


def count_runs(data: str, k: int=3) -> int:
    """
    Counts the number of runs of DNA bases in a given string that are at least of length k.

    Args:
        data: The string containing DNA bases.
        k: The minimum length of a run to be counted. Default is 3.

    Returns:
        The number of runs of DNA bases in the string that are at least of length k.
    """
    count: int = 0
    current_run_length: int = 1
    current_run_char: str = data[0]

    # Iterate through the string and count runs
    for letter in data:
        # If the current letter is the same as the last, increment the run length
        if letter == current_run_char:
            current_run_length += 1
        # If the current letter is different, check if the run length is at least k and reset
        else:
            if current_run_length >= k:
                count += 1
            current_run_char = letter
            current_run_length = 1

    # Check the last run
    if current_run_length >= k:
        count += 1

    return count


def print_size_info(encoded_size: int, table_size: int, method_name: str) -> None:
    """
    Prints the size information for the original and encoded data.

    Args:
        encoded_size: The size of the encoded data in bits.
        table_size: The size of the encoding table in bits.
        method_name: The name of the encoding method used.

    Returns:
        None
    """
    print(f"{method_name} Encoding:")
    print(f"\tEncoded Size: {encoded_size} bits")
    print(f"\tTable Size: {table_size} bits")


def check_match(original_file: str, decoded_file: str) -> bool:
    """
    Checks if the contents of the original file and the decoded file match.

    Args:
        original_file: The path to the original file.
        decoded_file: The path to the decoded file.

    Returns:
        True if the contents match, False otherwise.
    """
    with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
        original_data: str = f.read()
    with open(decoded_file, 'r', encoding='utf-8', errors='ignore') as f:
        decoded_data: str = f.read()

    return original_data == decoded_data


def test_huffman(in_path: str, encoded_dir: str, decoded_dir: str, huffman: Huffman) -> None:
    """
    Tests the encoding and decoding of a single file using Huffman coding and DNA mapping.

    Args:
        in_path: The path to the input file to be encoded and decoded.
        encoded_dir: The directory to store the encoded file.
        decoded_dir: The directory to store the decoded file.

    Returns:
        None

    Raises:
        ValueError: If the original and decoded files do not match after encoding and decoding.
    """

    # Create the encoded and decoded paths
    encoded_path: str = os.path.join(encoded_dir, huffman.__class__.__name__ + "_encoded.txt")
    decoded_path: str = os.path.join(decoded_dir, huffman.__class__.__name__ + "_decoded.txt")

    encoded_data = huffman.encode_from_file(in_path, encoded_path)
    huffman.decode_to_file(encoded_path, decoded_path)

    # Check if both match
    if not check_match(in_path, decoded_path):
        raise ValueError(f"Huffman encoding and decoding failed for {in_path}.")

    encoded_size: int
    table_size: int
    encoded_size, table_size = huffman.calculate_size(encoded_data)
    print_size_info(encoded_size, table_size, huffman.__class__.__name__)


def test_dna_huffman_mapping(in_path: str, encoded_dir: str, decoded_dir: str) -> None:
    """
    Tests the encoding and decoding of a single file using DNA mapping.

    Args:
        in_path: The path to the input file to be encoded and decoded.
        encoded_dir: The directory to the encoded file.
        decoded_dir: The directory to the decoded file.

    Returns:
        None

    Raises:
        ValueError: If the original and decoded files do not match after encoding and decoding.
        FileNotFoundError: If the Huffman encoded file does not exist.
    """

    # Grab the encoded huffman file
    huffman_encoded_path: str = os.path.join(encoded_dir, "Huffman_encoded.txt")
    if not os.path.exists(huffman_encoded_path):
        raise FileNotFoundError(f"Huffman encoded file {huffman_encoded_path} doesn't exist. Please run Huffman first.")

    # Now, encode the Huffman encoded file using DNA mapping
    dna_mapping = DNAEncodedMap()
    dnamap_encoded_path: str = os.path.join(encoded_dir, "DNAMap_encoded.txt")
    encodeded_data = dna_mapping.encode_to_file(huffman_encoded_path, dnamap_encoded_path, is_huffman=True)

    # Decode the DNA mapped file back to Huffman encoded file
    dnamap_decoded_path: str = os.path.join(decoded_dir, "DNAMap_decoded.txt")
    dna_mapping.decode_to_file(dnamap_encoded_path, dnamap_decoded_path)
 
    # Check if both match
    if not check_match(in_path, dnamap_decoded_path):
        raise ValueError(f"DNA mapping encoding and decoding failed for {in_path}.")

    encoded_size: int
    table_size: int
    encoded_size, table_size = dna_mapping.calculate_size(encodeded_data)
    print_size_info(encoded_size, table_size, "DNA Huffman Mapping")



def test_dna_standard_mapping(binary_string: str, in_path: str, encoded_dir: str, decoded_dir: str) -> None:
    """
    Tests the encoding and decoding of a single file using standard DNA mapping.

    Args:
        binary_string: The binary string to be encoded and decoded.
        in_path: The path to the input file to be encoded and decoded.
        encoded_dir: The directory to the encoded file.
        decoded_dir: The directory to the decoded file.

    Returns:
        None
    """
    # Encode the original file using standard DNA mapping
    dna_mapping = DNAEncodedMap()
    dnamap_encoded_path: str = os.path.join(encoded_dir, "DNAMap_standard_encoded.txt")



    """
    TODO: fix this mess
    """
    has_padding: bool
    dna_string: str
    has_padding, dna_string = dna_mapping.encode(binary_string)
    # Format the output to include the DNA string, the Huffman table (if applicable), and the padding indicator
    output: str = dna_string + "\n"
    output += Huffman().SEPARATOR + ("1" if has_padding else "0")
    with open(dnamap_encoded_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(output)

    # Decode the DNA mapped file back to original file
    dnamap_decoded_path: str = os.path.join(decoded_dir, "DNAMap_standard_decoded.txt")
    binary: str = dna_mapping.decode_to_file(dnamap_encoded_path, dnamap_decoded_path)
    with open(dnamap_decoded_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(binary_to_ascii(binary))



 
    # Check if both match
    if not check_match(in_path, dnamap_decoded_path):
        raise ValueError(f"Standard DNA mapping encoding and decoding failed for {in_path}.")

    encoded_size: int
    table_size: int
    encoded_size, table_size = dna_mapping.calculate_size(output)
    print_size_info(encoded_size, table_size, "Standard DNA Mapping")


def test_directory(dir: str) -> None:
    """
    Tests the encoding and decoding of files in a given directory using Huffman coding and DNA mapping.

    Args:
        dir: The path to the directory containing the files to be tested.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """

    # See if directory exists in Testing/Original
    original_dir: str = os.path.join("Testing", "Original", dir)
    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"Directory {original_dir} does not exist.")

    # Create a directory for the encoded and decoded files - if already exists, it will not raise an error
    encoded_dir: str = os.path.join("Testing", "Encoded", dir)
    os.makedirs(encoded_dir, exist_ok=True)
    decoded_dir: str = os.path.join("Testing", "Decoded", dir)
    os.makedirs(decoded_dir, exist_ok=True)

    # Get all file names in the original directory
    file_names: list[str] = os.listdir(original_dir)

    # Go through each file and test encoding and decoding for Huffman, Quaternary Huffman, and DNA mapping
    for file_name in file_names:
        # Skip .DS_Store or any hidden files
        if file_name.startswith('.'):
            continue
        # Skip directories, we only want to test files
        if os.path.isdir(os.path.join(original_dir, file_name)):
            continue

        # Strip the file extension for creating subdirectories
        file_name_no_ext: str = os.path.splitext(file_name)[0]

        # Create a folder in encoded and decoded directories for each file
        file_encoded_dir: str = os.path.join(encoded_dir, file_name_no_ext)
        os.makedirs(file_encoded_dir, exist_ok=True)
        file_decoded_dir: str = os.path.join(decoded_dir, file_name_no_ext)
        os.makedirs(file_decoded_dir, exist_ok=True)

        original_file: str = os.path.join(original_dir, file_name)

        # Convert to UTF-8 if not already
        with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
            data: str = f.read() 
            binary_data: str = ascii_to_binary(data)
            orginal_size: int = len(binary_data)
            print(f"File: {file_name} - Original Size: {orginal_size} bits")

        # Standard Huffman
        test_huffman(original_file, file_encoded_dir, file_decoded_dir, Huffman())
        test_huffman(original_file, file_encoded_dir, file_decoded_dir, HuffmanQuaternary())
        test_dna_huffman_mapping(original_file, file_encoded_dir, file_decoded_dir)
        test_dna_standard_mapping(binary_data, original_file, file_encoded_dir, file_decoded_dir)

        print(); print()


def main() -> None:
    """
    Main function to run tests on the encoding and decoding of files in the specified directories.

    Returns:
        None
    """
    # Test the directories
    test_directory("canterbury")


if __name__ == "__main__":
    main()