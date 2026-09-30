"""
This module contains functions to test the encoding and decoding of files using Huffman coding and DNA mapping.
"""

import os
from Encoding import Encoding
from HuffmanCoding import Huffman, HuffmanBitScanQuaternary, HuffmanQuaternary, HuffmanTrenary
from DNAMapping import DNAMap, DNAMapHuffman

def check_match(original_file: str, decoded_file: str) -> bool:
    """
    Checks if the contents of the original file and the decoded file match.

    Args:
        original_file: The path to the original file.
        decoded_file: The path to the decoded file.

    Returns:
        True if the contents match, False otherwise.

    Raises:
        FileNotFoundError: If either the original or decoded file does not exist.
    """
    with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
        original_data: str = f.read()
    with open(decoded_file, 'r', encoding='utf-8', errors='ignore') as f:
        decoded_data: str = f.read()

    return original_data == decoded_data


def test_encoding(in_path: str, encoded_dir: str, decoded_dir: str, encoding: Encoding, 
                  original_size: int, origin_path: str = "", as_binary: bool = False) -> str:
    """
    Tests the encoding and decoding of a single file using Huffman coding and DNA mapping.

    Args:
        in_path: The path to the input file to be encoded and decoded.
        encoded_dir: The directory to store the encoded file.
        decoded_dir: The directory to store the decoded file.
        encoding: An instance of the Encoding class (Huffman, HuffmanQuaternary, DNAMap, or DNAMapHuffman).
        original_size: The size of the original file in bits.
        origin_path: The path to the original file for comparison. If not provided, in_path will be used.
        convert_to_binary: A boolean indicating whether to convert the original file to binary before encoding.

    Returns:
        The path to the encoded file.

    Raises:
        ValueError: If the original and decoded files do not match after encoding and decoding.
    """
    if origin_path == "":
        origin_path = in_path

    # Create the encoded and decoded paths
    encoded_path: str = os.path.join(encoded_dir, encoding.__class__.__name__ + "_encoded.txt")
    decoded_path: str = os.path.join(decoded_dir, encoding.__class__.__name__ + "_decoded.txt")

    encoded_data = encoding.encode_from_file(in_path, encoded_path, as_binary=as_binary)
    encoding.decode_to_file(encoded_path, decoded_path, as_binary=as_binary)

    # Check if both match
    if not check_match(origin_path, decoded_path):
        raise ValueError(f"Encoding and decoding failed for {in_path} in {encoding.__class__.__name__}.")

    encoding.print_info(encoded_data, original_size)

    return encoded_path


def test_file(original_dir: str, encoded_dir: str, decoded_dir: str, file_name: str) -> None:
    """
    Tests the encoding and decoding of a single file using Huffman coding and DNA mapping.

    Args:
        original_dir: The directory containing the original file.
        encoded_dir: The directory to store the encoded file.
        decoded_dir: The directory to store the decoded file.
        file_name: The name of the input file to be encoded and decoded.

    Returns:
        None

    Raises:
        ValueError: If the original and decoded files do not match after encoding and decoding.
        FileNotFoundError: If the input file does not exist.
    """
    # Strip the file extension for creating subdirectories
    file_name_no_ext: str = os.path.splitext(file_name)[0]

    # Create a folder in encoded and decoded directories for each file
    file_encoded_dir: str = os.path.join(encoded_dir, file_name_no_ext)
    os.makedirs(file_encoded_dir, exist_ok=True)
    file_decoded_dir: str = os.path.join(decoded_dir, file_name_no_ext)
    os.makedirs(file_decoded_dir, exist_ok=True)

    origin: str = os.path.join(original_dir, file_name)
    # Convert bytes to bits
    original_size: int = os.path.getsize(origin) * 8  
    print(f"File: {file_name} - Original Size: {original_size} bits")

    # Test Huffmans
    huffman_encoded_path: str = test_encoding(origin, file_encoded_dir, file_decoded_dir, Huffman(), original_size)
    test_encoding(origin, file_encoded_dir, file_decoded_dir, HuffmanQuaternary(), original_size)
    test_encoding(origin, file_encoded_dir, file_decoded_dir, HuffmanTrenary(), original_size)

    # Find optimal k for HuffmanBitScanQuaternary and test encoding/decoding
    optimal_k: int = HuffmanBitScanQuaternary().find_optimal_k(origin)[0]
    test_encoding(origin, file_encoded_dir, file_decoded_dir, HuffmanBitScanQuaternary(k=optimal_k), original_size, 
                  as_binary=True)

    # Before testing DNA mapping, ensure that the Huffman encoded file exists
    if not os.path.exists(huffman_encoded_path):
        raise FileNotFoundError(f"Huffman encoded file {huffman_encoded_path} doesn't exist. Please run Huffman first.")

    # Test mapping
    test_encoding(origin, file_encoded_dir, file_decoded_dir, DNAMap(), original_size)
    test_encoding(huffman_encoded_path, file_encoded_dir, file_decoded_dir, DNAMapHuffman(), 
                  original_size, origin_path=origin)

    print(); print()


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

        test_file(original_dir, encoded_dir, decoded_dir, file_name)


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