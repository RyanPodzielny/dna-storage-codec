"""
This module contains functions to test the encoding and decoding of files using Huffman coding and DNA mapping.
"""

from HuffmanCoding import Huffman, HuffmanQuaternary
from DNAMapping import DNAEncodedMap
import os


# Count the number of runs of dna bases in a string
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

    huffman.encode_from_file(in_path, encoded_path)
    huffman.decode_to_file(encoded_path, decoded_path)

    # Check if both match
    if not check_match(in_path, decoded_path):
        raise ValueError(f"Huffman encoding and decoding failed for {in_path}.")



def test_dna_mapping(in_path: str, encoded_dir: str, decoded_dir: str) -> None:
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
    dna_mapping.encode_to_file(huffman_encoded_path, dnamap_encoded_path, is_huffman=True)

    # Decode the DNA mapped file back to Huffman encoded file
    dnamap_decoded_path: str = os.path.join(decoded_dir, "DNAMap_decoded.txt")
    dna_mapping.decode_to_file(dnamap_encoded_path, dnamap_decoded_path)
 

    # Check if both match
    if not check_match(in_path, dnamap_decoded_path):
        raise ValueError(f"DNA mapping encoding and decoding failed for {in_path}.")
    

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

        # Standard Huffman
        test_huffman(original_file, file_encoded_dir, file_decoded_dir, Huffman())
        test_huffman(original_file, file_encoded_dir, file_decoded_dir, HuffmanQuaternary())
        test_dna_mapping(original_file, file_encoded_dir, file_decoded_dir)


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