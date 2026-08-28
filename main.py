from HuffmanCoding import Huffman, HuffmanQuaternary
from DNAMapping import DNAEncodedMap

# Count the number of runs of dna bases in a string
def count_runs(data: str, k: int=3) -> int:
    count: int = 0
    current_run_length: int = 1
    current_run_char: str = data[0]

    for letter in data:
        if letter == current_run_char:
            current_run_length += 1
        else:
            if current_run_length >= k:
                count += 1
            current_run_char = letter
            current_run_length = 1

    # Check the last run
    if current_run_length >= k:
        count += 1

    return count


"""
TESTING
"""
def main() -> None:
    original_file: str = "Testing/huffman_test.txt"
    decoded_file: str = "Testing/decoded.txt"

    with open(original_file, 'r') as f:
        data = f.read()

    """
    STANDARD HUFFMAN ENCODING AND DECODING
    """
    encoded_huffman_file: str = "Testing/huffman_encoded.txt"

    test_huffman = Huffman()

    test_huffman.encode_from_file(original_file, encoded_huffman_file)
    test_huffman.decode_to_file(encoded_huffman_file, decoded_file)

    # Check if both match
    with open(original_file, 'r') as f:
        original_data: str = f.read()
    with open(decoded_file, 'r') as f:
        decoded_data: str = f.read()
    print("Huffman encoding and decoding successful:", original_data == decoded_data)


    """
    DNA ENCODING AND DECODING
    """
    encoded_dnamap_file: str = "Testing/dnamap_encoded.txt"

    test_DNAEncodedMap = DNAEncodedMap()
    # Use what we got from the huffman encoder
    test_DNAEncodedMap.encode_to_file(encoded_huffman_file, encoded_dnamap_file, is_huffman=True)
    test_DNAEncodedMap.decode_to_file(encoded_dnamap_file, decoded_file)

    # Check if both match
    with open(original_file, 'r') as f:
        original_data: str = f.read()
    with open(decoded_file, 'r') as f:
        decoded_data: str = f.read()

    print("DNA encoding and decoding successful:", original_data == decoded_data)

    encoded = open(encoded_dnamap_file).read().split(Huffman().SEPARATOR)[0]
    runs = count_runs(encoded)
    dnamap_length = len(encoded)
    print(f"DNA encoded length: {dnamap_length} bases")
    print(f"Runs: {runs}")

    """
    HUFFMAN QUARTNARY DNA ENCODING AND DECODING
    """
    encoded_huffman_quaternary_file: str = "Testing/huffman_quaternary_encoded.txt"
    test_HuffmanQuaternary = HuffmanQuaternary()

    test_HuffmanQuaternary.encode_from_file(original_file, encoded_huffman_quaternary_file)
    test_HuffmanQuaternary.decode_to_file(encoded_huffman_quaternary_file, decoded_file)

    # Check if both match
    with open(original_file, 'r') as f:
        original_data: str = f.read()
    with open(decoded_file, 'r') as f:
        decoded_data: str = f.read()

    print("Huffman quaternary encoding and decoding successful:", original_data == decoded_data)

    encoded = open(encoded_huffman_quaternary_file).read().split(Huffman().SEPARATOR)[0]
    runs = count_runs(encoded)
    dnaquaternary_length = len(encoded)
    print(f"Huffman quaternary encoded length: {dnaquaternary_length} bases")
    print(f"Runs: {runs}")
    

if __name__ == "__main__":
    main()