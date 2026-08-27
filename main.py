# TODO: make a base class for the priority queue, then extend for binary and quartnary
class PriorityQueueQuaternary:
    def __init__(self) -> None:
        # Since we need the character and its frequency, easier to store them as a tuple (frequency, character) now
        self.minheap: list[tuple[int, str]] = []
        self.size: int = 0


    def parent(self, index: int) -> int:
        return (index - 1) // 4


    # Has max four children, easier to retrieve all at once
    def children(self, index: int) -> list[int]:
        children_indices: list[int] = []
        for i in range(1, 5):
            child_index: int = 4 * index + i
            children_indices.append(child_index)
        return children_indices


    def heapify(self, index: int) -> None:
        # Get our children
        children: list[int] = self.children(index)

        # Get the smallest among the current index and its children
        smallest: int = index
        for child in children:
            if child < self.size and self.minheap[child][0] < self.minheap[smallest][0]:
                smallest = child

        # If smallest is not the current index, swap and heapify
        if smallest != index:
            self.minheap[index], self.minheap[smallest] = self.minheap[smallest], self.minheap[index]
            self.heapify(smallest)


    def build_queue(self, frequencies: dict[str, int]) -> None:
        # Add all the frequency-character pairs to the heap
        for char, freq in frequencies.items():
            self.minheap.append((freq, char))
            self.size += 1

        # Start from the last non-leaf node and heapify each node
        for i in range(self.size // 4 - 1, -1, -1):
            self.heapify(i)


    def is_empty(self) -> bool:
        return self.size == 0


    def top(self) -> tuple:
        # Can't return the top element if the heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        return self.minheap[0]


    def dequeue(self) -> tuple:
        # Can't dequeue if heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")

        # Our top
        top_element: tuple = self.top()

        # Move the last element to our root, removing it from the end
        self.minheap[0] = self.minheap[self.size - 1]
        self.size -= 1
        self.minheap.pop()

        # Heapify the root to maintain the min-heap property
        self.heapify(0)
        return top_element


class PriorityQueue:
    def __init__(self) -> None:
        # Since we need the character and its frequency, easier to store them as a tuple (frequency, character) now
        self.minheap: list[tuple[int, str]] = []
        self.size: int = 0


    def parent(self, index: int) -> int:
        return (index - 1) // 2


    def left_child(self, index: int) -> int:
        return 2 * index + 1


    def right_child(self, index: int) -> int:
        return 2 * index + 2


    def heapify(self, index: int) -> None:
        # Get our left and right child indices
        left: int = self.left_child(index)
        right: int = self.right_child(index)

        smallest: int = index

        # If left is smaller, update
        if left < self.size and self.minheap[left][0] < self.minheap[smallest][0]:
            smallest = left
        # If right is smaller, update
        if right < self.size and self.minheap[right][0] < self.minheap[smallest][0]:
            smallest = right

        # If smallest is not the current index, swap and heapify
        if smallest != index:
            self.minheap[index], self.minheap[smallest] = self.minheap[smallest], self.minheap[index]
            self.heapify(smallest)


    def build_queue(self, frequencies: dict[str, int]) -> None:
        # Add all the frequency-character pairs to the heap
        for char, freq in frequencies.items():
            self.minheap.append((freq, char))
            self.size += 1

        # Start from the last non-leaf node and heapify each node
        for i in range(self.size // 2 - 1, -1, -1):
            self.heapify(i)


    def is_empty(self) -> bool:
        return self.size == 0


    def top(self) -> tuple:
        # Can't return the top element if the heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        return self.minheap[0]


    def dequeue(self) -> tuple:
        # Can't dequeue if heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")

        # Our top
        top_element: tuple = self.top()

        # Move the last element to our root, removing it from the end
        self.minheap[0] = self.minheap[self.size - 1]
        self.size -= 1
        self.minheap.pop()

        # Heapify the root to maintain the min-heap property
        self.heapify(0)
        return top_element


    def enqueue(self, element: tuple) -> None:
        # Add the new element to the end of the heap
        self.minheap.append(element)
        self.size += 1

        # Bubble up the new element to maintain the min-heap property
        index: int = self.size - 1
        parent: int = self.parent(index)

        # While not at root and parent is greater, swap
        while index != 0 and self.minheap[parent][0] > self.minheap[index][0]:
            # Swap with parent
            self.minheap[index], self.minheap[parent] = self.minheap[parent], self.minheap[index]

            # Update index and parent for the next iteration
            index = self.parent(index)
            parent = self.parent(index)
    


class Huffman:
    def __init__(self,) -> None:
        self.priority_queue: PriorityQueue = PriorityQueue()

        # Separator to separate the table and encoded data in the output file
        # TODO: This is theoretical, application would store table completely differently
        self.SEPARATOR: str = "====="


    def count_frequencies(self, data: str) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for char in data:
            if char in frequencies:
                frequencies[char] += 1
            else:
                frequencies[char] = 1
        
        return frequencies


    def build_tree(self, data: str) -> None:
        # Set up our priority queue with the frequencies
        self.priority_queue.build_queue(self.count_frequencies(data))

        # Our children
        left: tuple
        right: tuple

        # Merged values
        merged_frequency: int
        merged_node: tuple

        # While there is more than one node in the priority queue, we need to merge nodes
        while self.priority_queue.size > 1:
            # Dequeue the two nodes with the smallest frequencies
            left = self.priority_queue.dequeue()
            right = self.priority_queue.dequeue()

            # Create a new internal node with these two nodes as children
            merged_frequency = left[0] + right[0]
            merged_node = (merged_frequency, (left, right))

            # Enqueue the new node back into the priority queue
            self.priority_queue.enqueue(merged_node)


    def generate_codes(self) -> dict[str, str]:
        # Ensure we only have one node left
        if self.priority_queue.size != 1:
            raise ValueError("Priority queue should have exactly one node left to generate codes.")

        # Initialize an empty dictionary to hold the codes
        codes: dict[str, str] = {}

        # Helper function to recursively generate codes
        def _create_code(node: tuple, current_code: str="") -> None:
            # If we have a string it is a leaf, otherwise it is an internal node
            if isinstance(node[1], str):  # Leaf node
                codes[node[1]] = current_code
                return

            # Internal node, traverse left and right
            _create_code(node[1][0], current_code + '0')  # Left child
            _create_code(node[1][1], current_code + '1')  # Right child

        # Start the recursive code generation from the root node
        root_node = self.priority_queue.top()
        _create_code(root_node)

        return codes
    

    def encode(self, data: str) -> tuple[dict[str, str], str]:

        # Build tree and generate Huffman codes
        self.build_tree(data)
        huffman_codes: dict[str, str] = self.generate_codes()
        encoded_data: str = "".join(huffman_codes[char] for char in data)

        # Return the table and encoded data as or string of encoded data
        return huffman_codes, encoded_data


    def encode_from_file(self, in_path: str, out_path: str) -> None:
        with open(in_path, 'r') as f:
            data: str = f.read()
        
        # Get the Huffman codes and encoded data
        huffman_codes, encoded_data = self.encode(data)

        # Write the table and encoded data to the output file
        with open(out_path, 'w') as f:
            # Encoded data first for easier parsing
            f.write(encoded_data)

            # Record the separator to separate the table and encoded data
            # TODO: This again is theoretical, will need to be stored differently in a real application
            f.write("\n")
            f.write(self.SEPARATOR)
            f.write("\n")

            # Then the table
            f.write("\n".join(f"{char}:{code}" for char, code in huffman_codes.items()))


    def format_table(self, raw_table: str) -> dict[str, str]:
        # Split the raw table into lines
        lines: list[str] = raw_table.strip().split('\n')

        # Initialize an empty dictionary to hold the formatted table
        huffman_codes: dict[str, str] = {}

        # Table is second
        lines = raw_table.strip().split('\n')
        # Read  until EOF
        huffman_codes: dict[str, str] = {}
        for index in range(len(lines)):
            line: str = lines[index]

            # Check if newline character (empty string after splitting) as we separate the table with newlines but still
            #   store it as a single string
            # TODO: Make this less hacky
            if line == "":
                char, code = "\n", lines[index + 1][1:]
                huffman_codes[code] = char
            # Happens after newline character, so skip if we encounter as already recorded
            elif len(line) > 1 and line[0] == ':' and line[1] != ':':
                continue

            # Normal line
            elif len(line) > 1 and line[1] == ':':
                char, code = line[0], line[2:]
                huffman_codes[code] = char
            else:
                raise ValueError(f"Invalid line in the encoded file: {index} {line}")

        return huffman_codes


    def decode(self,  encoded_data: str, huffman_codes: dict[str, str]) -> str:
        # Decode the encoded data
        decoded_data: str = ""
        current_code: str = ""

        # For all our bits, try to match them to a code, match = append and reset
        for bit in encoded_data:
            current_code += bit

            # If match found, append and reset current_code
            if current_code in huffman_codes:
                decoded_data += huffman_codes[current_code]
                current_code = ""

        return decoded_data


    def decode_to_file(self, in_path: str, out_path: str) -> None:
        # Parse encoded data and table from the input file
        with open(in_path, 'r') as f:
            data: str = f.read().strip()

        # Split the data into the table and encoded data
        values: list[str] = data.split(self.SEPARATOR)
        if len(values) < 2:
            raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

        # Enocoded data is first
        encoded_data: str = values[0]
        huffman_codes: dict[str, str] = self.format_table(values[1].strip())

        # Now decode the data using the table
        decoded_data: str = self.decode(encoded_data, huffman_codes)
        with open(out_path, 'w') as f:
            f.write(decoded_data)


class DNAEncodedMap:
    def __init__(self) -> None:
        # After encoding, use the binary string to map to DNA bases
        self.DNA_MAP = {
            '00': 'A',
            '01': 'C',
            '10': 'G',
            '11': 'T'
        }
        # Reverse now
        self.REVERSE_DNA_MAP = {v: k for k, v in self.DNA_MAP.items()}


    def encode(self, binary_string: str) -> tuple[bool, str]:
        # Remove any leading/trailing whitespace, causes problems
        binary_string = binary_string.strip()

        # Pad the binary string to make its length a multiple of 2
        dna_string: str = ""
        has_padding: bool = False

        # When decoding if the padding was added, we will chop it off
        if len(binary_string) % 2 != 0:
            binary_string += '0'
            has_padding = True

        # Now encode the binary string to DNA
        for index in range(0, len(binary_string), 2):
            pair: str = binary_string[index:index+2]
            dna_string += self.DNA_MAP[pair]

        return has_padding, dna_string


    def encode_to_file(self, in_path: str, out_path: str, is_huffman=True) -> None:
        # Read the binary string from the input file
        with open(in_path, 'r') as f:
            binary_string: str = f.read()

            # Incase we have the table attached
            if is_huffman:
                # Split the table and encoded data
                parts: list[str] = binary_string.split(Huffman().SEPARATOR)
                if len(parts) != 2:
                    raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

                # Get the data itself
                binary_string = parts[0].strip()
                # Get the table - will remain untouched, think of it as metadata
                raw_table: str = parts[1].strip()

        # Encode the binary string to DNA
        has_padding:bool 
        dna_string: str 
        has_padding, dna_string = self.encode(binary_string)

        # Write the DNA string to the output file
        with open(out_path, 'w') as f:
            f.write(dna_string)
            f.write("\n")

            if is_huffman:
                f.write(Huffman().SEPARATOR)
                f.write("\n")
                f.write(raw_table)
                f.write("\n")

            # TODO: should really be done as a metafile
            # Add indicator for if padding was added or not
            f.write(Huffman().SEPARATOR)
            f.write("\n")
            f.write("1") if has_padding else f.write("0")


    def decode(self, dna_string: str, has_padding: bool) -> str:
        # Decode the DNA string back to binary
        binary_string: str = ""
        for base in dna_string:
            binary_string += self.REVERSE_DNA_MAP[base]

        # If padding was added, we need to remove the last bit after decoding
        if has_padding:
            binary_string = binary_string[:-1]

        return binary_string


    def decode_to_file(self, in_path: str, out_path: str) -> None: 
        # Read the DNA string from the input file
        with open(in_path, 'r') as f:
            data: str = f.read()
            is_huffman: bool = False
            parts: list[str] = data.split(Huffman().SEPARATOR)

            # Had table, means is in huffman format
            if len(parts) == 3:
                dna_string: str = parts[0].strip()
                raw_table: str = parts[1].strip()
                is_huffman = True
            # No table, means is in standard format
            elif len(parts) == 2:
                dna_string: str = parts[0].strip()
                raw_table: str = ""
            # Something went wrong
            else:
                raise ValueError("Invalid input format. Expected a separator between table and encoded data.")

            # Padding is always last
            raw_padding: str = parts[-1].strip()

        # Format our table
        huffman_codes: dict[str, str] = Huffman().format_table(raw_table)

        # Ensure proper table
        if raw_table and not huffman_codes:
            raise ValueError("Invalid table format. Could not parse the Huffman table.")

        # Make sure the padding indicator is valid
        if raw_padding not in {"0", "1"}:
            raise ValueError("Invalid padding indicator. Expected '0' or '1'.")

        # Convert the padding indicator to a boolean
        has_padding: bool = raw_padding == "1"

        # Decode the DNA string back to binary
        binary_string: str = self.decode(dna_string, has_padding)

        if is_huffman:
            # Decode the binary string back to original data using the Huffman table
            decoded_data: str = Huffman().decode(binary_string, huffman_codes)
        else:
            decoded_data: str = binary_string

        # Write the decoded data to the output file
        with open(out_path, 'w') as f:
            f.write(decoded_data)


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


    """
    PRIORITY QUEUE QUARTNARY TESTING
    """
    PQQ = PriorityQueueQuaternary()
    frequencies: dict[str, int] = test_huffman.count_frequencies(data)
    PQQ.build_queue(frequencies)
    for _ in range(len(frequencies)):
        print(PQQ.dequeue())
    

if __name__ == "__main__":
    main()