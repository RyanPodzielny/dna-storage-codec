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
    def __init__(self, data: str) -> None:
        self.data: str = data
        self.frequencies: dict[str, int] = self.count_frequencies()

        self.priority_queue: PriorityQueue = PriorityQueue()
        self.priority_queue.build_queue(self.frequencies)

        self.SEPARATOR: str = "!"  # Separator for the table and encoded data


    def count_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for char in self.data:
            if char in frequencies:
                frequencies[char] += 1
            else:
                frequencies[char] = 1
        
        return frequencies


    def build_tree(self) -> None:
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


    def encode(self, in_path: str, out_path: str) -> None:
        # Read the file content
        with open(in_path, 'r') as f:
            data = f.read()

        # Build tree and generate Huffman codes
        self.build_tree()
        huffman_codes: dict[str, str] = self.generate_codes()
        encoded_data: str = "".join(huffman_codes[char] for char in data)

        # Write the encoded data to a new file, storing the table first
        with open(out_path, 'w') as f:
            for char, code in huffman_codes.items():
                f.write(f"{char}:{code}\n")

            # Add a separator to indicate the end of the table
            f.write(self.SEPARATOR + "\n") 
            f.write(encoded_data)


    def decode(self, in_path: str, out_path: str) -> None:
        # Read the file content
        with open(in_path, 'r') as f:
            data: str = f.read()

        # Parse the table and encoded data
        lines: list[str] = data.split('\n')
        huffman_codes: dict[str, str] = {}
        encoded_data: str = ""


        # Read the table until the separator is found
        for index in range(len(lines)):
            line: str = lines[index]

            # Everything after the separator is the encoded data
            if line == self.SEPARATOR:
                encoded_data = "".join(lines[index + 1:])
                break
            # Check if newline character (empty string)as we separate the table and encoded data with a newline, so we need to handle that case
            # TODO: Make this less hacky
            elif line == "":
                char, code = "\n", lines[index + 1][1:]
                huffman_codes[code] = char
            # Happens when we 
            elif len(line) > 1 and line[0] == ':' and line[1] != ':':
                continue

            # Normal line
            elif len(line) > 1 and line[1] == ':':
                char, code = line[0], line[2:]
                huffman_codes[code] = char
            else:
                raise ValueError(f"Invalid line in the encoded file: {index} {line}")


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

        # Write the decoded data to a new file
        with open(out_path, 'w') as f:
            f.write(decoded_data)


"""
TESTING
"""
def main() -> None:
    original_file: str = "Testing/huffman_test.txt"
    encoded_file: str = "Testing/huffman_encoded.txt"
    decoded_file: str = "Testing/huffman_decoded.txt"

    data: str = ""

    with open(original_file, 'r') as f:
        data = f.read()

    test = Huffman(data)
    test.encode(original_file, encoded_file)
    test.decode(encoded_file, decoded_file)

    with open(decoded_file, 'r') as f:
        decoded_data: str = f.read()
    print("Decoded data matches original:", decoded_data == data)


if __name__ == "__main__":
    main()