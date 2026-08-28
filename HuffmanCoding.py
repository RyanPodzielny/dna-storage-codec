from PriorityQueue import PriorityQueue

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


class HuffmanQuaternary(Huffman):
    def __init__(self) -> None:
        super().__init__()

        self.TREE_ORDER: tuple[str, ...] = ('A', 'C', 'G', 'T')


    def build_tree(self, data: str) -> None:
        self.priority_queue.build_queue(self.count_frequencies(data))

        # Merged values
        merged_frequency: int = 0
        merged_node: tuple

        # We will have four children, so we need to merge four nodes at a time
        while self.priority_queue.size > 1:
            # Should have 4 children per
            children: list[tuple[int, str]] = []

            # Dequeue four nodes with smallest frequencies. If we don't have four, just take what we can
            for _ in range(4):
                if not self.priority_queue.is_empty():
                    children.append(self.priority_queue.dequeue())
                else:
                    break

            # Sum up the frequencies
            for child in children:
                merged_frequency += child[0]

            # Create a new internal node and put pack onto queue
            merged_node = (merged_frequency, tuple(children))

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

            # Traverse the children, and append the proper DNA base based on the current child number
            for index, child in enumerate(node[1]):
                _create_code(child, current_code + self.TREE_ORDER[index]) 

            
        # Start the recursive code generation from the root node
        root_node = self.priority_queue.top()
        _create_code(root_node)

        return codes