from HuffmanCoding import Huffman

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