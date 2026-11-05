"""
Module handles saving and loading framed files. Saves everything the OligoDeframer needs to decode a file to JSON (the
framer config and its layout, the Huffman codes, and the framed oligos), and rebuilds a config or deframer from it. The
framer and deframer know nothing about JSON, this module is the only place the file format lives. Also saves and loads
whole file encodings (one encoded sequence, no framing), so the encoders know nothing about JSON either.

The Registry maps the names saved in JSON files to the component classes (encoders, indexers, inner codes, and outer
codes), so a saved framer can be rebuilt when decoding. Each component is saved as a spec, its registered name plus the
constructor arguments from its get_spec_args method. Components are saved under their class name.
"""


import json
from pathlib import Path
from typing import Any, Final
from Helpers.Errors import JSONError, NotInRegistryError
from Encode.OligoFramer import FramerConfig, OligoDeframer, OligoFramer
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import BitScanQuaternary, ComplementaryTree, Goldman, Huffman, Pgram, Quaternary, Ternary
from Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Encode.Indexer import DNAMapIndexer, EnumerativeIndexer, GoldmanIndexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, RSInner
from Encode.ErrorCorrection.OuterCodes import DirectDNAGoldmanRedundancy, DirectDNARSOuter, DirectDNAXOR, \
    GoldmanRedundancy, RSOuter, XOR


class Registry:
    """
    Converts components to and from specs that can be written to a JSON file. Basically a class registry to save and 
    load all the components of each encoder, indexer, inner code, and outer code. Each component is saved as a spec, 
    its registered name plus the constructor arguments from its get_spec_args method. Components are saved under their 
    class name if it's a type (e.g. indexer type of an inner code).
    """


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Every component that can be saved, add new classes here so they can be saved and loaded. Saved under their class
    #   name, so if a class is renamed, map its old name to it here too so older files still load
    COMPONENTS: Final[dict[str, type]] = {component.__name__: component for component in (
        # Encoders
        Huffman, Ternary, Quaternary, BitScanQuaternary, Pgram, ComplementaryTree, Goldman, DNAMap, DNAMapHuffman,

        # Indexers
        DNAMapIndexer, GoldmanIndexer, EnumerativeIndexer,

        # Inner codes
        CheckSum, RSInner,

        # Outer codes
        GoldmanRedundancy, DirectDNAGoldmanRedundancy, XOR, DirectDNAXOR, RSOuter, DirectDNARSOuter,
    )}
    NAMES: Final[dict[type, str]] = {component: name for name, component in COMPONENTS.items()}

    # JSON keys for a spec
    TYPE: Final[str] = "type"
    ARGS: Final[str] = "args"
    # Some arguments are classes rather than values (e.g. the indexer type of an inner code), these are saved as
    #   {"class": name} so they can be told apart from plain strings
    CLASS: Final[str] = "class"


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def get_name(component_type: type) -> str:
        """
        Gets the registered name of a component class.

        Args:
            component_type: The component class.

        Returns:
            The name the class is saved under.

        Raises:
            ValueError: If the class is not registered.
        """
        # Check if the component class is registered
        if component_type not in Registry.NAMES:
            raise NotInRegistryError(
                message=f"{component_type.__name__} is not registered, add it to Registry.COMPONENTS to save it",
                component_name=component_type.__name__
            )

        return Registry.NAMES[component_type]


    @staticmethod
    def get_type(name: str) -> type:
        """
        Gets the component class registered under a name.

        Args:
            name: The name the class is saved under.

        Returns:
            The component class.

        Raises:
            ValueError: If no class is registered under the name.
        """
        # Check if the name is registered
        if name not in Registry.COMPONENTS:
            raise NotInRegistryError(
                message=f"{name} is not registered, add it to Registry.COMPONENTS to load it",
                component_name=name
            )

        return Registry.COMPONENTS[name]


    @staticmethod
    def to_spec(component: Any) -> dict[str, Any]:
        """
        Converts a component to a spec that can be written to a JSON file.

        Args:
            component: An encoder, indexer, inner code, or outer code with a get_spec_args method.

        Returns:
            A dictionary with the component's registered name and constructor arguments.

        Raises:
            ValueError: If the component, or a class it takes as an argument, is not registered.
        """
        args: dict[str, Any] = {}
        # For each argument, if it is a class, save its registered name instead of the class itself
        for arg_name, value in component.get_spec_args().items():
            # If we have a type (i.e. indexer type), save it as {"class": name} so we can tell it apart from a string
            if isinstance(value, type):
                value = {Registry.CLASS: Registry.get_name(value)}

            args[arg_name] = value

        # Give back the spec with the component's registered name and its constructor arguments
        return {
            Registry.TYPE: Registry.get_name(type(component)),
            Registry.ARGS: args
        }


    @staticmethod
    def from_spec(spec: dict[str, Any], **extra_args: Any) -> Any:
        """
        Rebuilds a component from its spec.

        Args:
            spec: A dictionary made by to_spec.
            extra_args: Constructor arguments that are not part of the spec (e.g. the width of an indexer).

        Returns:
            A new instance of the component.

        Raises:
            ValueError: If the spec's type, or a class it takes as an argument, is not registered.
            KeyError: If the spec is missing its type or arguments.
        """
        args: dict[str, Any] = {}
        # For each argument, if it is a class, get the class from its registered name instead of the name itself
        for arg_name, value in spec[Registry.ARGS].items():
            if isinstance(value, dict) and Registry.CLASS in value:
                value = Registry.get_type(value[Registry.CLASS])

            args[arg_name] = value

        # Make a new instance of the component with its constructor arguments and any extra arguments
        return Registry.get_type(spec[Registry.TYPE])(**args, **extra_args)



class Serializer:
    """
    Converts framer configs, framed files, and whole file encodings to and from JSON.
    """


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS CONSTANTS                                                        #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Everything relating to the file itself
    FORMAT_VERSION: Final[int] = 1  # Ensure we are reading a file saved with the same version - also helps confirm file
    FORMAT_VERSION_KEY: Final[str] = "format_version"

    FILENAME: Final[str] = "filename"
    CONTENT_LENGTH: Final[str] = "content_length"
    CONTENT_HASH: Final[str] = "content_hash"

    HUFFMAN_CODES: Final[str] = "huffman_codes"
    FRAMED_OLIGOS: Final[str] = "framed_oligos"

    #--#
    # Everything relating to the config and layout of the framer
    CONFIG: Final[str] = "config"

    NUM_OLIGOS: Final[str] = "num_oligos"

    START_PRIMER: Final[str] = "start_primer"
    END_PRIMER: Final[str] = "end_primer"

    ENCODER: Final[str] = "encoder"
    INDEXER: Final[str] = "indexer"

    CHECKSUM: Final[str] = "checksum"
    INNER_CORRECTION: Final[str] = "inner_correction"
    OUTER_CORRECTION: Final[str] = "outer_correction"

    OLIGO_LENGTH: Final[str] = "oligo_length"
    INDEX_WIDTH: Final[str] = "index_width"
    PAYLOAD_LENGTH: Final[str] = "payload_length"
    FILLER_LENGTH: Final[str] = "filler_length"
    #--#


    # Everything relating to whole file encodings (encoded as one sequence, without framing). The encoder and Huffman
    #   codes use the same keys as above
    ENCODED_DATA: Final[str] = "encoded_data"
    HAS_PADDING: Final[str] = "has_padding"
    AS_BINARY: Final[str] = "as_binary"


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def read_json(json_path: str) -> dict[str, Any]:
        """
        Reads a JSON file saved by the Serializer and checks its format version.

        Args:
            json_path: The path to the JSON file.

        Returns:
            The contents of the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            JSONError: If the file is not valid JSON, or was saved with a different format version.
        """
        # Try to open it up and see whats in
        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data: Any = json.load(json_file)
        except json.JSONDecodeError:
            # Raise our own error - eaiser for us to handle
            raise JSONError(
                message="File is not valid JSON",
                json_path=json_path
            )

        # Ensure we are dealing with the actual framer JSON file, and not some random other one
        if not isinstance(data, dict) or data.get(Serializer.FORMAT_VERSION_KEY) != Serializer.FORMAT_VERSION:
            raise JSONError(
                message=f"File is not a version {Serializer.FORMAT_VERSION} Serializer JSON file",
                json_path=json_path
            )

        return data


    @staticmethod
    def write_json(save: dict[str, Any], json_path: str) -> None:
        """
        Writes a dictionary to a JSON file, creating the parent directories if needed.

        Args:
            save: The dictionary to write.
            json_path: The path to the output JSON file.

        Raises:
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        # Ensure the parent directories exist before writing the JSON file
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        # Save nicely with indentation
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(save, json_file, indent=4)


    @staticmethod
    def config_to_dict(config: FramerConfig) -> dict[str, Any]:
        """
        Converts a config to a dictionary that can be written to a JSON file. The in path is left out, it is the file
        being read, not part of how the oligos are framed. The overhead margin is left out too, it is only used to find
        the index width when framing, and the width is saved with the layout.

        Args:
            config: The config to convert.

        Returns:
            A dictionary of the config's settings and layout, with each component saved as a spec.

        Raises:
            ValueError: If a component is not registered.
        """
        # In path and the overhead margin are spefically for encoding, not needed for decoding
        return {
            Serializer.NUM_OLIGOS: config.num_oligos,

            Serializer.START_PRIMER: config.start_primer,
            Serializer.END_PRIMER: config.end_primer,

            Serializer.ENCODER: Registry.to_spec(config.encoder),
            Serializer.INDEXER: Registry.to_spec(config.indexer),

            Serializer.CHECKSUM: Registry.to_spec(config.checksum),
            Serializer.INNER_CORRECTION: Registry.to_spec(config.inner_correction) if config.inner_correction else None,
            Serializer.OUTER_CORRECTION: Registry.to_spec(config.outer_correction) if config.outer_correction else None,

            Serializer.OLIGO_LENGTH: config.oligo_length,
            Serializer.INDEX_WIDTH: config.indexer.width,
            Serializer.PAYLOAD_LENGTH: config.payload_length,
            Serializer.FILLER_LENGTH: config.filler_length
        }


    @staticmethod
    def config_from_dict(data: dict[str, Any], in_path: str) -> FramerConfig:
        """
        Rebuilds a config from a dictionary made by config_to_dict, including its layout. Every component is a new
        instance.

        The overhead margin is not saved, so it is left at its default, it is only used when framing.

        Args:
            data: A dictionary made by config_to_dict.
            in_path: The path to set as the config's in path.

        Returns:
            The rebuilt config.

        Raises:
            KeyError: If the dictionary is missing a setting.
            ValueError: If a component is not registered.
            FramerConfigError: If any of the settings are invalid.
        """
        # Get inner and outer codes, None if no inner or outer code were used
        inner_spec: dict[str, Any] | None = data[Serializer.INNER_CORRECTION]
        outer_spec: dict[str, Any] | None = data[Serializer.OUTER_CORRECTION]

        return FramerConfig(
            in_path=in_path,
            oligo_length=data[Serializer.OLIGO_LENGTH],
            start_primer=data[Serializer.START_PRIMER],
            end_primer=data[Serializer.END_PRIMER],
            encoder=Registry.from_spec(data[Serializer.ENCODER]),
            indexer=Registry.from_spec(data[Serializer.INDEXER], width=data[Serializer.INDEX_WIDTH]),
            checksum=Registry.from_spec(data[Serializer.CHECKSUM]),
            inner_correction=Registry.from_spec(inner_spec) if inner_spec else None,
            outer_correction=Registry.from_spec(outer_spec) if outer_spec else None,
            num_oligos=data[Serializer.NUM_OLIGOS],
            payload_length=data[Serializer.PAYLOAD_LENGTH],
            filler_length=data[Serializer.FILLER_LENGTH]
        )


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def save_framer(framer: OligoFramer, json_path: str) -> None:
        """
        Saves everything needed to decode a framer's oligos to a JSON file: the config with its layout (without the in
        path), the Huffman codes, and the framed oligos. Load it back with load_deframer.

        Args:
            framer: The framer, after frame_all_oligos has been called.
            json_path: The path to the output JSON file. Parent directories are created if needed.

        Raises:
            RuntimeError: If frame_all_oligos has not been called yet, so the layout is not final.
            ValueError: If a component is not registered.
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        if not framer.framed_oligos:
            raise RuntimeError("Call frame_all_oligos before saving, the layout is not final until then.")

        save: dict[str, Any] = {
            Serializer.FORMAT_VERSION_KEY: Serializer.FORMAT_VERSION,
            Serializer.FILENAME: Path(framer.in_path).name,
            Serializer.CONTENT_LENGTH: framer.content_length,
            Serializer.CONTENT_HASH: framer.content_hash,
            Serializer.CONFIG: Serializer.config_to_dict(framer.config),
            Serializer.HUFFMAN_CODES: framer.huffman_codes,
            Serializer.FRAMED_OLIGOS: framer.framed_oligos
        }

        Serializer.write_json(save, json_path)


    @staticmethod
    def load_deframer(json_path: str) -> OligoDeframer:
        """
        Creates a deframer from a JSON file saved by save_framer. Reads the file once, the original file is never needed.

        Args:
            json_path: The path to the JSON file.

        Returns:
            A deframer with the saved config, layout, Huffman codes, and framed oligos.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            JSONError: If the JSON file is not a framer JSON file, was saved with a different format version, or is
                missing part of the config or layout.
            ValueError: If a component is not registered.
        """
        data: dict[str, Any] = Serializer.read_json(json_path)

        # Give back everything the deframer needs to decode and confirm the content is correct
        try:
            return OligoDeframer(
                Serializer.config_from_dict(data[Serializer.CONFIG], json_path),
                huffman_codes=data[Serializer.HUFFMAN_CODES],
                oligos=data[Serializer.FRAMED_OLIGOS],
                filename=data[Serializer.FILENAME],
                content_length=data[Serializer.CONTENT_LENGTH],
                content_hash=data[Serializer.CONTENT_HASH]
            )
        # If we get a key error, really means json error, record our own stuff
        except KeyError as error:
            raise JSONError(
                message=f"JSON file is missing {error}",
                json_path=json_path
            )


    @staticmethod
    def encode_to_file(encoder: Encoding,
                       in_path: str,
                       json_path: str,
                       as_binary: bool = False) -> tuple[dict[str, str], str]:
        """
        Encodes a whole file as one sequence (no segments or framing) and saves the encoded data, Huffman codes, and the
        encoder to a JSON file. Load it back with decode_from_file.

        Args:
            encoder: The encoder to use.
            in_path: The path to the file to encode.
            json_path: The path to the output JSON file. Parent directories are created if needed.
            as_binary: Whether to convert the file to binary before encoding (e.g. BitScanQuaternary scans bits).
                DNAMap always maps bits, so its input is always converted.

        Returns:
            A tuple of the Huffman codes (the DNA map for DNAMap) and the encoded data, same as the encoder's encode.

        Raises:
            FileNotFoundError: If the input file does not exist, or the output path is invalid.
            ValueError: If the encoder (or a class it takes) is not registered, or the encoder cannot encode the data.
        """
        with open(in_path, 'r', encoding='utf-8') as in_file:
            data: str = in_file.read()

        # DNAMap maps pairs of bits to bases, so it always needs binary. DNAMapHuffman takes text, it Huffman codes it
        #   to bits itself
        if as_binary or type(encoder) is DNAMap:
            data = Encoding.ascii_to_binary(data)

        # DNAMap encoders also say if a padding bit was added
        result: tuple = encoder.encode(data)
        huffman_codes: dict[str, str] = result[0]
        encoded_data: str = result[1]
        has_padding: bool = result[2] if len(result) > 2 else False

        Serializer.write_json({
            Serializer.FORMAT_VERSION_KEY: Serializer.FORMAT_VERSION,
            Serializer.FILENAME: Path(in_path).name,
            Serializer.ENCODER: Registry.to_spec(encoder),
            Serializer.AS_BINARY: as_binary,
            Serializer.HAS_PADDING: has_padding,
            Serializer.HUFFMAN_CODES: huffman_codes,
            Serializer.ENCODED_DATA: encoded_data
        }, json_path)

        return huffman_codes, encoded_data


    @staticmethod
    def decode_from_file(json_path: str, out_path: str) -> str:
        """
        Decodes a JSON file saved by encode_to_file and writes the decoded data to a file. The encoder is rebuilt from
        the saved file, so it doesn't need to be given.

        Args:
            json_path: The path to the JSON file.
            out_path: The path to write the decoded data to. Parent directories are created if needed.

        Returns:
            The decoded data.

        Raises:
            FileNotFoundError: If the JSON file does not exist, or the output path is invalid.
            JSONError: If the JSON file is not a Serializer JSON file, or is missing part of the encoding.
            ValueError: If the saved encoder is not registered.
        """
        data: dict[str, Any] = Serializer.read_json(json_path)

        try:
            encoder: Encoding = Registry.from_spec(data[Serializer.ENCODER])
            encoded_data: str = data[Serializer.ENCODED_DATA]
            huffman_codes: dict[str, str] = data[Serializer.HUFFMAN_CODES]
            has_padding: bool = data[Serializer.HAS_PADDING]
            as_binary: bool = data[Serializer.AS_BINARY]
        # If we get a key error, really means json error, record our own stuff
        except KeyError as error:
            raise JSONError(
                message=f"JSON file is missing {error}",
                json_path=json_path
            )

        # DNAMap encoders need to know if a padding bit was added, the others only need the codes
        decoded_data: str
        if isinstance(encoder, DNAMap):
            decoded_data = encoder.decode(encoded_data, huffman_codes=huffman_codes, has_padding=has_padding)
        else:
            decoded_data = encoder.decode(encoded_data, huffman_codes)

        # Undo the conversion to binary from encoding
        if as_binary or type(encoder) is DNAMap:
            decoded_data = Encoding.binary_to_ascii(decoded_data)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as out_file:
            out_file.write(decoded_data)

        return decoded_data