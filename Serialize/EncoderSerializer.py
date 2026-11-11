"""
Module handles saving and loading framed files. Saves everything the OligoDeframer needs to decode a file to JSON (the
framer config and its layout, the Huffman codes, and the framed oligos), and rebuilds a config or deframer from it. The
framer and deframer know nothing about JSON, this module is the only place the file format lives. Also saves and loads
whole file encodings (one encoded sequence, no framing), so the encoders know nothing about JSON either.

Each file is a pydantic model (SavedFramer and SavedSingleSegment), so the keys are only written in one place. Loading
reads the file with model_validate_json, which checks it is valid JSON, the format version, and every key and value in
one go. A framer's config is kept as a plain dictionary in the file, it is checked by FramerConfig itself when it is
rebuilt, so there is only one place that says what a valid config is.

The Registry maps the names saved in JSON files to the component classes (encoders, indexers, inner codes, and outer
codes), so a saved framer can be rebuilt when decoding. Each component is saved as a spec, its registered name plus the
constructor arguments from its get_spec_args method. Components are saved under their class name.
"""


from pathlib import Path
from typing import Annotated, Any, Final, Literal
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt
from pydantic.dataclasses import dataclass
from Helpers.Errors import NotInRegistryError
from Encode.OligoFramer import FramerConfig, OligoDeframer, OligoFramer, Primer
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import BitScanQuaternary, ComplementaryTree, Goldman, Huffman, Pgram, Quaternary, \
    Ternary
from Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Encode.Indexer import DNAMapIndexer, EnumerativeIndexer, GoldmanIndexer, Indexer
from Encode.ErrorCorrection.InnerCodes import CheckSum, InnerCodes, RSInner
from Encode.ErrorCorrection.OuterCodes import DirectDNAGoldmanRedundancy, DirectDNARSOuter, DirectDNAXOR, \
    GoldmanRedundancy, OuterCodes, RSOuter, XOR


#------------------------------------------------------------------------------#
#                                                                              #
#   TYPES                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#


# Oligos and encoded data are only bases, same rule as a primer (but named for what they are)
DNASequence = Primer

# The hash from OligoFormat.hash_content - SHA-256 is always 64 lowercase characters
ContentHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

# Make life easier by saying component rather than doing Encoding | Indexer ...
Component = Annotated[Encoding, Indexer, InnerCodes, OuterCodes]



class SavedFramer(BaseModel):
    """
    Everything the deframer needs to decode a file, the whole file saved by save_framer. The config is a plain
    dictionary here, FramerConfig checks it when it is rebuilt.
    """

    model_config = ConfigDict(extra="forbid")

    # Ensure we are reading a file saved with the same version
    format_version: Literal[1] = 1

    filename: str
    content_length: NonNegativeInt
    content_hash: ContentHash

    config: dict[str, Any]

    huffman_codes: dict[str, str]
    framed_oligos: list[DNASequence]



class SavedSingleSegment(BaseModel):
    """
    A whole file encoded as one sequence (no framing), the whole file saved by encode_as_single_segment. Mostly used
    for testing, the user can't do this - but a model means the keys are written once here instead of as strings in
    both the save and the load.
    """

    model_config = ConfigDict(extra="forbid")

    # Same as the framer file, bump the number here if the format changes
    format_version: Literal[1] = 1

    filename: str

    # The encoder's spec, see Registry.to_spec
    encoder: dict[str, Any]

    # If the file was converted to binary before encoding, and if DNAMap added a padding bit
    as_binary: bool
    has_padding: bool

    huffman_codes: dict[str, str]
    # Not always bases - the plain Huffman encoder outputs binary (0s and 1s), so any string is allowed here
    encoded_data: str



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
            NotInRegistryError: If the class is not registered.
        """
        # Check if the component class is registered
        if component_type not in Registry.NAMES:
            raise NotInRegistryError(
                message=f"{component_type.__name__} is not registered, add it to Registry.COMPONENTS to save it",
                component_name=component_type.__name__
            )

        # Return the name the class is registered under
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
            NotInRegistryError: If no class is registered under the name.
        """
        # Check if the name is registered
        if name not in Registry.COMPONENTS:
            raise NotInRegistryError(
                message=f"{name} is not registered, add it to Registry.COMPONENTS to load it",
                component_name=name
            )

        # Return the class that is registered under tht name
        return Registry.COMPONENTS[name]


    @staticmethod
    def to_spec(component: Component) -> dict[str, Any]:
        """
        Converts a component to a spec that can be written to a JSON file.

        Args:
            component: An encoder, indexer, inner code, or outer code with a get_spec_args method.

        Returns:
            A dictionary with the component's registered name and constructor arguments.

        Raises:
            NotInRegistryError: If the component, or a class it takes as an argument, is not registered.
        """
        args: dict[str, Any] = {}
        # For each argument, if it is a class, save its registered name instead of the class itself
        for arg_name, value in component.get_spec_args().items():
            # If we have a type (e.g. indexer type), save it as {"class": name} 
            if isinstance(value, type):
                value = {Registry.CLASS: Registry.get_name(value)}

            # Then save it our args for that value as {"args": {arg_name: value, ...}} 
            args[arg_name] = value

        # Listed as {"type": class_name, "args": {arg_name: value, ...}} so it can be rebuilt with from_spec
        spec: dict[str, Any] = {
            Registry.TYPE: Registry.get_name(type(component)),
            Registry.ARGS: args
        }

        return spec


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
            NotInRegistryError: If the spec's type, or a class it takes as an argument, is not registered.
            KeyError: If the spec is missing its type or arguments.
            TypeError: If the arguments don't match the component's constructor.
        """
        args: dict[str, Any] = {}
        # For each argument, if it is a class, get the class from its registered name instead of the name itself
        for arg_name, value in spec[Registry.ARGS].items():
            # If we have a {"class": name}, get the class from the registry instead of the name
            if isinstance(value, dict) and Registry.CLASS in value:
                value = Registry.get_type(value[Registry.CLASS])

            args[arg_name] = value

        # Make a new instance of the component with its constructor arguments and any extra arguments. Looks spooky,
        #   but it's just a way to call consutrctor with dict of arguments, same as component(**args, **extra_args).
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


    # FramerConfig fields that are objects, saved as specs instead of their values
    COMPONENT_FIELDS: Final[tuple[str, ...]] = (
        "encoder", "indexer", "checksum", "inner_correction", "outer_correction"
    )
    # FramerConfig fields only used when encoding, so not saved
    UNSAVED_FIELDS: Final[tuple[str, ...]] = ("in_path", "overhead_margin")

    # The indexer's width isn't a constructor argument in its spec, so it is saved next to the config's settings
    INDEX_LENGTH: Final[str] = "index_length"
    # The name of the width argument in an indexer's spec
    WIDTH: Final[str] = "width"  

    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def _write_json(save: BaseModel, json_path: str) -> None:
        """
        Writes a model to a JSON file, creating the parent directories if needed.

        Args:
            save: The model to write.
            json_path: The path to the output JSON file.

        Raises:
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        # Ensure the parent directories exist before writing the JSON file
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        # Save nicely with indentation
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(save.model_dump_json(indent=4))


    @staticmethod
    def _config_to_dict(config: FramerConfig) -> dict[str, Any]:
        """
        Converts a config to a dictionary that can be written to a JSON file. Plain settings come straight from the
        config's model_dump, components are saved as specs, and the indexer's width is saved with the layout.

        The in path is left out, it is the file being read, not part of how the oligos are framed. The overhead margin
        is left out too, it is only used to find the index width when framing, and the width is saved instead.

        Args:
            config: The config to convert.

        Returns:
            A dictionary of the config's settings and layout, with each component saved as a spec.

        Raises:
            NotInRegistryError: If a component is not registered.
        """
        exclude_fields: set[str] = set()
        # Our components fields are objects, so we have to save them differently as they have arguments
        for name in Serializer.COMPONENT_FIELDS:
            exclude_fields.add(name)  
        # Fields that aren't needed, but needed for Framer creation (like overhead margin and inpath) are left out
        for name in Serializer.UNSAVED_FIELDS:
            exclude_fields.add(name)

        # Everything that isn't a component or only used for encoding is already plain JSON
        saved: dict[str, Any] = config.model_dump(exclude=exclude_fields)

        # Index width is part of the Indexer's class rather than the config - want to save next to payload length
        saved[Serializer.INDEX_LENGTH] = config.indexer.width

        # Components are objects, save them as specs - None if no inner or outer code was used
        for name in Serializer.COMPONENT_FIELDS:
            component: Component = getattr(config, name)
            # Example save: {component_name: {"type": class_name, "args": {arg_name: value, ...}}} or None
            saved[name] = Registry.to_spec(component) if component else None

        return saved


    @staticmethod
    def _config_from_dict(data: dict[str, Any], in_path: str) -> FramerConfig:
        """
        Rebuilds a config from a dictionary made by _config_to_dict, including its layout. Every component is a new
        instance. Goes through FramerConfig's own fields instead of naming each one, so a new field is loaded without
        changing this. FramerConfig checks every value when it is built, and a setting missing from the file is left
        out, so FramerConfig reports it.

        The overhead margin is not saved, so it is left at its default, it is only used when framing.

        Args:
            data: A dictionary made by _config_to_dict.
            in_path: The path to set as the config's in path.

        Returns:
            The rebuilt config.

        Raises:
            ValidationError: If any setting is missing or invalid for a FramerConfig.
            NotInRegistryError: If a component is not registered.
            KeyError: If the index width is missing, or a component spec is missing its type or arguments.
            TypeError: If a component spec's arguments don't match its constructor.
        """
        # Kinda jank but gets the job done
        IN_PATH: Final[str] = "in_path"
        INDEXER: Final[str] = "indexer"

        # The in path isn't saved, it is the file we just loaded
        settings: dict[str, Any] = {IN_PATH: in_path}

        # FramerConfig.model_fields is every field name on the config (oligo_length, encoder, num_oligos, ...)
        for name in FramerConfig.model_fields:
            # Not saved (in path, overhead margin) - or missing from the file, then FramerConfig says it is required
            if name not in data:
                continue

            # Get the value from the file, it is either a plain value or a spec for a component
            value: Any = data[name]

            # Components are saved as specs, so rebuild each into a new object - None if no inner or outer code
            if name in Serializer.COMPONENT_FIELDS and value is not None:
                # The indexer's width isn't part of its spec, it was saved next to the settings, so give it back here
                extra_args: dict[str, Any] = {}
                if name == INDEXER:
                    extra_args[Serializer.WIDTH] = data[Serializer.INDEX_LENGTH]

                # Get the spec from the registry with unpacking any extra args
                value = Registry.from_spec(value, **extra_args)

            # Everything else was saved as is, so it goes straight in
            settings[name] = value

        # Build the config from every setting we found; this also checks them all.
        return FramerConfig.model_validate(settings)


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
            NotInRegistryError: If a component is not registered.
            ValidationError: If the framer has something that can't be saved (e.g. an oligo that isn't only bases).
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        if not framer.framed_oligos:
            raise RuntimeError("Call frame_all_oligos before saving, the layout is not final until then.")

        # Building the model checks it too, so a bad framer is caught before it is written, not when it is loaded
        save: SavedFramer = SavedFramer(
            filename=Path(framer.in_path).name,
            content_length=framer.content_length,
            content_hash=framer.content_hash,
            config=Serializer._config_to_dict(framer.config),
            huffman_codes=framer.huffman_codes,
            framed_oligos=framer.framed_oligos
        )

        Serializer._write_json(save, json_path)


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
            ValidationError: If the file is not valid JSON, not a framer file, was saved with a different format
                version, or has anything missing, extra, or invalid - including in the config, checked by FramerConfig.
            NotInRegistryError: If a saved component is not registered.
            KeyError: If a component spec is missing its type or arguments.
            TypeError: If a component spec's arguments don't match its constructor.
        """
        # Checks the whole file in one go - valid JSON, the format version, every key, and every value
        json_content: str = Path(json_path).read_text(encoding='utf-8')
        saved: SavedFramer = SavedFramer.model_validate_json(json_content)

        # Give back everything the deframer needs to decode and confirm the content is correct
        return OligoDeframer(
            Serializer._config_from_dict(saved.config, json_path),
            huffman_codes=saved.huffman_codes,
            oligos=saved.framed_oligos,
            filename=saved.filename,
            content_length=saved.content_length,
            content_hash=saved.content_hash
        )


    @staticmethod
    def save_segments_to_text(segments: list[str], out_path: str) -> None:
        """
        Saves a list of segments to a text file - one segment per line.
        Args:
            segments: The list of segments to save.
            output_path: The path to the output text file. Parent directories are created if needed.

        Raises:
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        with open(out_path, 'w') as out_file:
            for segment in segments:
                out_file.write(segment + '\n')


    @staticmethod
    def encode_as_single_segment(encoder: Encoding,
                                 in_path: str,
                                 json_path: str,
                                 as_binary: bool = False) -> tuple[dict[str, str], str]:
        """
        Encodes a whole file as one sequence (no segments or framing) and saves the encoded data, Huffman codes, and the
        encoder to a JSON file. Load it back with decode_from_single_segment.

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
            NotInRegistryError: If the encoder (or a class it takes) is not registered.
            ValueError: If the encoder cannot encode the data.
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

        # Building the model checks it too, so a bad encoding is caught before it is written
        save: SavedSingleSegment = SavedSingleSegment(
            filename=Path(in_path).name,
            encoder=Registry.to_spec(encoder),
            as_binary=as_binary,
            has_padding=has_padding,
            huffman_codes=huffman_codes,
            encoded_data=encoded_data
        )

        Serializer._write_json(save, json_path)

        return huffman_codes, encoded_data


    @staticmethod
    def decode_from_single_segment(json_path: str, out_path: str) -> str:
        """
        Decodes a JSON file saved by encode_as_single_segment and writes the decoded data to a file. The encoder is
        rebuilt from the saved file, so it doesn't need to be given.

        Args:
            json_path: The path to the JSON file.
            out_path: The path to write the decoded data to. Parent directories are created if needed.

        Returns:
            The decoded data.

        Raises:
            FileNotFoundError: If the JSON file does not exist, or the output path is invalid.
            ValidationError: If the file is not valid JSON, not a single segment file, or has anything missing, extra,
                or invalid.
            NotInRegistryError: If the saved encoder is not registered.
        """
        # Checks the whole file in one go - valid JSON, the format version, every key, and every value
        json_content: str = Path(json_path).read_text(encoding='utf-8')
        saved: SavedSingleSegment = SavedSingleSegment.model_validate_json(json_content)

        encoder: Encoding = Registry.from_spec(saved.encoder)

        # DNAMap encoders need to know if a padding bit was added, the others only need the codes. The encoded data is
        #   passed by position, the encoders don't agree on its name (encoded_data for Huffman, dna_string for DNAMap)
        decoded_data: str
        if isinstance(encoder, DNAMap):
            decoded_data = encoder.decode(
                dna_string=saved.encoded_data,
                huffman_codes=saved.huffman_codes,
                has_padding=saved.has_padding
            )
        else:
            decoded_data = encoder.decode(
                data=saved.encoded_data,
                huffman_codes=saved.huffman_codes
            )

        # Undo the conversion to binary from encoding
        if saved.as_binary or type(encoder) is DNAMap:
            decoded_data = Encoding.binary_to_ascii(decoded_data)

        # Write the decoded data to the output file
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as out_file:
            out_file.write(decoded_data)

        return decoded_data