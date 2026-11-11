"""
Module handles saving and loading error simulator configs. A single stage (synthesis, aging, PCR, or sequencing) can be
saved and loaded on its own, so a user can reuse e.g. one sequencing setup across different simulations. All four
stages can also be saved together as one simulation file.

The configs are already pydantic models, so saving is just model_dump_json and loading is model_validate_json - every
value is checked by the config itself when it is loaded, the same as when it is built in code. Each stage config has a
stage field fixed to its own value, so a single stage file says which stage it is, and loading builds the right config
from the file alone.
"""


from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict
from Simulate.ErrorConfig import AgingConfig, PCRConfig, SequencingConfig, Stage, SynthesisConfig
from Simulate.ErrorSimulator import ErrorDictKeys, ErrorRecord, ErrorType


#------------------------------------------------------------------------------#
#                                                                              #
#   TYPES                                                                      #
#                                                                              #   
#------------------------------------------------------------------------------#


StageConfig = AgingConfig | PCRConfig | SequencingConfig | SynthesisConfig



class SavedStage(BaseModel):
    """
    A single stage's config, the whole file saved by save_stage.
    """

    # No extra fields allowed, e.g. PCR config saved under synthesis is rejected when loading
    model_config = ConfigDict(extra="forbid")

    # Ensure we are reading a file saved with the same version. Bump the number here if the format changes.
    format_version: Literal[1] = 1

    config: StageConfig



class SavedSimulation(BaseModel):
    """
    Every stage's config for a whole simulation, the whole file saved by save_simulation. Each field only takes its own
    stage, so e.g. a PCR config saved under synthesis is rejected when loading.
    """

    # Don't allow for extra fields
    model_config = ConfigDict(extra="forbid")

    # Same as a single stage file, bump the number here if the format changes
    format_version: Literal[1] = 1

    # In the order the strands go through them
    synthesis: SynthesisConfig
    aging: AgingConfig
    pcr: PCRConfig
    sequencing: SequencingConfig



class SavedErrors(BaseModel):
    """
    Every error from a simulation, the whole file saved by save_errors. Only made for the user to look at, so it is
    never loaded back.
    """

    # Don't allow for extra fields
    model_config = ConfigDict(extra="forbid")

    # How many of each error happened, then every error in the order they happened. Each error is the dict the error
    #   record already makes (index, position, stage, error type, and the strand after the error)
    counts: dict[ErrorType, int]
    records: list[dict[str, (int | None) | Stage | ErrorType | str]]



class ErrorConfigSerializer:
    """
    Saves and loads error simulator configs, one stage at a time or a whole simulation at once.
    """


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


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def save_stage(config: StageConfig, json_path: str) -> None:
        """
        Saves a single stage's config to a JSON file. The config's stage is saved with it. Load it back with
        load_stage.

        Args:
            config: The stage config to save.
            json_path: The path to the output JSON file. Parent directories are created if needed.

        Raises:
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        ErrorConfigSerializer._write_json(SavedStage(config=config), json_path)


    @staticmethod
    def load_stage(json_path: str) -> StageConfig:
        """
        Loads a single stage's config from a JSON file saved by save_stage. The stage saved in the file decides which
        config is built.

        Args:
            json_path: The path to the JSON file.

        Returns:
            The stage config - check its stage (or use isinstance) to see which one it is.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValidationError: If the file is not valid JSON, not a stage file, was saved with a different format version,
                has an unknown stage, or has anything missing, extra, or invalid in the config.
        """
        # Checks the whole file in one go - valid JSON, the format version, the stage, and every config value
        json_content: str = Path(json_path).read_text(encoding='utf-8')
        saved: SavedStage = SavedStage.model_validate_json(json_content)

        return saved.config


    @staticmethod
    def save_simulation(synthesis: SynthesisConfig,
                        aging: AgingConfig,
                        pcr: PCRConfig,
                        sequencing: SequencingConfig,
                        json_path: str) -> None:
        """
        Saves every stage's config for a whole simulation to one JSON file. Load it back with load_simulation.

        Args:
            synthesis: The synthesis config.
            aging: The aging config.
            pcr: The PCR config.
            sequencing: The sequencing config.
            json_path: The path to the output JSON file. Parent directories are created if needed.

        Raises:
            ValidationError: If a config is given for the wrong stage (e.g. a PCR config as synthesis).
            FileNotFoundError: If the output path is invalid or cannot be created.
        """
        # Building the model checks each config is for the right stage before anything is written
        save: SavedSimulation = SavedSimulation(
            synthesis=synthesis,
            aging=aging,
            pcr=pcr,
            sequencing=sequencing
        )

        ErrorConfigSerializer._write_json(save, json_path)


    @staticmethod
    def load_simulation(json_path: str) -> SavedSimulation:
        """
        Loads every stage's config for a whole simulation from a JSON file saved by save_simulation.

        Args:
            json_path: The path to the JSON file.

        Returns:
            The saved simulation, with each stage's config as saved.synthesis, saved.aging, saved.pcr, and
            saved.sequencing.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValidationError: If the file is not valid JSON, not a simulation file, was saved with a different format
                version, has a config under the wrong stage, or has anything missing, extra, or invalid in a config.
        """
        # Checks the whole file in one go - valid JSON, the format version, each stage, and every config value
        json_content: str = Path(json_path).read_text(encoding='utf-8')
        saved: SavedSimulation = SavedSimulation.model_validate_json(json_content)

        return saved


    @staticmethod
    def save_errors(record: ErrorRecord, json_path: str, include_sequences: bool = True) -> None:
        """
        Saves an error record to a JSON file for the user to look at - the count of each error type, then every error
        that happened (index of the strand, position, stage, error type, and the strand after the error). Using a JSON
        makes it easier to read and analyze later than just a simple text file.

        Args:
            record: The error record to save, e.g. simulator.error_record after a simulation.
            json_path: The path to the output JSON file. Parent directories are created if needed.
            include_sequences: If we want to save the strand after every error. Each one is a whole strand, so
                leaving them out makes the file a lot smaller.

        Raises:
            FileNotFoundError: If the output path is invalid or cannot be created.
        """

        # Records are already dicts, so they can go in as they are
        records: list[dict[str, (int | None) | Stage | ErrorType | str]] = record.records

        # Strands take up most of the file, so leave them out of every record if we don't want them
        if not include_sequences:
            records = []
            for error in record.records:
                without_sequence: dict[str, (int | None) | Stage | ErrorType | str] = dict(error)
                del without_sequence[ErrorDictKeys.SEQUENCE]
                records.append(without_sequence)

        # A run with no errors is still a result, so this saves an empty list rather than complaining
        save: SavedErrors = SavedErrors(counts=record.counts, records=records)
        ErrorConfigSerializer._write_json(save, json_path)
