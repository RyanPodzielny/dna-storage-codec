"""
Module for the Simulate Errors page, which runs strands through the DNA storage channel to see the errors they pick up.
There are four stages (synthesis, aging, PCR, and sequencing), each can be turned on or off, and each has its own simple
or advanced settings.
"""

from enum import StrEnum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Final, cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer
from Model.Serialize.SimulatorSerializer import ErrorConfigSerializer
from Model.Simulate.ErrorConfig import AgingConfig, ErrorConfig, PCRConfig, SequencingConfig, Stage, SynthesisConfig
from Model.Simulate.ErrorSimulator import ErrorDictKeys, ErrorSimulator, ErrorType
from View.Helpers import ViewHelpers
from View.Helpers.ViewHelpers import UploadKeys, show_input_settings, show_downloads


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class SimulatorInputs(StrEnum):
    # Keys for the JSON uploader, so we can store the uploaded file in session state and use it across pages
    JSON_UPLOAD_WIDGET = "simulator_json_upload"
    JSON_FILE_NAME = "simulator_json_file_name"
    JSON_FILE_PATH = "simulator_json_file_path"
    JSON_UPLOAD_ERROR = "simulator_json_upload_error"

    # Text uploader gets its own keys - otherwise an uploaded JSON would still be used after switching to text
    TEXT_UPLOAD_WIDGET = "simulator_text_upload"
    TEXT_FILE_NAME = "simulator_text_file_name"
    TEXT_FILE_PATH = "simulator_text_file_path"
    TEXT_UPLOAD_ERROR = "simulator_text_upload_error"


# Every key each file uploader keeps its upload under
JSON_UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=SimulatorInputs.JSON_UPLOAD_WIDGET,
    name=SimulatorInputs.JSON_FILE_NAME,
    path=SimulatorInputs.JSON_FILE_PATH,
    error=SimulatorInputs.JSON_UPLOAD_ERROR,
)
TEXT_UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=SimulatorInputs.TEXT_UPLOAD_WIDGET,
    name=SimulatorInputs.TEXT_FILE_NAME,
    path=SimulatorInputs.TEXT_FILE_PATH,
    error=SimulatorInputs.TEXT_UPLOAD_ERROR,
)



#------------------------------------------------------------------------------#
#                                                                              #
#   STAGE CONSTANTS                                                            #
#                                                                              #
#------------------------------------------------------------------------------#



# The stages in the order they are applied, so the page always shows them in the same order the simulator runs them
STAGES: Final[list[Stage]] = [Stage.SYNTHESIS, Stage.AGING, Stage.PCR, Stage.SEQUENCING]

# The config class for each stage, not an instance - so we can build a new config from the user's settings each rerun,
#   which is when pydantic validates them. Defaults come straight from them so the page always matches the model
STAGE_CONFIGS: Final[dict[Stage, type[ErrorConfig]]] = {
    Stage.SYNTHESIS: SynthesisConfig,
    Stage.AGING: AgingConfig,
    Stage.PCR: PCRConfig,
    Stage.SEQUENCING: SequencingConfig,
}
STAGE_DEFAULTS: Final[dict[Stage, ErrorConfig]] = {stage: config() for stage, config in STAGE_CONFIGS.items()}

# What the user sees for each stage
STAGE_NAMES: Final[dict[Stage, str]] = {
    Stage.SYNTHESIS: "Synthesis",
    Stage.AGING: "Aging",
    Stage.PCR: "PCR",
    Stage.SEQUENCING: "Sequencing",
}

# A short description of each stage, shown under its toggle when it is on
STAGE_DESCRIPTIONS: Final[dict[Stage, str]] = {
    Stage.SYNTHESIS: "Writes each strand base by base and makes copies of each one. Most insertions and deletions come"
                     " from here, deletions being the most common.",
    Stage.AGING: "The strands sitting in storage. Aging mostly loses whole strands rather than adding errors to them, "
                 " so every rate here is per year.",
    Stage.PCR: "Copies every strand each cycle. Rates are per base per copy, so an error made in an early cycle gets"
               " copied into everything after it.",
    Stage.SEQUENCING: "Reads strands back from the pool. Nearly all of its errors are substitutions, and strands with"
                      " more copies get read more.",
}


# Both modes every stage can be in
class ModeNames(StrEnum):
    SIMPLE = "Simple"
    ADVANCED = "Advanced"


# Session state keys for each stage, every stage gets its own by adding the stage after "simulator_"
class StageKeys(StrEnum):
    ENABLED = "enabled"
    MODE = "mode"


def _stage_key(stage: Stage, name: str) -> str:
    """
    Makes the session state key for one of a stage's widgets. Session state is shared across every page and every stage
    uses the same settings, so each one needs its own key or they would overwrite each other. For example, sub for 
    syntehsis would be "simulator_synthesis_substitution_rate". Basically we can reuse same widget but different keys.

    Args:
        stage: The stage the widget is for.
        name: The name of the widget, usually the config field it sets.

    Returns:
        The key, e.g. "simulator_synthesis_substitution_rate".
    """
    return f"simulator_{stage}_{name}"



#------------------------------------------------------------------------------#
#                                                                              #
#   SIMPLE MODE SETTINGS                                                       #
#                                                                              #
#------------------------------------------------------------------------------#



# Rates are tiny (some are 0.000001), so no sliders - and %g so they don't show trailing 0's like %f would
RATE_FORMAT: Final[str] = "%.6g"
RATE_STEP: Final[float] = 0.0001
PROBABILITY_RANGE: Final[tuple[float, float]] = (0.0, 1.0)

# Ranges for the stage specific settings. Kept low as we can't simulate millions of strands, PCR especially as every
#   cycle can double the number of strands.
DEPTH_RANGE: Final[tuple[int, int]] = (1, 100)
CYCLES_RANGE: Final[tuple[int, int]] = (0, 10)
YEARS_RANGE: Final[tuple[float, float]] = (0.0, 100000.0)


# The rates every stage shares in simple mode, the values are the config fields they set
class RateFields(StrEnum):
    SUBSTITUTION = "substitution_rate"
    INSERTION = "insertion_rate"
    DELETION = "deletion_rate"
    DROPOUT = "dropout_rate"


# Label of each rate
RATE_LABELS: Final[dict[RateFields, str]] = {
    RateFields.SUBSTITUTION: "Substitution rate:",
    RateFields.INSERTION: "Insertion rate:",
    RateFields.DELETION: "Deletion rate:",
    RateFields.DROPOUT: "Dropout rate:"
}

# Tooltip of each rate
RATE_HELP: Final[dict[RateFields, str]] = {
    RateFields.SUBSTITUTION: "Chance each base is substituted for a different base.",
    RateFields.INSERTION: "Chance each base gets an extra base inserted right after it.",
    RateFields.DELETION: "Chance each base is deleted.",
    RateFields.DROPOUT: "Chance a whole strand is lost, no mater how long it is."
}


def _show_rate_settings(stage: Stage) -> dict[str, Any]:
    """
    The substitution, insertion, deletion, and dropout rates - every stage has these in simple mode, so every stage
    uses this. One rate per column, so they all fit on one row.

    Args:
        stage: The stage the rates are for, used for the keys and defaults.

    Returns:
        The rates picked, keyed by the config field they set.
    """
    defaults: ErrorConfig = STAGE_DEFAULTS[stage]

    # Each rate's default for this stage, written out so it is clear which config field each one comes from
    rate_defaults: dict[RateFields, float] = {
        RateFields.SUBSTITUTION: defaults.substitution_rate,
        RateFields.INSERTION: defaults.insertion_rate,
        RateFields.DELETION: defaults.deletion_rate,
        RateFields.DROPOUT: defaults.dropout_rate,
    }

    # Create a dictionary to store the rates
    rates: dict[str, int | float] = {}
    columns: list[Any] = st.columns(len(RateFields))

    # Go over all 4 rates and make a number input for each one. They are probabilities, so 0 to 1.
    for column, field in zip(columns, RateFields):
        with column:
            rates[field] = st.number_input(
                label=RATE_LABELS[field],
                min_value=PROBABILITY_RANGE[0],
                max_value=PROBABILITY_RANGE[1],
                step=RATE_STEP,
                value=rate_defaults[field],         # Use config default as default for the width
                format=RATE_FORMAT,                 # No trailing 0's
                key=_stage_key(stage, field),
                help=RATE_HELP[field],
                persist_state="session"             # Keep value for whole session
            )

    return rates


def _show_stage_specific_settings(stage: Stage) -> dict[str, Any]:
    """
    The one setting each stage has on its own in simple mode - how many copies, how long it is stored, how many cycles,
    or how many reads.

    Args:
        stage: The stage to show the setting for.

    Returns:
        The setting picked, keyed by the config field it sets.
    """
    settings: dict[str, int | float] = {}
    match stage:
        # Syntehsis just has the depth, how many copies of each strand are made
        case Stage.SYNTHESIS:
            synthesis_defaults: SynthesisConfig = SynthesisConfig()
            settings = {"synthesis_depth": st.number_input(
                label="Synthesis depth:",
                min_value=DEPTH_RANGE[0],
                max_value=DEPTH_RANGE[1],
                step=1,
                value=synthesis_defaults.synthesis_depth,
                key=_stage_key(stage, "synthesis_depth"),
                help="How many copies are made of each strand.",
                persist_state="session"
            )}

        # Aging just has the years
        case Stage.AGING:
            aging_defaults: AgingConfig = AgingConfig()
            settings = {"years": st.number_input(
                label="Years stored:",
                min_value=YEARS_RANGE[0],
                max_value=YEARS_RANGE[1],
                step=10.0,
                value=aging_defaults.years,
                format=RATE_FORMAT,                 # Same here, no trailing 0's
                key=_stage_key(stage, "years"),
                help="How long the strands are stord. Every rate is per year, so 0 years means no aging at all.",
                persist_state="session"
            )}

        # PCR has cycles, how many times every strand is copied
        case Stage.PCR:
            pcr_defaults: PCRConfig = PCRConfig()
            settings = {"cycles": st.number_input(
                label="Cycles:",
                min_value=CYCLES_RANGE[0],
                max_value=CYCLES_RANGE[1],
                step=1,
                value=pcr_defaults.cycles,
                key=_stage_key(stage, "cycles"),
                help="How many times every strand is copied. Each cycle can double the number of strands, so this"
                     " gets slow fast.",
                persist_state="session"
            )}

        # Sequencing has read depth, basically how many of the strands we currently have are read
        case Stage.SEQUENCING:
            sequencing_defaults: SequencingConfig = SequencingConfig()
            settings = {"read_depth": st.number_input(
                label="Read depth:",
                min_value=DEPTH_RANGE[0],
                max_value=DEPTH_RANGE[1],
                step=1,
                value=sequencing_defaults.read_depth,
                key=_stage_key(stage, "read_depth"),
                help="How many reads each original strand gets.",
                persist_state="session"
            )}

        case _:
            raise ValueError(f"Unknown stage: {stage}")

    return settings


def _show_simple_settings(stage: Stage) -> dict[str, Any]:
    """
    Simple mode settings - the four shared rates, then the one setting the stage has on its own.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    settings: dict[str, Any] = _show_rate_settings(stage)
    settings.update(_show_stage_specific_settings(stage))

    return settings



#------------------------------------------------------------------------------#
#                                                                              #
#   ADVANCED MODE SETTINGS                                                     #
#                                                                              #
#------------------------------------------------------------------------------#



def _show_advanced_settings():
    # TODO: add the advanced settings
    st.info(f"not implemented, need to add")



#------------------------------------------------------------------------------#
#                                                                              #
#   STAGE SETTINGS                                                             #
#                                                                              #
#------------------------------------------------------------------------------#



def show_stage_settings(stage: Stage) -> ErrorConfig | None:
    """
    One stage's settings. A toggle to turn the stage on or off, and when on, a control to pick simple or advanced mode
    and that mode's settings. Shows nothing else when the stage is off.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The config built with the picked settings (a turned off config if the stage is off), or None if a setting is
        not valid.
    """
    # Toggle for the stage, on by default
    enabled: bool = st.toggle(
        label=f"**{STAGE_NAMES[stage]}**",      # Make it bold so its more clear
        value=STAGE_DEFAULTS[stage].enabled,    # Use config default as default for the toggle (will be on)
        key=_stage_key(stage, StageKeys.ENABLED),
        help=f"Turn the {STAGE_NAMES[stage].lower()} stage on or off.",
        persist_state="session",                # Keep value for whole session
    )

    # Stage is off - nothing to show, and the simulator just skips it
    if not enabled:
        return STAGE_CONFIGS[stage](enabled=False)

    # Add our little description, so the user knows what it does
    st.caption(STAGE_DESCRIPTIONS[stage])

    # Simple or advanced, same way the input method is picked
    mode: str = st.segmented_control(
        label="Mode:",
        options=[ModeNames.SIMPLE, ModeNames.ADVANCED],
        default=ModeNames.ADVANCED if STAGE_DEFAULTS[stage].advanced_mode else ModeNames.SIMPLE,
        required=True,                          # Must be either one
        key=_stage_key(stage, StageKeys.MODE),
        persist_state="session",
    )

    # Every mode gives back the config fields it sets, so the config is built the same way for both
    settings: dict[str, Any] | None
    match mode:
        case ModeNames.SIMPLE:
            settings = _show_simple_settings(stage)
        case ModeNames.ADVANCED:
            settings = _show_advanced_settings()
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # Advanced isn't ready, already let the user know
    if settings is None:
        return None

    # Pydantic checks everything again, the widgets stop most bad values but not e.g. the rates summing past 1
    try:
        return STAGE_CONFIGS[stage](
            enabled=True,
            advanced_mode=mode == ModeNames.ADVANCED,
            **settings,             # Need to unpack as stages have specific settings that we got before
        )
    except ValidationError as error:
        st.error(f"A {STAGE_NAMES[stage].lower()} setting is not valid: {error.errors()[0]['msg']}")
        return None



#------------------------------------------------------------------------------#
#                                                                              #
#   SIMULATION                                                                 #
#                                                                              #
#------------------------------------------------------------------------------#



# Session state keys for the last run
class SimulateResultKeys(StrEnum):
    # Session state is shared across every page, so these need to be different from the other pages' keys
    IN_PATH = "simulator_in_path"
    NUM_STRANDS = "simulator_num_strands"
    READS = "simulator_reads"
    COUNTS = "simulator_counts"
    RECORDS = "simulator_records"
    JSON = "simulator_json"
    ERRORS_JSON = "simulator_errors_json"
    ERROR = "simulator_error"


def simulate(in_path: Path, is_json: bool, configs: dict[Stage, ErrorConfig]) -> None:
    """
    Loads the strands, runs them through every stage, and saves the results to session state. The errors are saved to a
    JSON for the user to download, and if the input is a JSON file, a copy of it with its oligos replaced by the reads
    is saved too. Any error is saved, so it can be shown on the page instead of crashing it.

    Args:
        in_path: The path to the JSON or text file to simulate.
        is_json: Whether the input is a JSON file saved from the Encoding page, False if it is a text file of oligos.
        configs: The config of every stage, turned off stages included.
    """
    # Forget the last run first, so an error never shows next to old results
    for key in SimulateResultKeys:
        st.session_state.pop(key, None)

    # Remember which file these results are for, so they are hidden if the user changes the input
    st.session_state[SimulateResultKeys.IN_PATH] = in_path

    # Try to load and simulate - show error if anything goes wrong
    json_path: Path | None = None
    try:
        # Strands come from the JSON's oligos, or one per line from the text file
        strands: list[str]
        if is_json:
            strands = Serializer.load_framed_oligos(str(in_path))
        else:
            strands = Serializer.load_segments_from_text(str(in_path))

        # A file of only blank lines passes the upload check, but there is nothing to simulate
        if not strands:
            st.session_state[SimulateResultKeys.ERROR] = "The input has no oligos to simulate."
            return

        # Each config is already the right stage (built from STAGE_CONFIGS), the dict just can't tell the type checker
        simulator: ErrorSimulator = ErrorSimulator(
            synthesis=cast(SynthesisConfig, configs[Stage.SYNTHESIS]),
            aging=cast(AgingConfig, configs[Stage.AGING]),
            PCR=cast(PCRConfig, configs[Stage.PCR]),
            sequencing=cast(SequencingConfig, configs[Stage.SEQUENCING]),
        )

        # And now preform the work
        reads: list[str] = simulator.simulate(strands)

        # Both downloads go in the same temporary folder
        out_directory: Path = Path(mkdtemp(prefix="dna_codec_"))
        errors_path: Path = out_directory / f"{in_path.stem}-errors.json"
        ErrorConfigSerializer.save_errors(simulator.error_record, str(errors_path))

        # The reads replace the JSON's oligos (none of the old ones kept), so it can be clustered straight away
        if is_json:
            json_path = out_directory / f"{in_path.stem}-simulated.json"
            Serializer.save_with_replaced_oligos(str(in_path), reads, str(json_path))

    # The JSON no longer matches what is expected (it is checked on upload, but the encoded one is not)
    except ValidationError as error:
        st.session_state[SimulateResultKeys.ERROR] = f"The JSON file is not valid: {error.errors()[0]['msg']}"
        return

    # Our own errors, or the simulator failing on the strands
    except (Error, ValueError) as error:
        st.session_state[SimulateResultKeys.ERROR] = f"{type(error).__name__}: {error}"
        return

    # Save the results to session state so they can be shown on the page
    st.session_state[SimulateResultKeys.NUM_STRANDS] = len(strands)
    st.session_state[SimulateResultKeys.READS] = reads
    st.session_state[SimulateResultKeys.COUNTS] = simulator.error_record.counts
    st.session_state[SimulateResultKeys.RECORDS] = simulator.error_record.records
    st.session_state[SimulateResultKeys.JSON] = json_path
    st.session_state[SimulateResultKeys.ERRORS_JSON] = errors_path

    # The reads are now the oligos the rest of the pages use (e.g. clustering), same as a fresh encode
    if json_path is not None:
        st.session_state[ViewHelpers.ENCODED_JSON_PATH_KEY] = json_path



#------------------------------------------------------------------------------#
#                                                                              #
#   RESULTS                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



# Every error type in the order the record counts them, so the charts and metrics always line up
ERROR_TYPES: Final[list[ErrorType]] = list(ErrorType)

# What the user sees for each error type
ERROR_TYPE_NAMES: Final[dict[ErrorType, str]] = {
    ErrorType.SUBSTITUTION: "Substitutions",
    ErrorType.INSERTION: "Insertions",
    ErrorType.DELETION: "Deletions",
    ErrorType.DROPOUT: "Dropouts",
    ErrorType.TRUNCATION: "Truncations"
}

# Tooltip of each error type's metric
ERROR_TYPE_HELP: Final[dict[ErrorType, str]] = {
    ErrorType.SUBSTITUTION: "Bases swapped for a different base.",
    ErrorType.INSERTION: "Extra bases inserted into a strand.",
    ErrorType.DELETION: "Bases deleted from a strand.",
    ErrorType.TRUNCATION: "Strands cut short, everything past the cut is gone.",
    ErrorType.DROPOUT: "Whole strands lost."
}


def _show_summary(num_strands: int, reads: list[str], counts: dict[ErrorType, int]) -> None:
    """
    The main numbers from the run, strands in and reads out on the first row, then how many of each error happened on
    the second.

    Args:
        num_strands: The number of strands that went in.
        reads: The reads that came out.
        counts: How many of each error type happened.
    """
    strands_column: Any
    reads_column: Any
    errors_column: Any
    strands_column, reads_column, errors_column = st.columns(3)

    # Strands that went in
    strands_column.metric(
        label="Strands",
        value=f"{num_strands:,}",
        help="How many strands went into the simulation.",
    )

    # Reads that came out
    reads_column.metric(
        label="Reads",
        value=f"{len(reads):,}",
        help="How many reads came out of the last stage. Copies made in synthesis and PCR can make this higher than"
             " the strands that went in, and dropouts can make it lower.",
    )

    # Every error added up
    errors_column.metric(
        label="Total Errors",
        value=f"{sum(counts.values()):,}",
        help="Every error from every stage added up.",
    )

    # One column per error type
    type_columns: list[Any] = st.columns(len(ERROR_TYPES))
    for column, error_type in zip(type_columns, ERROR_TYPES):
        column.metric(
            label=ERROR_TYPE_NAMES[error_type],
            value=f"{counts[error_type]:,}",
            help=ERROR_TYPE_HELP[error_type],
        )


def _show_stage_chart(records_frame: pd.DataFrame) -> None:
    """
    Stacked bar chart of how many errors each stage added, split by error type.

    Args:
        records_frame: Every error, one row each, with a "stage" and "error_type" column.
    """
    st.markdown("**Errors by stage**")

    # Table of how many of each error type each stage added, one row per stage
    stage_counts: pd.DataFrame = pd.crosstab(
        index=records_frame[ErrorDictKeys.STAGE],
        columns=records_frame[ErrorDictKeys.ERROR_TYPE],
    )

    # Stages or types with no errors are missing, so add them back as 0 - also puts them in the order they run
    stage_counts = stage_counts.reindex(index=STAGES, columns=ERROR_TYPES, fill_value=0)

    # Rename to what the user sees
    stage_counts.index = [STAGE_NAMES[stage] for stage in STAGES]
    stage_counts.columns = [ERROR_TYPE_NAMES[error_type] for error_type in ERROR_TYPES]

    # Horizontal so the stage names read left to right, Streamlit turns them sideways otherwise. Labels still go by the
    #   data (x is the stage), Streamlit moves them to the right axis itself
    st.bar_chart(
        data=stage_counts,
        x_label="Stage",
        y_label="Errors",
        horizontal=True,
        sort=False,                 # Keep the stages in the order they run, not alphabetical
        stack=True,
    )


def _show_error_table(records_frame: pd.DataFrame) -> None:
    """
    Every error in the order it happened, hidden in an expander as it can be a lot of rows.

    Args:
        records_frame: Every error, one row each.
    """
    # want to make it an expandable thing so it doesnt hog the screen. Kinda unimportant information, dont need to show
    with st.expander(f"Every error ({len(records_frame):,})"):
        # Rename to what the user sees - copy so the charts frame isn't changed
        table: pd.DataFrame = records_frame.copy()
        # Kinda jank but we want to show a better name than the enum, so we are just replacing the values for formatting
        table[ErrorDictKeys.STAGE] = [STAGE_NAMES[stage] for stage in table[ErrorDictKeys.STAGE]]
        table[ErrorDictKeys.ERROR_TYPE] = [ERROR_TYPE_NAMES[error] for error in table[ErrorDictKeys.ERROR_TYPE]]

        # Basically just a better table
        st.dataframe(
            data=table,
            hide_index=True,
            # Basically make it look better for the user
            column_config={
                ErrorDictKeys.INDEX: st.column_config.NumberColumn("Strand", help="Index of the original strand."),
                ErrorDictKeys.POSITION: st.column_config.NumberColumn("Position", help="Empty for dropouts."),
                ErrorDictKeys.STAGE: st.column_config.TextColumn("Stage"),
                ErrorDictKeys.ERROR_TYPE: st.column_config.TextColumn("Error"),
                ErrorDictKeys.SEQUENCE: st.column_config.TextColumn("Strand after error"),
            },
        )


def show_results(num_strands: int,
                 reads: list[str],
                 counts: dict[ErrorType, int],
                 records: list[dict[str, Any]]) -> None:
    """
    Report of the last run - the main numbers, where the errors came from, where in the strand they happened, and a
    table of every error.

    Args:
        num_strands: The number of strands that went in.
        reads: The reads that came out.
        counts: How many of each error type happened.
        records: Every error that happened, as the error record saves them.
    """
    _show_summary(num_strands, reads, counts)

    # A run with no errors at all has nothing to chart
    if not records:
        st.info("No errors happened in this run.")
        return

    # One row per error, the columns are the record's keys - used by both charts and the table
    records_frame: pd.DataFrame = pd.DataFrame.from_records(records, columns=list(ErrorDictKeys))

    # Show charts of the errors and just the dataframe of them
    _show_stage_chart(records_frame)
    _show_error_table(records_frame)


#------------------------------------------------------------------------------#
#                                                                              #
#   SHOW UI                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



# Page title and info
st.title("Simulate Errors")
# TODO: add a description of the page
st.write("TODO: add a descirption of the page")


# Input section
st.subheader("Input", divider="gray")

# Where the file to simulate can be found - or None if nothing valid yet
in_path: Path | None
is_json: bool
in_path, is_json = show_input_settings("simulator_input_method", JSON_UPLOAD_KEYS, TEXT_UPLOAD_KEYS)

# A run sets its simulated JSON as the sessions oligos (so clustering picks it up), which would make the next run
#   simulate the already simulated reads and stack the errors. Swap it back to the oligos that run started from
last_in_path: Path | None = st.session_state.get(SimulateResultKeys.IN_PATH, None)
last_json_path: Path | None = st.session_state.get(SimulateResultKeys.JSON, None)
#   Only a JSON input gets a simulated JSON, so the original is a JSON too
if last_in_path is not None and last_json_path is not None and in_path == last_json_path:
    in_path = last_in_path
    st.caption(f"Simulating the original oligos ({last_in_path.name}), not the last simulated ones.")


# Settings section
st.subheader("Settings", divider="gray")

# Every stage gets the whole width, one after another in the order they run. None if a stage's settings aren't valid
stage_configs: dict[Stage, ErrorConfig | None] = {}
for stage in STAGES:
    with st.container(border=True):
        stage_configs[stage] = show_stage_settings(stage)


# Simulate section, only enabled if there is an input and every stage's settings are valid
st.subheader("Simulate", divider="gray")

# Only the stages with valid settings - if any are missing the simulation can't run
valid_configs: dict[Stage, ErrorConfig] = {
    stage: config for stage, config in stage_configs.items() if config is not None
}

# If all stages have valid settings, we can simulate
ready: bool = in_path is not None and len(valid_configs) == len(STAGES)
if in_path is None:
    st.info("Upload a file above, or encode something on the Encoding page, to simulate errors.")
if len(valid_configs) != len(STAGES):
    st.info("Fix the stage settings above to simulate errors.")

# The actual button, disabled if not ready. When pressed it runs the simulation and shows the results below
if st.button("Simulate Errors", type="primary", disabled=not ready):
    # When pressed show our little spinner so we know that we are doing something
    with st.spinner("Simulating Errors..."):
        # Same as the convergence page, can't reuse ready for typing reasons
        if in_path is not None:
            simulate(in_path, is_json, valid_configs)

# Only show results for the input currently picked, so old results don't show next to a new file. Read again, as
#   pressing the button just replaced the last run
last_in_path = st.session_state.get(SimulateResultKeys.IN_PATH, None)
last_json_path = st.session_state.get(SimulateResultKeys.JSON, None)
if last_in_path == in_path:
    # If there was an error show
    if SimulateResultKeys.ERROR in st.session_state:
        st.error(f"Simulation failed. {st.session_state[SimulateResultKeys.ERROR]}")

    # Everything was great - show the report and the downloads
    if SimulateResultKeys.READS in st.session_state and last_in_path is not None:
        st.success("Simulation finished successfully!")

        show_results(
            num_strands=st.session_state[SimulateResultKeys.NUM_STRANDS],
            reads=st.session_state[SimulateResultKeys.READS],
            counts=st.session_state[SimulateResultKeys.COUNTS],
            records=st.session_state[SimulateResultKeys.RECORDS],
        )

        # Named after what was simulated, not the simulated JSON it might be using now
        show_downloads(
            in_path=last_in_path,
            strands=st.session_state[SimulateResultKeys.READS],
            json_path=last_json_path,
            json_help="The JSON with the simulated reads, can be clustered on the Clustering and Consensus page.",
            strand_help="The simulated reads, one per line. Cannot be uploaded on the decode page.",
            suffix="simulated",
            error_path=st.session_state[SimulateResultKeys.ERRORS_JSON],
            error_help="How many of each error happened, then every error (strand, position, stage, error type, and the"
                       " strand after it).",
        )


# TODO: add this in for the conditionals
from streamlit_segment_slider import segment_slider

cuts = segment_slider(
    "Roughly how is it split?",
    min_value=0,
    max_value=100,
    values=[25, 75],
    segment_labels=["Cash", "Growth", "Income"],
    key="my_split",
)

cash_pct = cuts[0]
growth_pct = cuts[1] - cuts[0]
income_pct = 100 - cuts[1]