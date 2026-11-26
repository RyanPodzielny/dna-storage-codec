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

from Model.Encode.Methods.Encoding import Base
from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer
from Model.Serialize.SimulatorSerializer import ErrorConfigSerializer, SavedSimulation
from Model.Simulate import ErrorConfigs
from Model.Simulate.ErrorConfigs import AgingConfig, CoverageDistribution, GCWindow, ErrorConfig, PCRConfig, \
    SequencingConfig, Stage, SynthesisConfig
from Model.Simulate.ErrorConfigs import AgingFields, ErrorConfigFields, GCWindowFields, PCRFields, SequencingFields, \
    SynthesisFields
from Model.Simulate.ErrorSimulator import ErrorDictKeys, ErrorSimulator, ErrorType
from View.Helpers import ViewHelpers
from View.Helpers.ViewHelpers import UploadKeys, coverage_settings, file_uploader, length_settings, rate_input, \
    per_base_rate, show_downloads, show_input_settings, split_slider


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUTS                                                                     #
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

    # Settings uploader gets its own keys too, it fills in every stage's settings instead of giving strands to simulate
    SETTINGS_UPLOAD_WIDGET = "simulator_settings_upload"
    SETTINGS_FILE_NAME = "simulator_settings_file_name"
    SETTINGS_FILE_PATH = "simulator_settings_file_path"
    SETTINGS_UPLOAD_ERROR = "simulator_settings_upload_error"

    # The configs from the uploaded settings, used as every stage's defaults. And the file they came from, so they are
    #   only filled in once per file - anything the user changes after is kept
    LOADED_SETTINGS = "simulator_loaded_settings"
    SETTINGS_FROM = "simulator_settings_from"

    # Goes up by one each time settings are uploaded, and is part of every stage widget's key (see _stage_key)
    SETTINGS_VERSION = "simulator_settings_version"


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

# Settings uploader is a bit different - it fills in every stage's settings instead of giving strands to simulate
SETTINGS_UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=SimulatorInputs.SETTINGS_UPLOAD_WIDGET,
    name=SimulatorInputs.SETTINGS_FILE_NAME,
    path=SimulatorInputs.SETTINGS_FILE_PATH,
    error=SimulatorInputs.SETTINGS_UPLOAD_ERROR,
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

# Every widget starts from these. Built-in defaults come straight from the configs, but once the user uploads settings
#   those are the defaults instead - so every widget starts from the uploaded values (see _load_settings)
STAGE_DEFAULTS: Final[dict[Stage, ErrorConfig]] = st.session_state.get(
    SimulatorInputs.LOADED_SETTINGS,
    {stage: config() for stage, config in STAGE_CONFIGS.items()},
)

# What the user sees for each stage. Member names match Stage's, so look one up with StageNames[stage.name]
class StageNames(StrEnum):
    SYNTHESIS = "Synthesis"
    AGING = "Aging"
    PCR = "PCR"
    SEQUENCING = "Sequencing"

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
    SEED = "seed"


def _stage_key(stage: Stage, name: str) -> str:
    """
    Makes the session state key for one of a stage's widgets. Session state is shared across every page and every stage
    uses the same settings, so each one needs its own key or they would overwrite each other. For example, sub for 
    syntehsis would be "simulator_synthesis_0_substitution_rate". Basically we can reuse same widget but different keys.

    The number is the settings version, which goes up each time settings are uploaded. Streamlit keeps a keyed widget
    (e.g. a toggle) as the same widget even after its session state is cleared, so the browser keeps showing the old
    value. A new key makes it a new widget, so it is drawn again from the uploaded values.

    Args:
        stage: The stage the widget is for.
        name: The name of the widget, usually the config field it sets.

    Returns:
        The key, e.g. "simulator_synthesis_0_substitution_rate".
    """
    version: int = st.session_state.get(SimulatorInputs.SETTINGS_VERSION, 0)
    return f"simulator_{stage}_{version}_{name}"



#------------------------------------------------------------------------------#
#                                                                              #
#   SIMPLE MODE SETTINGS                                                       #
#                                                                              #
#------------------------------------------------------------------------------#



# Ranges for the stage specific settings. Kept low as we can't simulate millions of strands, PCR especially as every
#   cycle can double the number of strands.
DEPTH_RANGE: Final[tuple[int, int]] = (1, 100)
CYCLES_RANGE: Final[tuple[int, int]] = (0, 10)
YEARS_RANGE: Final[tuple[float, float]] = (0.0, 100000.0)


# The rates every stage shares in simple mode, the values are the config fields they set (taken from the config's own
#   field names, so they always line up)
class RateFields(StrEnum):
    SUBSTITUTION = ErrorConfigFields.SUBSTITUTION_RATE
    INSERTION = ErrorConfigFields.INSERTION_RATE
    DELETION = ErrorConfigFields.DELETION_RATE
    DROPOUT = ErrorConfigFields.DROPOUT_RATE


# Label of each rate. Member names match RateFields', so look one up with RateLabels[field.name]
class RateLabels(StrEnum):
    SUBSTITUTION = "Substitution rate:"
    INSERTION = "Insertion rate:"
    DELETION = "Deletion rate:"
    DROPOUT = "Dropout rate:"

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
    DEFAULT: ErrorConfig = STAGE_DEFAULTS[stage]

    # Each rate's default for this stage, written out so it is clear which config field each one comes from
    RATE_DEFAULTS: dict[RateFields, float] = {
        RateFields.SUBSTITUTION: DEFAULT.substitution_rate,
        RateFields.INSERTION: DEFAULT.insertion_rate,
        RateFields.DELETION: DEFAULT.deletion_rate,
        RateFields.DROPOUT: DEFAULT.dropout_rate,
    }

    # Create a dictionary to store the rates
    rates: dict[str, Any] = {}
    columns: list[Any] = st.columns(len(RateFields))

    # Go over all 4 rates and make a number input for each one. They are probabilities, so 0 to 1.
    for column, field in zip(columns, RateFields):
        with column:
            rates[field] = st.number_input(
                label=RateLabels[field.name],
                min_value=ViewHelpers.PROBABILITY_RANGE[0],
                max_value=ViewHelpers.PROBABILITY_RANGE[1],
                step=ViewHelpers.RATE_STEP,
                value=RATE_DEFAULTS[field],         # Use config default as default for the width
                format=ViewHelpers.RATE_FORMAT,                 # No trailing 0's
                key=_stage_key(stage, field),
                help=RATE_HELP[field],
                persist_state="session"             # Keep value for whole session
            )

    return rates


def _show_simple_stage_specific_settings(stage: Stage) -> dict[str, Any]:
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
            SYTHNESIS_DEFAULTS: SynthesisConfig = cast(SynthesisConfig, STAGE_DEFAULTS[stage])
            settings = {"synthesis_depth": st.number_input(
                label="Synthesis depth:",
                min_value=DEPTH_RANGE[0],
                max_value=DEPTH_RANGE[1],
                step=1,
                value=SYTHNESIS_DEFAULTS.synthesis_depth,
                key=_stage_key(stage, "synthesis_depth"),
                help="How many copies are made of each strand.",
                persist_state="session"
            )}

        # Aging just has the years
        case Stage.AGING:
            AGING_DEFAULTS: AgingConfig = cast(AgingConfig, STAGE_DEFAULTS[stage])
            settings = {"years": st.number_input(
                label="Years stored:",
                min_value=YEARS_RANGE[0],
                max_value=YEARS_RANGE[1],
                step=10.0,
                value=AGING_DEFAULTS.years,
                format=ViewHelpers.RATE_FORMAT,                 # Same here, no trailing 0's
                key=_stage_key(stage, "years"),
                help="How long the strands are stord. Every rate is per year, so 0 years means no aging at all.",
                persist_state="session"
            )}

        # PCR has cycles, how many times every strand is copied
        case Stage.PCR:
            PCR_DEFAULTS: PCRConfig = cast(PCRConfig, STAGE_DEFAULTS[stage])
            settings = {"cycles": st.number_input(
                label="Cycles:",
                min_value=CYCLES_RANGE[0],
                max_value=CYCLES_RANGE[1],
                step=1,
                value=PCR_DEFAULTS.cycles,
                key=_stage_key(stage, "cycles"),
                help="How many times every strand is copied. Each cycle can double the number of strands, so this"
                     " gets slow fast.",
                persist_state="session"
            )}

        # Sequencing has read depth, basically how many of the strands we currently have are read
        case Stage.SEQUENCING:
            SEQUENCING_DEFAULTS: SequencingConfig = cast(SequencingConfig, STAGE_DEFAULTS[stage])
            settings = {"read_depth": st.number_input(
                label="Read depth:",
                min_value=DEPTH_RANGE[0],
                max_value=DEPTH_RANGE[1],
                step=1,
                value=SEQUENCING_DEFAULTS.read_depth,
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
    settings.update(_show_simple_stage_specific_settings(stage))

    return settings



#------------------------------------------------------------------------------#
#                                                                              #
#   ADVANCED MODE SETTINGS                                                     #
#                                                                              #
#------------------------------------------------------------------------------#



class AdvancedSettingNames(StrEnum):
    SUBSTITUTION = "Substitution"
    INSERTION = "Insertion"
    DELETION = "Deletion"
    HOMOPOLYMERS = "Homopolymers"
    GC_CONTENT = "GC Content"
    POSITION_BIAS = "Position Bias"
    STRAND_LOSS = "Strand Loss"

    STAGE_SPECIFIC = "Stage Specific"


ADVANCED_SETTINGS_LABELS: Final[list[AdvancedSettingNames]] = [
    AdvancedSettingNames.SUBSTITUTION,
    AdvancedSettingNames.INSERTION,
    AdvancedSettingNames.DELETION,
    AdvancedSettingNames.HOMOPOLYMERS,
    AdvancedSettingNames.GC_CONTENT,
    AdvancedSettingNames.POSITION_BIAS,
    AdvancedSettingNames.STRAND_LOSS,
    
    AdvancedSettingNames.STAGE_SPECIFIC
]

# Range for how long an insertion can be. Nearly all are a single base, so this just needs a sane cap
INSERTION_LENGTH_RANGE: Final[tuple[int, int]] = (1, 20)

# Range for how long a deletion can be. Can be a lot longer than an insertion, as it can run to the end of the strand
DELETION_LENGTH_RANGE: Final[tuple[int, int]] = (1, 1000)

# Range for a homopolymer run length. A run of 1 is just a base, so 2 is the shortest homopolymer
RUN_LENGTH_RANGE: Final[tuple[int, int]] = (2, 50)

# How much a homopolymer multiplier moves per step
MULTIPLIER_STEP: Final[float] = 0.5

# Range for the GC window size, much bigger than a strand is pointless as it gets cut at the ends anyway
WINDOW_SIZE_RANGE: Final[tuple[int, int]] = (1, 200)


def _overall_substitution_rate(stage: Stage) -> float:
    """
    The overall substitution rate for a stage, used as the starting point for each base's rate. The user can change
    each base's rate, but this is the default.

    Args:
        stage: The stage to get the overall substitution rate for.

    Returns:
        The overall substitution rate for the stage.
    """
    overall_rate: float = rate_input(
        label="Overall substitution rate:",
        key=_stage_key(stage, "overall_substitution_rate"),
        default=STAGE_DEFAULTS[stage].substitution_rate,
        help="The overall rate at which substitutions occur - overriden by per base rates below.",
    )

    return overall_rate


# SUBSITUTION SETTINGS --------------------------------------------------------#


def _conditional_substitution_rate(stage: Stage) -> dict[Base, dict[Base, float]]:
    """
    The conditional probabilities of what a base becomes when it is substituted. For example, A has a 0.5 chance of
    becoming C, a 0.3 chance of becoming G, and a 0.2 chance of becoming T.

    Args:
        stage: The stage to show the settings for.
    
    Returns:
        The conditional probabilities picked, keyed by the config field they set.
    """
    # Show what setting we are changing
    st.markdown("What each base becomes when substituted:")

    # One slider per base, split into a segment for each of the three other bases it could become. The segments cover
    #   0 to 1 with no gaps, so each row always adds to 1 - no need to check or scale it. Two sliders per row
    conditional: dict[Base, dict[Base, float]] = {}
    slider_columns: Any = st.columns(2)

    # For each base and their position (2 per column)
    for position, base in enumerate(ErrorConfigs.DNA_BASES):
        # Left, right, left, right - so A and C on the first row, G and T on the second
        with slider_columns[position % 2]:
            # A base can't become itself. Not a set difference - sets have no order, so the segments could come out in
            #   a different order every time the app starts (and not match the saved cuts)
            other_bases: list[Base] = list(set(ErrorConfigs.DNA_BASES) - {base})

            # Now show with the slider, starting at the defaults
            chances: list[float] = split_slider(
                label=f"{base} becomes:",
                key=_stage_key(stage, f"substitution_{base}"),
                segment_labels=[str(base) + str(other_base) for other_base in other_bases],
                default_chances=[STAGE_DEFAULTS[stage].substitution_conditional[base][other_base] 
                                 for other_base in other_bases],
            )

            # Chances come back in the same order as the bases
            conditional[base] = {}
            for other_base, chance in zip(other_bases, chances):
                conditional[base][other_base] = chance

    return conditional


def _show_substitution_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced substitution settings - a rate for each base (starting at the overall rate), and what each base becomes
    when it is substituted.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # OVERALL RATE - get overall rate first, so it can be used as starting point for each base's rate
    overall_rate: float = _overall_substitution_rate(stage)

    # PER BASE RATES - overides overall rate, so each base can have its own rate. Is its own thing
    per_base: dict[Base, float] = per_base_rate(
        name="substitution",
        key=_stage_key(stage, "substitution_per_base"),
        stage_rate=STAGE_DEFAULTS[stage].substitution_rate_per_base,
        default=overall_rate
    )

    # WHAT EACH BASE BECOMES - conditional probabilities, e.g. A has a 0.5 chance of becoming C, 0.3 of G, and 0.2 of T
    conditional = _conditional_substitution_rate(stage)

    return {
        ErrorConfigFields.SUBSTITUTION_RATE: overall_rate,
        ErrorConfigFields.SUBSTITUTION_RATE_PER_BASE: per_base,
        ErrorConfigFields.SUBSTITUTION_CONDITIONAL: conditional,
    }


# INSERTION SETTINGS ----------------------------------------------------------#


def _overall_insertion_rate(stage: Stage) -> float:
    """
    The overall insertion rate for a stage, used as the starting point for each base's rate. The user can change
    each base's rate, but this is the default.

    Args:
        stage: The stage to get the overall insertion rate for.

    Returns:
        The overall insertion rate for the stage.
    """
    overall_rate: float = rate_input(
        label="Overall insertion rate:",
        key=_stage_key(stage, "overall_insertion_rate"),
        default=STAGE_DEFAULTS[stage].insertion_rate,
        help="The overall rate at which insertions occur - overriden by per base rates below.",
    )

    return overall_rate


def _conditional_insertion_rate(stage: Stage) -> tuple[dict[Base, float], float]:
    """
    Which base gets inserted when an insertion happens. For example, a 0.4 chance of inserting A, 0.2 of C, 0.2 of G,
    and 0.2 of T. Also the chance the inserted base just repeats the base before it (stuttering), which overrides
    these chances when it happens.

    Args:
        stage: The stage to show the settings for.

    Returns:
        A tuple containing:
            - The chance of each base being inserted, keyed by the base.
            - The chance the inserted base repeats the base before it.
    """
    # One slider split into a segment for every base, starting at the defaults. Only one, so it gets the whole width
    chances: list[float] = split_slider(
        label="Inserted base:",
        key=_stage_key(stage, "insertion"),
        segment_labels=[str(base) for base in ErrorConfigs.DNA_BASES],
        default_chances=[STAGE_DEFAULTS[stage].insertion_conditional[base] for base in ErrorConfigs.DNA_BASES],
    )

    # Chances come back in the same order as the bases
    conditional: dict[Base, float] = {}
    for base, chance in zip(ErrorConfigs.DNA_BASES, chances):
        conditional[base] = chance

    # Stuttering - the inserted base is just a copy of the one before it, very common in homopolymers on nanopore
    repeat_previous: float = st.number_input(
        label="Repeat previous base chance:",
        min_value=ViewHelpers.PROBABILITY_RANGE[0],
        max_value=ViewHelpers.PROBABILITY_RANGE[1],
        step=0.01,
        value=STAGE_DEFAULTS[stage].insertion_repeat_previous,
        format=ViewHelpers.RATE_FORMAT,
        key=_stage_key(stage, "insertion_repeat_previous"),
        help="Chance the inserted base is just a copy of the base before it (stuttering). Overrides the inserted base"
             " chances above when it happens.",
        persist_state="session",
    )

    return conditional, repeat_previous


def _show_insertion_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced insertion settings - a rate after each base (starting at the overall rate), which base gets inserted, and
    how long each insertion is.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # OVERALL RATE - get overall rate first, so it can be used as starting point for each base's rate
    overall_rate: float = _overall_insertion_rate(stage)

    # PER BASE RATES - overides overall rate, so each base can have its own rate
    per_base: dict[Base, float] = per_base_rate(
        name="insertion",
        key=_stage_key(stage, "insertion_per_base"),
        stage_rate=STAGE_DEFAULTS[stage].insertion_rate_after_base,
        default=overall_rate
    )

    # WHICH BASE GETS INSERTED - e.g. 0.4 chance of A, 0.2 of C, G, and T. Plus the chance it just repeats the last base
    conditional: dict[Base, float]
    repeat_previous: float
    conditional, repeat_previous = _conditional_insertion_rate(stage)

    # HOW LONG - min length, max length, and the chance it stops at the min
    min_length: int
    length_probability: float
    max_length: int | None
    min_length, length_probability, max_length = length_settings(
        name="insertion",
        key=_stage_key(stage, "insertion"),
        length_range=INSERTION_LENGTH_RANGE,
        min_default=STAGE_DEFAULTS[stage].insertion_min_length,
        probability_default=STAGE_DEFAULTS[stage].insertion_length_probability,
        max_default=STAGE_DEFAULTS[stage].insertion_max_length,
    )

    return {
        ErrorConfigFields.INSERTION_RATE: overall_rate,
        ErrorConfigFields.INSERTION_RATE_AFTER_BASE: per_base,

        ErrorConfigFields.INSERTION_CONDITIONAL: conditional,
        ErrorConfigFields.INSERTION_REPEAT_PREVIOUS: repeat_previous,

        ErrorConfigFields.INSERTION_MIN_LENGTH: min_length,
        ErrorConfigFields.INSERTION_LENGTH_PROBABILITY: length_probability,
        ErrorConfigFields.INSERTION_MAX_LENGTH: max_length,
    }


# DELETION SETTINGS ----------------------------------------------------------#


def _overall_deletion_rate(stage: Stage) -> float:
    """
    The overall deletion rate for a stage, used as the starting point for each base's rate. The user can change
    each base's rate, but this is the default.

    Args:
        stage: The stage to get the overall deletion rate for.

    Returns:
        The overall deletion rate for the stage.
    """
    overall_rate: float = rate_input(
        label="Overall deletion rate:",
        key=_stage_key(stage, "overall_deletion_rate"),
        default=STAGE_DEFAULTS[stage].deletion_rate,
        help="The overall rate at which deletions occur - overriden by per base rates below.",
    )

    return overall_rate


def _show_deletion_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced deletion settings - a rate for each base (starting at the overall rate), and how long each deletion is.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # OVERALL RATE - get overall rate first, so it can be used as starting point for each base's rate
    overall_rate: float = _overall_deletion_rate(stage)

    # PER BASE RATES - overides overall rate, so each base can have its own rate
    per_base: dict[Base, float] = per_base_rate(
        name="deletion",
        key=_stage_key(stage, "deletion_per_base"),
        stage_rate=STAGE_DEFAULTS[stage].deletion_rate_per_base,
        default=overall_rate
    )

    # HOW LONG - min length, optional max length, and the chance it stops at the min
    min_length: int
    length_probability: float
    max_length: int | None
    min_length, length_probability, max_length = length_settings(
        name="deletion",
        key=_stage_key(stage, "deletion"),
        length_range=DELETION_LENGTH_RANGE,
        min_default=STAGE_DEFAULTS[stage].deletion_min_length,
        probability_default=STAGE_DEFAULTS[stage].deletion_length_probability,
        max_default=STAGE_DEFAULTS[stage].deletion_max_length,     # None, so a deletion has no max by default
    )

    return {
        ErrorConfigFields.DELETION_RATE: overall_rate,

        ErrorConfigFields.DELETION_RATE_PER_BASE: per_base,

        ErrorConfigFields.DELETION_MIN_LENGTH: min_length,
        ErrorConfigFields.DELETION_LENGTH_PROBABILITY: length_probability,
        ErrorConfigFields.DELETION_MAX_LENGTH: max_length,
    }


# HOMOPOLYMER SETTINGS --------------------------------------------------------#


# What the user sees for each multiplier column. Member names match the multipliers in ErrorConfigFields, so look one
#   up with MultiplierLabels[field.name]
class MultiplierLabels(StrEnum):
    HOMOPOLYMER_SUBSTITUTION_MULTIPLIER = "Substitution"
    HOMOPOLYMER_INSERTION_MULTIPLIER = "Insertion"
    HOMOPOLYMER_DELETION_MULTIPLIER = "Deletion"


def _homopolymer_multiplier_settings(stage: Stage) -> dict[str, dict[int, float]]:
    """
    How much each error's rate is multiplied by inside a homopolymer, per run length. One row per run length the config
    lists, one column per error. Runs longer than the last row use the last row's multiplier.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The multiplier of each run length for every error, keyed by the config field they set.
    """
    DEFAULT: ErrorConfig = STAGE_DEFAULTS[stage]

    # Each error's default multipliers, written out so it is clear which config field each one comes from
    MULTIPLIER_DEFAULT: dict[ErrorConfigFields, dict[int, float]] = {
        ErrorConfigFields.HOMOPOLYMER_SUBSTITUTION_MULTIPLIER: DEFAULT.homopolymer_substitution_multiplier,
        ErrorConfigFields.HOMOPOLYMER_INSERTION_MULTIPLIER: DEFAULT.homopolymer_insertion_multiplier,
        ErrorConfigFields.HOMOPOLYMER_DELETION_MULTIPLIER: DEFAULT.homopolymer_deletion_multiplier,
    }

    # Every run length any error lists - the built-in config lists the same ones for all three, but an uploaded file
    #   might not. An error missing a run length gets a multiplier of 1 there, so nothing changes for it
    run_lengths: list[int] = []
    for field_multipliers in MULTIPLIER_DEFAULT.values():
        for run_length in field_multipliers:
            if run_length not in run_lengths:
                run_lengths.append(run_length)
    # Ensure they are in order
    run_lengths.sort()

    # Show what setting we are changing
    st.markdown(
        body="Error multipliers:",
        help="Each error's rate inside a run is multiplied by this. Runs longer than the last row use the last"
                " row's multiplier."
    )

    # Header row - the run length, then one column per error
    header_columns: list[Any] = st.columns(len(MULTIPLIER_DEFAULT) + 1)
    header_columns[0].markdown("Run length")
    for column, field in zip(header_columns[1:], MULTIPLIER_DEFAULT):
        column.markdown(MultiplierLabels[field.name])

    # One row per run length, with a multiplier for every error
    multipliers: dict[str, dict[int, float]] = {}
    for field in MULTIPLIER_DEFAULT:
        multipliers[field] = {}

    # For each run length, show a number input for every error's multiplier
    for run_length in run_lengths:
        row_columns: list[Any] = st.columns(len(MULTIPLIER_DEFAULT) + 1)
        row_columns[0].markdown(f"{run_length}")

        # Show all the defaults for eacj
        for column, field in zip(row_columns[1:], MULTIPLIER_DEFAULT):
            with column:
                multipliers[field][run_length] = st.number_input(
                    label=f"{MultiplierLabels[field.name]} multiplier for a run of {run_length}:",
                    min_value=0.0,
                    step=MULTIPLIER_STEP,
                    value=MULTIPLIER_DEFAULT[field].get(run_length, 1.0),
                    format=ViewHelpers.RATE_FORMAT,                 # No trailing 0's
                    key=_stage_key(stage, f"{field}_{run_length}"),
                    label_visibility="collapsed",                   # Header row already says what each column is
                    persist_state="session",
                )

    return multipliers


def _homopolymer_dropout_settings(stage: Stage) -> dict[str, Any]:
    """
    Long runs can lose the whole strand. Every base of a strand's longest run from the dropout length on adds the
    dropout rate as another chance of losing it, e.g. a run of 9 with a dropout length of 7 is 3 more chances.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The dropout settings picked, keyed by the config field they set.
    """
    # Show what setting we are changing
    st.markdown(
            body="Strand dropout:",
            help="Heckel et al. note runs longer than six raise dropout - synthesis fails on them, PCR stalls, or the"
                 " sequencer loses the cluster."
        )

    # Both fit on one row
    length_column: Any
    rate_column: Any
    length_column, rate_column = st.columns(2)

    # Run length the dropout starts at
    with length_column:
        dropout_length: int = st.number_input(
            label="Dropout length:",
            min_value=RUN_LENGTH_RANGE[0],
            max_value=RUN_LENGTH_RANGE[1],
            step=1,
            value=STAGE_DEFAULTS[stage].homopolymer_dropout_length,
            key=_stage_key(stage, "homopolymer_dropout_length"),
            help="Runs at least this long can lose the whole strand.",
            persist_state="session",
        )

    # Chance of losing the strand for every base past it
    with rate_column:
        dropout_rate: float = rate_input(
            label="Dropout rate per base:",
            key=_stage_key(stage, "homopolymer_dropout_rate"),
            default=STAGE_DEFAULTS[stage].homopolymer_dropout_rate,
            help="Chance of losing the strand for every base of its longest run from the dropout length on.",
        )

    return {
        ErrorConfigFields.HOMOPOLYMER_DROPOUT_LENGTH: dropout_length,
        ErrorConfigFields.HOMOPOLYMER_DROPOUT_RATE: dropout_rate,
    }


def _show_homopolymer_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced homopolymer settings - whether runs are looked at, the run length they start at, how much each error is
    multiplied by in a run, and how long runs can lose the whole strand. Shows nothing else when turned off.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # Toggle for homopolymers, on by default (same as the config)
    homopolymer_check: bool = st.toggle(
        label="Check homopolymers",
        value=STAGE_DEFAULTS[stage].homopolymer_check,
        key=_stage_key(stage, "homopolymer_check"),
        help="Scale errors up inside runs of the same base, and let long runs lose the whole strand.",
        persist_state="session",
    )

    # Turned off - nothing to show, the simulator skips homopolymers
    if not homopolymer_check:
        return {ErrorConfigFields.HOMOPOLYMER_CHECK: False}

    # MIN LENGTH - runs shorter than this are treated as normal bases
    min_length: int = st.number_input(
        label="Min run length:",
        min_value=RUN_LENGTH_RANGE[0],
        max_value=RUN_LENGTH_RANGE[1],
        step=1,
        value=STAGE_DEFAULTS[stage].homopolymer_min_length,
        key=_stage_key(stage, "homopolymer_min_length"),
        help="Once a run reaches this length, every base from there on has its error rates multiplied.",
        persist_state="session",
    )

    # MULTIPLIERS - how much each error goes up in a run of each length
    settings: dict[str, Any] = _homopolymer_multiplier_settings(stage)

    # DROPOUT - long runs can lose the whole strand
    settings.update(_homopolymer_dropout_settings(stage))

    settings.update({
        ErrorConfigFields.HOMOPOLYMER_CHECK: homopolymer_check,
        ErrorConfigFields.HOMOPOLYMER_MIN_LENGTH: min_length,
    })
    return settings


# GC SETTINGS -----------------------------------------------------------------#


def _gc_window_settings(stage: Stage) -> dict[str, Any]:
    """
    The GC window - how big it is, the safe zone of GC content, and how fast errors go up outside of it. For each base
    we look at the window around it, inside the safe zone the multiplier is 1, outside it grows linearly by the slope.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The window settings picked, keyed by the GCWindow field they set. Left as a dict so pydantic builds the window
        along with the rest of the config, and any error is shown the same way.
    """
    defaults: GCWindow = STAGE_DEFAULTS[stage].gc_window

    # All three fit on one row, aligned at the bottom so the slider lines up with the number inputs
    bounds_column: Any
    size_column: Any
    slope_column: Any
    bounds_column, size_column, slope_column = st.columns(3, vertical_alignment="bottom")

    # SAFE ZONE - a range slider, so the lower bound can never be past the upper bound
    with bounds_column:
        lower_bound: float
        upper_bound: float
        lower_bound, upper_bound = st.slider(
            label="Safe GC content:",
            min_value=ViewHelpers.PROBABILITY_RANGE[0],
            max_value=ViewHelpers.PROBABILITY_RANGE[1],
            value=(defaults.lower_bound, defaults.upper_bound),     # Bound inbetween a box (usually its .4 and .6)
            step=0.01,
            key=_stage_key(stage, "gc_bounds"),
            help="Fraction of G and C in the window that is considered safe. Between 0.4 and 0.6 is the safe zone in"
                 " most papers, outside it strands are harder to work with.",
            persist_state="session",
        )

    # How many bases around each base are looked at
    with size_column:
        window_size: int = st.number_input(
            label="Window size:",
            min_value=WINDOW_SIZE_RANGE[0],
            max_value=WINDOW_SIZE_RANGE[1],
            step=1,
            value=defaults.window_size,
            key=_stage_key(stage, "gc_window_size"),
            help="How many bases around each base are looked at, cut short at the ends of the strand.",
            persist_state="session",
        )

    # How fast the multiplier grows outside the safe zone. Not a rate, it can go past 1 (default is 10), so no rate_input
    with slope_column:
        slope: float = st.number_input(
            label="Slope:",
            min_value=0.0,                      # Config only needs it to not be negative
            step=1.0,
            value=defaults.slope,
            format=ViewHelpers.RATE_FORMAT,     # No trailing 0's
            key=_stage_key(stage, "gc_slope"),
            help="How fast errors go up outside the safe zone. The multiplier is 1 + slope times how far outside it"
                 " is, e.g. a slope of 10 and 0.7 GC (0.1 past 0.6) doubles the errors.",
            persist_state="session",
        )

    return {
        GCWindowFields.WINDOW_SIZE: window_size,
        GCWindowFields.LOWER_BOUND: lower_bound,
        GCWindowFields.UPPER_BOUND: upper_bound,
        GCWindowFields.SLOPE: slope,
    }


def _show_gc_content_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced GC content settings - whether GC content is looked at, and the window used to look at it. Shows nothing
    else when turned off.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # Toggle for GC content, on by default (same as the config)
    gc_content_check: bool = st.toggle(
        label="Check GC content",
        value=STAGE_DEFAULTS[stage].gc_content_check,
        key=_stage_key(stage, "gc_content_check"),
        help="Scale errors up where the GC content around a base is outside the safe zone.",
        persist_state="session",
    )

    # Turned off - nothing to show, the simulator skips GC content
    if not gc_content_check:
        return {ErrorConfigFields.GC_CONTENT_CHECK: False}

    # WINDOW - the safe zone, how big the window is, and how fast errors go up outside it
    gc_window: dict[str, Any] = _gc_window_settings(stage)

    return {
        ErrorConfigFields.GC_CONTENT_CHECK: gc_content_check,
        ErrorConfigFields.GC_WINDOW: gc_window,
    }


# POSITION BIAS SETTINGS ------------------------------------------------------#


def _show_position_bias_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced position bias settings - errors typically go up toward the end of the strand. The multiplier goes linearly
    from 1 at the start of the strand to 1 + slope at the end. Only one setting, a slope of 0 turns it off.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # How much more likely an error is at the end of the strand than at the start
    position_slope: float = rate_input(
        label="Position slope:",
        key=_stage_key(stage, "position_slope"),
        default=STAGE_DEFAULTS[stage].position_slope,
        help="Error rates go from 1x at the start of the strand to 1 + slope at the end, e.g. a slope of 0.05 means"
             " errors are 5% more likely at the end. A slope of 0 turns position bias off.",
    )

    return {
        ErrorConfigFields.POSITION_SLOPE: position_slope
    }


# STRAND LOSS SETTINGS --------------------------------------------------------#


def _show_strand_loss_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced strand loss settings - truncation (the strand is cut and everything past it is gone) and dropout (the
    whole strand is lost). Truncation is per base so longer strands are cut more, dropout is per strand so length
    doesn't matter.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    # Both fit on one row
    truncation_column: Any
    dropout_column: Any
    truncation_column, dropout_column = st.columns(2)

    # Per base - a strand stays whole with (1 - rate)^length
    with truncation_column:
        truncation_rate: float = rate_input(
            label="Truncation rate:",
            key=_stage_key(stage, "truncation_rate"),
            default=STAGE_DEFAULTS[stage].truncation_rate,
            help="Chance the strand is cut at each base, everything past the cut is gone. Per base, so longer strands"
                 " are more likely to be cut.",
        )

    # Per strand - stuck to a tube, not amplified, never read, etc.
    with dropout_column:
        dropout_rate: float = rate_input(
            label="Dropout rate:",
            key=_stage_key(stage, "advanced_dropout_rate"),
            default=STAGE_DEFAULTS[stage].dropout_rate,
            help="Chance a whole strand is lost, no matter how long it is.",
        )

    return {
        ErrorConfigFields.TRUNCATION_RATE: truncation_rate,
        ErrorConfigFields.DROPOUT_RATE: dropout_rate,
    }


# STAGE SPECIFIC SETTINGS -----------------------------------------------------#


def _show_synthesis_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced synthesis settings - how many copies are made of each strand, and how they are handed out.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    synthesis_defaults: SynthesisConfig = cast(SynthesisConfig, STAGE_DEFAULTS[stage])

    # Depth, distribution, and dispersion - same as sequencing, so shared
    depth: int
    distribution: CoverageDistribution
    dispersion: float
    depth, distribution, dispersion = coverage_settings(
        key=_stage_key(stage, "synthesis_coverage"),
        depth_label="Synthesis depth:",
        depth_help="Mean number of copies made of each strand.",
        depth_range=DEPTH_RANGE,
        depth_default=synthesis_defaults.synthesis_depth,
        distribution_default=synthesis_defaults.copy_distribution,
        dispersion_default=synthesis_defaults.copy_dispersion,
    )

    return {
        SynthesisFields.SYNTHESIS_DEPTH: depth,
        SynthesisFields.COPY_DISTRIBUTION: distribution,
        SynthesisFields.COPY_DISPERSION: dispersion,
    }


def _show_aging_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced aging settings - how long the strands are stored, their half life, and how fast bases deaminate.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    aging_defaults: AgingConfig = cast(AgingConfig, STAGE_DEFAULTS[stage])

    # All three fit on one row, aligned at the bottom so they line up
    years_column: Any
    half_life_column: Any
    deamination_column: Any
    years_column, half_life_column, deamination_column = st.columns(3, vertical_alignment="bottom")

    # How long they sit in storage
    with years_column:
        years: float = st.number_input(
            label="Years stored:",
            min_value=YEARS_RANGE[0],
            max_value=YEARS_RANGE[1],
            step=10.0,
            value=aging_defaults.years,
            format=ViewHelpers.RATE_FORMAT,                 # No trailing 0's
            key=_stage_key(stage, "advanced_years"),
            help="How long the strands are stored. Every rate is per year, so 0 years means no aging at all.",
            persist_state="session",
        )

    # Each strand survives with 0.5^(years / half life)
    with half_life_column:
        half_life_years: float = st.number_input(
            label="Half life (years):",
            min_value=1.0,                      # Has to be above 0
            step=50.0,
            value=aging_defaults.half_life_years,
            format=ViewHelpers.RATE_FORMAT,
            key=_stage_key(stage, "half_life_years"),
            help="Half of the strands are lost every half life. ~500 years is cited for DNA in bone (Heckel et al.),"
                 " encapsulated DNA lasts far longer.",
            persist_state="session",
        )

    # Extra substitutions per year
    with deamination_column:
        deamination_rate: float = rate_input(
            label="Deamination rate:",
            key=_stage_key(stage, "deamination_rate"),
            default=aging_defaults.deamination_rate,
            help="Chance a base is deaminated each year, basically an extra substitution rate that builds up over the"
                 " years.",
        )

    return {
        AgingFields.YEARS: years,
        AgingFields.HALF_LIFE_YEARS: half_life_years,
        AgingFields.DEAMINATION_RATE: deamination_rate,
    }


def _show_pcr_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced PCR settings - how many cycles, how well each strand copies, how much that varies, and how much GC content
    outside the safe zone slows it down.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    pcr_defaults: PCRConfig = cast(PCRConfig, STAGE_DEFAULTS[stage])

    # First row - how many cycles and how well strands copy
    cycles_column: Any
    efficiency_column: Any
    cycles_column, efficiency_column = st.columns(2, vertical_alignment="bottom")

    # How many times every strand is copied
    with cycles_column:
        cycles: int = st.number_input(
            label="Cycles:",
            min_value=CYCLES_RANGE[0],
            max_value=CYCLES_RANGE[1],
            step=1,
            value=pcr_defaults.cycles,
            key=_stage_key(stage, "advanced_cycles"),
            help="How many times every strand is copied. Each cycle can double the number of strands, so this gets"
                 " slow fast.",
            persist_state="session",
        )

    # Fraction copied each cycle
    with efficiency_column:
        efficiency: float = rate_input(
            label="Efficiency:",
            key=_stage_key(stage, "efficiency"),
            default=pcr_defaults.efficiency,
            help="Fraction of strands copied each cycle, 1 means every strand doubles.",
        )

    # Second row - how much the efficiency varies, and how GC content changes it
    strand_std_column: Any
    cycle_std_column: Any
    gc_slope_column: Any
    strand_std_column, cycle_std_column, gc_slope_column = st.columns(3, vertical_alignment="bottom")

    # Across strands - this is what turns into bias over the cycles
    with strand_std_column:
        efficiency_std: float = st.number_input(
            label="Efficiency std (strands):",
            min_value=0.0,
            step=0.01,
            value=pcr_defaults.efficiency_std,
            format=ViewHelpers.RATE_FORMAT,
            key=_stage_key(stage, "efficiency_std"),
            help="How much the efficiency differs between strands. Small differences turn into a big difference in"
                 " copies over many cycles.",
            persist_state="session",
        )

    # Across cycles
    with cycle_std_column:
        efficiency_cycle_std: float = st.number_input(
            label="Efficiency std (cycles):",
            min_value=0.0,
            step=0.01,
            value=pcr_defaults.efficiency_cycle_std,
            format=ViewHelpers.RATE_FORMAT,
            key=_stage_key(stage, "efficiency_cycle_std"),
            help="How much the efficiency differs between cycles.",
            persist_state="session",
        )

    # Strands outside the safe GC zone copy worse
    with gc_slope_column:
        gc_efficiency_slope: float = st.number_input(
            label="GC efficiency slope:",
            min_value=0.0,
            step=0.1,
            value=pcr_defaults.gc_efficiency_slope,
            format=ViewHelpers.RATE_FORMAT,
            key=_stage_key(stage, "gc_efficiency_slope"),
            help="How much the efficiency drops per unit of the strand's GC content outside the safe zone. Only used"
                 " when GC content is checked.",
            persist_state="session",
        )

    return {
        PCRFields.CYCLES: cycles,
        PCRFields.EFFICIENCY: efficiency,
        PCRFields.EFFICIENCY_STD: efficiency_std,
        PCRFields.EFFICIENCY_CYCLE_STD: efficiency_cycle_std,
        PCRFields.GC_EFFICIENCY_SLOPE: gc_efficiency_slope,
    }


def _show_sequencing_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced sequencing settings - how many reads each strand gets, and how they are handed out.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    sequencing_defaults: SequencingConfig = cast(SequencingConfig, STAGE_DEFAULTS[stage])

    # Depth, distribution, and dispersion - same as synthesis, so shared
    depth: int
    distribution: CoverageDistribution
    dispersion: float
    depth, distribution, dispersion = coverage_settings(
        key=_stage_key(stage, "sequencing_coverage"),
        depth_label="Read depth:",
        depth_help="Mean number of reads each original strand gets.",
        depth_range=DEPTH_RANGE,
        depth_default=sequencing_defaults.read_depth,
        distribution_default=sequencing_defaults.read_distribution,
        dispersion_default=sequencing_defaults.read_dispersion,
    )

    return {
        SequencingFields.READ_DEPTH: depth,
        SequencingFields.READ_DISTRIBUTION: distribution,
        SequencingFields.READ_DISPERSION: dispersion,
    }


def _show_stage_specific_settings(stage: Stage) -> dict[str, Any]:
    """
    Advanced stage specific settings - each stage has its own settings that don't fit into the other categories. Shows
    nothing when the stage has no specific settings.

    Args:
        stage: The stage to show the settings for.

    Returns:
        The settings picked, keyed by the config field they set.
    """
    match stage:
        case Stage.SYNTHESIS:
            return _show_synthesis_settings(stage)
        case Stage.AGING:
            return _show_aging_settings(stage)
        case Stage.PCR:
            return _show_pcr_settings(stage)
        case Stage.SEQUENCING:
            return _show_sequencing_settings(stage)
        case _:
            return {}


def _show_advanced_settings(stage: Stage) -> dict[str, Any]:
    # Overall rates first, same as simple mode - the sections below build on them with uppdate
    settings: dict[str, Any] = {}

    # Tabs instead of a segmented control - every tab runs on every rerun (only the picked one is shown), so a section
    #   not being looked at still gives back its settings. With a segmented control they would fall back to the defaults
    tabs_ui_elements: Any = st.tabs(
        tabs=ADVANCED_SETTINGS_LABELS,
        key=_stage_key(stage, "advanced_section"),
    )
    # Actual want as a dict so we can use the labels to pick the right tab (makes it easier to view)
    tabs: dict[AdvancedSettingNames, Any] = {}
    for tab, label in zip(tabs_ui_elements, ADVANCED_SETTINGS_LABELS):
        tabs[label] = tab

    # For each tab tally up our settings
    with tabs[AdvancedSettingNames.SUBSTITUTION]:
        settings.update(_show_substitution_settings(stage))
    with tabs[AdvancedSettingNames.INSERTION]:
        settings.update(_show_insertion_settings(stage))
    with tabs[AdvancedSettingNames.DELETION]:
        settings.update(_show_deletion_settings(stage))
    with tabs[AdvancedSettingNames.HOMOPOLYMERS]:
        settings.update(_show_homopolymer_settings(stage))
    with tabs[AdvancedSettingNames.GC_CONTENT]:
        settings.update(_show_gc_content_settings(stage))
    with tabs[AdvancedSettingNames.POSITION_BIAS]:
        settings.update(_show_position_bias_settings(stage))
    with tabs[AdvancedSettingNames.STRAND_LOSS]:
        settings.update(_show_strand_loss_settings(stage))
    with tabs[AdvancedSettingNames.STAGE_SPECIFIC]:
        settings.update(_show_stage_specific_settings(stage))

    return settings



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
        label=f"**{StageNames[stage.name]}**",      # Make it bold so its more clear
        value=STAGE_DEFAULTS[stage].enabled,    # Use config default as default for the toggle (will be on)
        key=_stage_key(stage, StageKeys.ENABLED),
        help=f"Turn the {StageNames[stage.name].lower()} stage on or off.",
        persist_state="session",                # Keep value for whole session
    )

    # Stage is off - nothing to show, and the simulator just skips it
    if not enabled:
        return STAGE_CONFIGS[stage](enabled=False)

    # Add our little description, so the user knows what it does
    st.caption(STAGE_DESCRIPTIONS[stage])

    # Mode and seed on one row - both are used in either mode, so they go above the mode's own settings
    mode_column: Any
    seed_column: Any
    mode_column, seed_column = st.columns(2, vertical_alignment="bottom")

    # Simple or advanced, same way the input method is picked
    with mode_column:
        mode: str = st.segmented_control(
            label="Mode:",
            options=[ModeNames.SIMPLE, ModeNames.ADVANCED],
            default=ModeNames.ADVANCED if STAGE_DEFAULTS[stage].advanced_mode else ModeNames.SIMPLE,
            required=True,                          # Must be either one
            key=_stage_key(stage, StageKeys.MODE),
            persist_state="session",
        )

    # Seed for this stage's random numbers. Left empty it is None, so the stage gets different random numbers every run
    with seed_column:
        seed: int | None = st.number_input(
            label="Seed (optional):",
            min_value=0,                            # numpy only takes seeds that aren't negative
            step=1,
            value=STAGE_DEFAULTS[stage].seed,       # None unless uploaded settings have one
            placeholder="Random",
            key=_stage_key(stage, StageKeys.SEED),
            help="The same seed with the same strands and settings gives the same errors every run. Leave it empty"
                 " for different errors each run. Every stage has its own seed.",
            persist_state="session",
        )

    # Every mode gives back the config fields it sets, so the config is built the same way for both
    settings: dict[str, Any] | None
    match mode:
        case ModeNames.SIMPLE:
            settings = _show_simple_settings(stage)
        case ModeNames.ADVANCED:
            settings = _show_advanced_settings(stage)
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
            seed=seed,
            **settings,             # Need to unpack as stages have specific settings that we got before
        )
    except ValidationError as error:
        st.error(f"A {StageNames[stage.name].lower()} setting is not valid: {error.errors()[0]['msg']}")
        return None



#------------------------------------------------------------------------------#
#                                                                              #
#   SETTINGS FILE                                                              #
#                                                                              #
#------------------------------------------------------------------------------#



# What we call the save file
SIMULATION_SETTINGS: Final[str] = "simulation-settings.json"


def _is_simulation_settings(json_path: Path) -> bool:
    """
    Check the uploaded JSON is a whole simulation's settings, saved by the download button below. Runs once in the
    uploader's callback, not on every rerun.

    Args:
        json_path: The path to our copy of the uploaded JSON.

    Returns:
        True if the file loads as a simulation's settings, False if not.
    """
    # Only need to know it loads, it is loaded again when the settings are filled in
    try:
        ErrorConfigSerializer.load_simulation(str(json_path))
    except (ValidationError, FileNotFoundError):
        return False

    return True


def _load_settings(settings_path: Path) -> None:
    """
    Fills in every stage's settings from an uploaded settings file. The file's configs become every widget's defaults
    (see STAGE_DEFAULTS), and every stage widget is forgotten so it is drawn again from them. Must be called before any
    stage widget is drawn, and followed by a rerun so STAGE_DEFAULTS picks the new configs up.

    Args:
        settings_path: The path to our copy of the uploaded settings.
    """
    saved: SavedSimulation = ErrorConfigSerializer.load_simulation(str(settings_path))

    # The uploaded configs are the defaults now
    st.session_state[SimulatorInputs.LOADED_SETTINGS] = {
        Stage.SYNTHESIS: saved.synthesis,
        Stage.AGING: saved.aging,
        Stage.PCR: saved.pcr,
        Stage.SEQUENCING: saved.sequencing,
    }

    # Forget every stage widget (and the split sliders' saved cuts), otherwise they would keep what the user had before
    #   instead of starting from the uploaded values. Every one of them is keyed with _stage_key, so they share a 
    #   start. Prevents keys from piling up every time an upload is done (memory could skyrocket)
    for stage in STAGES:
        stage_prefix: str = _stage_key(stage, "")
        for key in list(st.session_state.keys()):
            if str(key).startswith(stage_prefix):
                st.session_state.pop(key)

    # New keys for every stage widget, so the browser draws them again too instead of keeping its old values - basically
    #   a force update
    st.session_state[SimulatorInputs.SETTINGS_VERSION] = st.session_state.get(SimulatorInputs.SETTINGS_VERSION, 0) + 1

    # Remember which file these came from, so they are only filled in once
    st.session_state[SimulatorInputs.SETTINGS_FROM] = settings_path


def show_settings_upload() -> None:
    """
    Upload a settings file saved by the download button, and fill in every stage's settings from it. Only filled in
    once per file, so anything the user changes after is kept. Must be shown before any stage's settings.
    """
    settings_path: Path | None = file_uploader(
        label="Upload simulation settings (optional):",
        keys=SETTINGS_UPLOAD_KEYS,
        error_message="is not a valid simulation settings file. Please upload a JSON file saved from this page.",
        file_type="json",                   # Only the JSON files saved by the download button
        is_valid=_is_simulation_settings,   # Only valid if it loads as a simulation's settings
    )

    # Nothing uploaded, nothing to fill in
    if settings_path is None:
        return

    # New file - fill in every stage from it, then rerun so every widget starts from the uploaded values
    if st.session_state.get(SimulatorInputs.SETTINGS_FROM, None) != settings_path:
        _load_settings(settings_path)
        
        # Reruns so we get the new filled in settings (force reloads page serverside)
        st.rerun()

    # Let the user know where the settings came from, while that file is still uploaded
    st.caption(f"Settings filled in from {st.session_state[SimulatorInputs.SETTINGS_FILE_NAME]}.")


def show_settings_download(configs: dict[Stage, ErrorConfig]) -> None:
    """
    Download every stage's settings as one JSON file, so they can be uploaded again later. Only enabled when every
    stage's settings are valid.

    Args:
        configs: The config of every stage with valid settings.
    """
    ready: bool = len(configs) == len(STAGES)

    # Easier just to keep the callback here to keep it contained
    def settings_json() -> bytes:
        """
        Writes the settings to a temporary file and reads it back. Only runs when the button is pressed, so a file
        isn't written on every rerun.

        Returns:
            The settings file's contents.
        """
        json_path: Path = Path(mkdtemp(prefix=ViewHelpers.TEMPORARY_FILE_PREFIX)) / SIMULATION_SETTINGS

        # Each config is already the right stage (built from STAGE_CONFIGS), the dict just can't tell the type checker
        ErrorConfigSerializer.save_simulation(
            synthesis=cast(SynthesisConfig, configs[Stage.SYNTHESIS]),
            aging=cast(AgingConfig, configs[Stage.AGING]),
            pcr=cast(PCRConfig, configs[Stage.PCR]),
            sequencing=cast(SequencingConfig, configs[Stage.SEQUENCING]),
            json_path=str(json_path),
        )

        return json_path.read_bytes()

    # Show a download button in case user wants to save for later
    st.download_button(
        label="Download settings (.json)",
        help="Every stage's settings, can be uploaded above to fill them all in again." if ready else
             "Fix the stage settings above to download them.",      # Dont allow to download bad files
        data=settings_json,                                         # Only made when pressed
        file_name=SIMULATION_SETTINGS,
        mime="application/json",
        icon=":material/download:",
        disabled=not ready,
    )



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

        # Both downloads go in the same temporary folder - only get the name not the whole path
        out_directory: Path = Path(mkdtemp(prefix=ViewHelpers.TEMPORARY_FILE_PREFIX))
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

# What the user sees for each error type. Member names match ErrorType's, so look one up with
#   ErrorTypeNames[error_type.name]
class ErrorTypeNames(StrEnum):
    SUBSTITUTION = "Substitutions"
    INSERTION = "Insertions"
    DELETION = "Deletions"
    DROPOUT = "Dropouts"
    TRUNCATION = "Truncations"


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
            label=ErrorTypeNames[error_type.name],
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
    stage_counts.index = [StageNames[stage.name] for stage in STAGES]
    stage_counts.columns = [ErrorTypeNames[error_type.name] for error_type in ERROR_TYPES]

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
        table[ErrorDictKeys.STAGE] = [StageNames[stage.name] for stage in table[ErrorDictKeys.STAGE]]
        table[ErrorDictKeys.ERROR_TYPE] = [ErrorTypeNames[error.name] for error in table[ErrorDictKeys.ERROR_TYPE]]

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

# Settings saved from this page can fill in every stage at once - has to come before the stages are drawn
show_settings_upload()

# Every stage gets the whole width, one after another in the order they run. None if a stage's settings aren't valid
stage_configs: dict[Stage, ErrorConfig | None] = {}
for stage in STAGES:
    with st.container(border=True):
        stage_configs[stage] = show_stage_settings(stage)

# Only the stages with valid settings - if any are missing the simulation can't run (or be downloaded)
valid_configs: dict[Stage, ErrorConfig] = {
    stage: config for stage, config in stage_configs.items() if config is not None
}

# Save every stage's settings, so they can be uploaded again later
show_settings_download(valid_configs)


# Simulate section, only enabled if there is an input and every stage's settings are valid
st.subheader("Simulate", divider="gray")

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