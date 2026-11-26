"""
Module for the Clustering and Consensus page, which groups noisy reads by the strand they came from and finds the best
guess of each strand. The input is either a JSON file saved from the Encoding page (or the oligos encoded earlier in the
session), or a text file of oligos one per line. A JSON input gets the consensus strands added to its oligos, so it can
be decoded straight after.
"""


from enum import StrEnum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Final

import streamlit as st
from pydantic import ValidationError

from Model.Convergence.Clustering import Clustering, ClusteringConfig
from Model.Convergence.Consensus import Alignment, Consensus, ConsensusConfig
from Model.Encode.Methods.Encoding import Encoding
from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer
from View.Helpers import ViewHelpers
from View.Helpers.ViewHelpers import UploadKeys, clean_primer, slider_with_input, show_input_settings, show_downloads


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class ConvergenceInputs(StrEnum):
    # Need to make these keys as Streamlit doesn't keep the text_input box written, when user switches. Just easier.
    JSON_UPLOAD = "JSON Upload"
    TEXT_UPLOAD = "Raw Oligo Upload"

    # Keys for the JSON uploader, so we can store the uploaded file in session state and use it across pages
    JSON_UPLOAD_WIDGET = "convergence_json_upload"
    JSON_FILE_NAME = "convergence_json_file_name"
    JSON_FILE_PATH = "convergence_json_file_path"
    JSON_UPLOAD_ERROR = "convergence_json_upload_error"

    # Text uploader gets its own keys - otherwise an uploaded JSON would still be used after switching to text
    TEXT_UPLOAD_WIDGET = "convergence_text_upload"
    TEXT_FILE_NAME = "convergence_text_file_name"
    TEXT_FILE_PATH = "convergence_text_file_path"
    TEXT_UPLOAD_ERROR = "convergence_text_upload_error"


# Every key each file uploader keeps its upload under
JSON_UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=ConvergenceInputs.JSON_UPLOAD_WIDGET,
    name=ConvergenceInputs.JSON_FILE_NAME,
    path=ConvergenceInputs.JSON_FILE_PATH,
    error=ConvergenceInputs.JSON_UPLOAD_ERROR,
)
TEXT_UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=ConvergenceInputs.TEXT_UPLOAD_WIDGET,
    name=ConvergenceInputs.TEXT_FILE_NAME,
    path=ConvergenceInputs.TEXT_FILE_PATH,
    error=ConvergenceInputs.TEXT_UPLOAD_ERROR,
)



#------------------------------------------------------------------------------#
#                                                                              #
#   CLUSTERING SETTINGS                                                        #
#                                                                              #
#------------------------------------------------------------------------------#



# Defaults come straight from the config, so the page always matches what the model uses
CLUSTERING_DEFAULTS: Final[ClusteringConfig] = ClusteringConfig()

# Ranges for the integer settings. Seeds past 32 barely ever survive a few errors, and the rest just need a sane cap
PROBABILITY_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
SEED_LENGTH_RANGE: Final[tuple[int, int]] = (4, 32)
MIN_SEED_HITS_RANGE: Final[tuple[int, int]] = (1, 20)
MAX_REJECTS_RANGE: Final[tuple[int, int]] = (1, 1000)


# Session state keys for the primer boxes, so they can be filled in from the input JSON
class ClusteringKeys(StrEnum):
    START_PRIMER = "convergence_start_primer"
    END_PRIMER = "convergence_end_primer"

    # The JSON the primers were last filled in from, so they are only filled in once per input
    PRIMERS_FROM = "convergence_primers_from"


def _set_default_primers(in_path: Path | None, is_json: bool) -> None:
    """
    Fills the primer boxes in with the input JSON's primers, once each time the input JSON changes - so anything the
    user types in is kept until they pick a different input. Text files have no primers saved, so the boxes are left
    as they are. Must be called before the primer boxes are drawn, Streamlit won't let us change them after.

    Args:
        in_path: The path to the input, or None if there is no valid input yet.
        is_json: Whether the input is a JSON file saved from the Encoding page.
    """
    # Nothing to fill in from, or already filled in from this one
    if in_path is None or not is_json or st.session_state.get(ClusteringKeys.PRIMERS_FROM, None) == in_path:
        return

    # Checked on upload, but the encoded JSON is not 
    try:
        start_primer, end_primer = Serializer.load_primers(str(in_path))
    # If it can't be read just leave the to config defaults (would literally just be empty for now)
    except (ValidationError, FileNotFoundError):
        st.session_state.setdefault(ClusteringKeys.START_PRIMER, CLUSTERING_DEFAULTS.start_primer)
        st.session_state.setdefault(ClusteringKeys.END_PRIMER, CLUSTERING_DEFAULTS.end_primer)
        return

    # And update our session so it stays filled
    st.session_state[ClusteringKeys.START_PRIMER] = start_primer
    st.session_state[ClusteringKeys.END_PRIMER] = end_primer
    st.session_state[ClusteringKeys.PRIMERS_FROM] = in_path


def _show_primer_settings(in_path: Path | None, is_json: bool) -> tuple[str | None, str | None, float]:
    """
    Settings for the primers trimmed off each read before clustering - the primers themselves, and how much of each
    can be wrong and still be found. The primers are filled in from the input JSON if it has them.

    Args:
        in_path: The path to the input, used to fill in the primers if it is a JSON. None if there is no input yet.
        is_json: Whether the input is a JSON file saved from the Encoding page.

    Returns:
        The start primer, end primer (None if a primer is not valid), and the primer error fraction.
    """
    # Primers come from the input JSON if it has them, otherwise blank since text reads could come from anywhere. Set
    #   the starting value through session state instead of value=, Streamlit warns if a widget gets both
    _set_default_primers(in_path, is_json)

    # Primers as text, cleaned up after so lowercase or spaces don't count as wrong
    start_primer_text: str = st.text_input(
        label="Start primer:",
        key=ClusteringKeys.START_PRIMER,
        help="Primer on the start of every read, trimmed off before clustering so it doesn't make every strand look"
             " alike. Filled in from the input JSON if it has one. Optional - leave empty to not trim anything.",
        persist_state="session",            # Keep value for whole session
    )
    end_primer_text: str = st.text_input(
        label="End primer:",
        key=ClusteringKeys.END_PRIMER,
        help="Primer on the end of every read, trimmed off before clustering. Filled in from the input JSON if it has"
             " one. Optional - leave empty to not trim anything.",
        persist_state="session",
    )

    # Let the user know where the primers came from, while that JSON is still the input
    if is_json and in_path is not None and st.session_state.get(ClusteringKeys.PRIMERS_FROM, None) == in_path:
        st.caption(f"Primers filled in from {Path(in_path).name}.")

    start_primer: str | None = clean_primer(start_primer_text)
    end_primer: str | None = clean_primer(end_primer_text)

    # Let the user know right away if a primer is wrong, instead of when they press the button
    if start_primer is None:
        st.error(f"Start primer must be in set of {Encoding.DNA_BASES}")
    if end_primer is None:
        st.error(f"End primer must be in set of {Encoding.DNA_BASES}")

    # Slider for the primer error fraction, a probability so between 0 and 1. Only matters if there is a primer to
    #   find, so grey it out otherwise
    primer_error_fraction: float = slider_with_input(
        label="Primer error fraction:",
        key="convergence_primer_error_fraction",
        min_value=PROBABILITY_RANGE[0],
        max_value=PROBABILITY_RANGE[1],
        default=CLUSTERING_DEFAULTS.primer_error_fraction,
        step=0.01,
        help="How much of a primer can be wrong and still be found. If it can't be found, its length is cut off"
             " instead. Only used if there is a primer.",
        disabled=not start_primer_text.strip() and not end_primer_text.strip(),
    )

    return start_primer, end_primer, primer_error_fraction


def _show_vsearch_settings() -> tuple[float, int, int, int]:
    """
    Settings for how each read finds its cluster, the seed-then-align search from VSEARCH - how similar a read has to
    be to join, which centroids are worth aligning against, and when to give up.

    Returns:
        The identity threshold, seed length, minimum seed hits, and maximum rejects as a tuple.
    """
    # Slider for the identity threshold, also a probability
    identity: float = slider_with_input(
        label="Identity threshold:",
        key="convergence_identity",
        min_value=PROBABILITY_RANGE[0],
        max_value=PROBABILITY_RANGE[1],
        default=CLUSTERING_DEFAULTS.identity,
        step=0.01,
        help="A read joins a cluster if it is at least this similar to the cluster's centroid. Too low and different"
             " strands get merged together, too high and noisy reads of the same strand get split up.",
    )

    # Slider for the seed length
    seed_length: int = slider_with_input(
        label="Seed length (k):",
        key="convergence_seed_length",
        min_value=SEED_LENGTH_RANGE[0],
        max_value=SEED_LENGTH_RANGE[1],
        default=CLUSTERING_DEFAULTS.seed_length,
        help="Length of the k-mers used as seeds. Longer seeds have fewer random matches, but every error destroys more"
             " of them - so the higher the error rate, the shorter this should be.",
    )

    # Slider for the minimum seed hits
    min_seed_hits: int = slider_with_input(
        label="Minimum seed hits:",
        key="convergence_min_seed_hits",
        min_value=MIN_SEED_HITS_RANGE[0],
        max_value=MIN_SEED_HITS_RANGE[1],
        default=CLUSTERING_DEFAULTS.min_seed_hits,
        help="How many seeds a read has to share with a centroid before they are aligned.",
    )

    # Max rejects - feels weird adding a slider the upperbound is so large
    max_rejects: int = st.number_input(
        label="Maximum rejects:",
        min_value=MAX_REJECTS_RANGE[0],
        max_value=MAX_REJECTS_RANGE[1],
        step=1,
        value=CLUSTERING_DEFAULTS.max_rejects,
        key="convergence_max_rejects",
        help="How many centroids a read can be rejected by before it gives up and starts its own cluster. Higher is"
             " more accurate but slower.",
        persist_state="session",
    )

    return identity, seed_length, min_seed_hits, max_rejects


def show_clustering_settings(in_path: Path | None, is_json: bool) -> ClusteringConfig | None:
    """
    Settings for the clusterer - the primers to trim off each read, and the seed-then-align settings from VSEARCH.

    Args:
        in_path: The path to the input, used to fill in the primers if it is a JSON. None if there is no input yet.
        is_json: Whether the input is a JSON file saved from the Encoding page.

    Returns:
        The config built with the picked settings, or None if a setting is not valid (e.g. a primer with non-bases).
    """
    st.markdown("**Clustering**")
    st.caption("Greedy centroid clustering, similar to VSEARCH's cluster_fast. Reads go longest first, and each one" 
               " joins the closest cluster that shares enough seeds with it and is similar enough, otherwise it starts"
               " its own cluster.")

    # Primers first, then how each read finds its cluster
    start_primer: str | None
    end_primer: str | None
    primer_error_fraction: float
    start_primer, end_primer, primer_error_fraction = _show_primer_settings(in_path, is_json)

    identity: float
    seed_length: int
    min_seed_hits: int
    max_rejects: int
    identity, seed_length, min_seed_hits, max_rejects = _show_vsearch_settings()

    # Can't build the config with a bad primer, already showed the error above
    if start_primer is None or end_primer is None:
        return None

    # Pydantic checks everything again, the widgets should stop anything bad but just in case
    try:
        return ClusteringConfig(
            start_primer=start_primer,
            end_primer=end_primer,
            primer_error_fraction=primer_error_fraction,
            identity=identity,
            seed_length=seed_length,
            min_seed_hits=min_seed_hits,
            max_rejects=max_rejects,
        )
    except ValidationError as error:
        st.error(f"A clustering setting is not valid: {error.errors()[0]['msg']}")
        return None



#------------------------------------------------------------------------------#
#                                                                              #
#   CONSENSUS SETTINGS                                                         #
#                                                                              #
#------------------------------------------------------------------------------#



# Same as clustering, defaults come from the config
CONSENSUS_DEFAULTS: Final[ConsensusConfig] = ConsensusConfig()


# What spoa calls each alignment type, will just do it in the order they show in the drop down
class AlignmentNames(StrEnum):
    LOCAL = "Local (Smith-Waterman)"
    GLOBAL = "Global (Needleman-Wunsch)"
    SEMI_GLOBAL = "Semi-global (Overlap)"


# What spoa calls each alignment type, in the order they show in the drop down
ALIGNMENT_TYPES: Final[dict[AlignmentNames, Alignment]] = {
    AlignmentNames.LOCAL: 0,
    AlignmentNames.GLOBAL: 1,
    AlignmentNames.SEMI_GLOBAL: 2,
}

# A short description of each alignment type, shown under the drop down
ALIGNMENT_DESCRIPTIONS: Final[dict[AlignmentNames, str]] = {
    AlignmentNames.LOCAL: "Aligns only the best matching part of each read, ignoring the rest. Good when reads only"
                          " partly overlap, but can lose the ends of a strand.",
    AlignmentNames.GLOBAL: "Aligns every read end to end. Works best when reads are about the same length as the"
                           " original strand, which they should be after clustering.",
    AlignmentNames.SEMI_GLOBAL: "End to end, but gaps hanging off either end are free. So reads cut short at the start"
                                " or end don't drag the consensus down.",
}


def show_consensus_settings() -> ConsensusConfig:
    """
    Drop down to pick the alignment type used to build each cluster's partial order graph. Literally just one setting.

    Returns:
        The config built with the picked settings.
    """
    st.markdown("**Consensus**")
    st.caption("Partial order alignment (POA). Every read in a cluster is aligned into one graph, and the heaviest path"
               " through it is the best guess of the original strand.")

    # Start on whatever the config uses by default
    options: list[AlignmentNames] = list(ALIGNMENT_TYPES)
    default_index: int = CONSENSUS_DEFAULTS.alignment_type

    # Selectbox for the alignment type
    choice: AlignmentNames = st.selectbox(
        label="Alignment type:",
        options=options,
        index=default_index,
        key="convergence_alignment_choice",
        help="How each read is aligned against the graph.",
        persist_state="session",            # Keep value for whole session
    )
    st.caption(f"{ALIGNMENT_DESCRIPTIONS[choice]}")

    return ConsensusConfig(alignment_type=ALIGNMENT_TYPES[choice])



#------------------------------------------------------------------------------#
#                                                                              #
#   CLUSTERING AND CONSENSUS                                                   #
#                                                                              #
#------------------------------------------------------------------------------#



# Session state keys for the last run
class ConvergenceResultKeys(StrEnum):
    # Session state is shared across every page, so these need to be different from the other pages' keys
    IN_PATH = "convergence_in_path"
    NUM_READS = "convergence_num_reads"
    STRANDS = "convergence_strands"
    JSON = "convergence_json"
    ERROR = "convergence_error"


def cluster_and_consensus(in_path: Path,
                          is_json: bool,
                          clustering_config: ClusteringConfig,
                          consensus_config: ConsensusConfig) -> None:
    """
    Loads the reads, clusters them, finds the consensus of every cluster, and saves the results to session state. If
    the input is a JSON file, a copy of it with the consensus strands added after its oligos is saved too. Any error is
    saved, so it can be shown on the page instead of crashing it.

    Args:
        in_path: The path to the JSON or text file to cluster.
        is_json: Whether the input is a JSON file saved from the Encoding page, False if it is a text file of oligos.
        clustering_config: The config for the clusterer.
        consensus_config: The config for the consensus algorithm.
    """
    # Forget the last run first, so an error never shows next to old results
    for key in ConvergenceResultKeys:
        st.session_state.pop(key, None)

    # Remember which file these results are for, so they are hidden if the user changes the input
    st.session_state[ConvergenceResultKeys.IN_PATH] = in_path

    # Try to load, cluster, and find the consensus - show error if anything goes wrong
    json_path: Path | None = None
    try:
        # Reads come from the JSON's oligos, or one per line from the text file
        reads: list[str]
        if is_json:
            reads = Serializer.load_framed_oligos(str(in_path))
        else:
            reads = Serializer.load_segments_from_text(str(in_path))

        # A file of only blank lines passes the upload check, but there is nothing to cluster
        if not reads:
            st.session_state[ConvergenceResultKeys.ERROR] = "The input has no oligos to cluster."
            return

        # And now preform the work
        consensus: Consensus = Consensus(consensus_config)
        strands: list[str] = consensus.cluster_and_consensus(reads, Clustering(clustering_config))

        # Add the consensus strands to the JSON's oligos (the old ones are kept), so it can be decoded straight away
        if is_json:
            json_path = Path(mkdtemp(prefix=ViewHelpers.TEMPORARY_FILE_PREFIX)) / f"{in_path.stem}-converged.json"
            Serializer.save_with_added_oligos(str(in_path), strands, str(json_path))

    # The JSON no longer matches what is expected (it is checked on upload, but the encoded one is not)
    except ValidationError as error:
        st.session_state[ConvergenceResultKeys.ERROR] = f"The JSON file is not valid: {error.errors()[0]['msg']}"
        return

    # Our own errors, or spoa failing on the reads
    except (Error, ValueError) as error:
        st.session_state[ConvergenceResultKeys.ERROR] = f"{type(error).__name__}: {error}"
        return

    # Save the results to session state so they can be shown on the page
    st.session_state[ConvergenceResultKeys.NUM_READS] = len(reads)
    st.session_state[ConvergenceResultKeys.STRANDS] = strands
    st.session_state[ConvergenceResultKeys.JSON] = json_path

    # The consensus strands are now the oligos the rest of the pages use (e.g. decoding), same as a fresh encode
    #   This page swaps it back to the input it came from, so the primers are not filled in again either
    if json_path is not None:
        st.session_state[ViewHelpers.ENCODED_JSON_PATH_KEY] = json_path


def show_results(num_reads: int, strands: list[str]) -> None:
    """
    Shows how many reads went in and how many strands came out, one per column. Basically just the main metrics from it.

    Args:
        num_reads: The number of reads that were clustered.
        strands: The consensus strand of every cluster.
    """
    reads_column: Any
    strands_column: Any
    average_column: Any
    reads_column, strands_column, average_column = st.columns(3)

    # Reads that went in
    reads_column.metric(
        label="Reads",
        value=f"{num_reads:,}",
        help="How many reads went into clustering.",
    )

    # How many strands came out
    strands_column.metric(
        label="Consensus strands",
        value=f"{len(strands):,}",
        help="One per cluster. Ideally the same as the number of oligos that were encoded.",
    )

    # Average reads per strand
    average_column.metric(
        label="Reads per strand",
        value=f"{num_reads / len(strands):.2f}",
        help="The average size of a cluster.",
    )



#------------------------------------------------------------------------------#
#                                                                              #
#   SHOW UI                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



# Page title and info
st.title("Clustering and Consensus")
# TODO: add a description of the page
st.write("TODO: add a descirption of the page")


# Input section
st.subheader("Input", divider="gray")

# Where the file to cluster can be found - or None if nothing valid yet
in_path: Path | None
is_json: bool
in_path, is_json = show_input_settings("convergence_input_method", JSON_UPLOAD_KEYS, TEXT_UPLOAD_KEYS)

# A run sets its consensus JSON as the session's oligos (so decoding picks it up), which would make the next run cluster
#   the old oligos plus the consensus strands added to them. Swap it back to the oligos that run started from. Done
#   before the settings, so the primers are filled in from the original too
last_in_path: Path | None = st.session_state.get(ConvergenceResultKeys.IN_PATH, None)
last_json_path: Path | None = st.session_state.get(ConvergenceResultKeys.JSON, None)
if last_in_path is not None and last_json_path is not None and in_path == last_json_path:
    in_path = last_in_path
    st.caption(f"Clustering the original oligos ({last_in_path.name}), not the last consensus ones.")


# Settings section
st.subheader("Settings", divider="gray")

# Two columns - clustering on the left, consensus on the right
clustering_column: Any
consensus_column: Any
clustering_column, consensus_column = st.columns(2)

# Left is clustering 
clustering_config: ClusteringConfig | None
with clustering_column:
    with st.container(border=True):
        clustering_config = show_clustering_settings(in_path, is_json)

# Right is consensus
consensus_config: ConsensusConfig
with consensus_column:
    with st.container(border=True):
        consensus_config = show_consensus_settings()


# Cluster section, only enabled if there is an input and the settings are valid
st.subheader("Cluster and Consensus", divider="gray")

ready: bool = in_path is not None and clustering_config is not None
if in_path is None:
    st.info("Upload a file above, or encode something on the Encoding page, to perform clustering and consensus.")

if st.button("Perform Clustering and Consensus", type="primary", disabled=not ready):
    # When pressed show our little spinner so we know that we are doing something
    with st.spinner("Performing Clustering and Consensus..."):
        # Ensure we can actually cluster (can't reuse bool for typing reasons, why must I stay strongly tyoped)
        if in_path is not None and clustering_config is not None:
            cluster_and_consensus(in_path, is_json, clustering_config, consensus_config)

# Only show results for the input currently picked, so old results don't show next to a new file. Read again, as
#   pressing the button just replaced the last run
last_in_path = st.session_state.get(ConvergenceResultKeys.IN_PATH, None)
last_json_path = st.session_state.get(ConvergenceResultKeys.JSON, None)
if last_in_path == in_path:
    # If there was an error show
    if ConvergenceResultKeys.ERROR in st.session_state:
        st.error(f"Clustering and consensus failed. {st.session_state[ConvergenceResultKeys.ERROR]}")

    # Everything was great - show the results and the downloads
    if ConvergenceResultKeys.STRANDS in st.session_state and last_in_path is not None:
        st.success("Clustering and consensus finished successfully!")

        show_results(st.session_state[ConvergenceResultKeys.NUM_READS], st.session_state[ConvergenceResultKeys.STRANDS])
        # Named after what was clustered, not the consensus JSON it might be using now
        show_downloads(
            in_path=last_in_path,
            strands=st.session_state[ConvergenceResultKeys.STRANDS],
            json_path=last_json_path,
            json_help="The JSON with the consensus strands added to its oligos, can be uploaded on the decode page.",
            strand_help="The consensus strands, one per line. Cannot be uploaded on the decode page.",
            suffix="converged"
        )