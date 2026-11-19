# TODO: add module descirption

from enum import StrEnum
from pathlib import Path
from tempfile import mkdtemp
from typing import Final

import pandas as pd
import streamlit as st
from pydantic import ValidationError
from streamlit.runtime.uploaded_file_manager import UploadedFile

from Model.Encode.ErrorCorrection.InnerCodes import CheckSum, RSInner
from Model.Encode.ErrorCorrection.OuterCodes import DirectDNAGoldmanRedundancy, DirectDNARSOuter, DirectDNAXOR, \
    GoldmanRedundancy, OuterCodes, RSOuter, XOR
from Model.Encode.Indexer import DNAMapIndexer, EnumerativeIndexer, GoldmanIndexer, Indexer
from Model.Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Model.Encode.Methods.Encoding import Encoding, EncodingStats
from Model.Encode.Methods.HuffmanCoding import BitScanQuaternary, ComplementaryTree, Goldman, Pgram, Quaternary, Ternary
from Model.Encode.OligoFramer import FramerConfig, OligoFramer
from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class Inputs(StrEnum):
    # Need to make these keys as Streamlit doesn't keep the text_input box written, when user switches. Just easier.
    FILE_UPLOAD = "File Upload"
    TEXT_INPUT = "Text Input"

    UPLOAD_FILE_NAME = "uploaded_file_name"
    UPLOAD_FILE_PATH = "uploaded_file_path"
    UPLOAD_ERROR = "upload_error"

    SUBMITTED_TEXT = "submitted_text"
    SUBMITTED_TEXT_PATH = "submitted_text_path"


def _clear_uploaded_file() -> None:
    """
    Forget the uploaded file, so nothing from the file upload is used for encoding.
    """
    st.session_state.pop(Inputs.UPLOAD_FILE_NAME, None)
    st.session_state.pop(Inputs.UPLOAD_FILE_PATH, None)
    st.session_state.pop(Inputs.UPLOAD_ERROR, None)


def _clear_submitted_text() -> None:
    """
    Forget the submitted text and empty the text box, so nothing from the text input is used for encoding. Only call
    this while the text box is not drawn (e.g. from the uploader's callback).
    """
    st.session_state.pop(Inputs.SUBMITTED_TEXT, None)
    st.session_state.pop(Inputs.SUBMITTED_TEXT_PATH, None)
    st.session_state[Inputs.TEXT_INPUT] = ""


def _contents_to_file(contents: str, filename: str) -> Path:
    """
    Write the contents to a temporary file and return the path to that file.

    Returns:
        A Path object pointing to the temporary file containing the contents.
    """
    # Want to write to temp file as our backend accepts an in-path. Makes it much easier. Using mkdtemp instead of
    #   TemporaryDirectory, as TemporaryDirectory deletes the folder as soon as the object is garbage collected (i.e.
    #   right after this function returns), so the path we hand back would point at nothing.
    temp_directory: str = mkdtemp(prefix="dna_codec_")
    file_path: Path = Path(temp_directory) / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(contents)
    
    return file_path


def _store_uploaded_file() -> None:
    """
    Callback for the file uploader, runs only when the user uploads a new file or removes one. Streamlit seems to just 
    drop the uploaded file when the uploader isn't drawn (when user switches to text input), so we keep our own copy.
    Using a callback means we can tell user removed the file" apart from uploader was hidden and came back empty.
    """
    upload: UploadedFile | None = st.session_state["file_upload"]
    st.session_state.pop(Inputs.UPLOAD_ERROR, None)

    # User removed the file, so just clear it 
    if upload is None:
        _clear_uploaded_file()
        return

    # Ensure we can read the file as text, and handle any errors - letting users know if the file is not valid text
    file_contents: str
    try:
        file_contents = upload.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        # Can't show the error from inside a callback (it would draw above everything), so flag it for the page
        _clear_uploaded_file()
        st.session_state[Inputs.UPLOAD_ERROR] = upload.name
        return

    # Only one input is used at a time, so a valid file replaces any submitted text
    _clear_submitted_text()

    # Convert to a temporary file to be used by the encoder - done once here, not on every rerun
    st.session_state[Inputs.UPLOAD_FILE_NAME] = upload.name
    st.session_state[Inputs.UPLOAD_FILE_PATH] = _contents_to_file(file_contents, upload.name)


def _file_uploader() -> Path | None:
    """
    Upload a text-based file and return the path to a temporary file containing its contents. The file is kept in
    session state, so it is still used after switching to text input and back.

    Returns:
        The path to the temporary file containing the uploaded file's contents, or None if there is no valid file.
    """
    # Actual file upload button
    st.file_uploader(
        label="Upload any text-based file to encode you wish to encode on:",
        key="file_upload",
        max_upload_size=None,               # No limit on file size - probs a bad idea but thats on the user
        accept_multiple_files=False,        # Only one at a time
        on_change=_store_uploaded_file,     # Save (or forget) the file only when the user changes it
    )

    # The last file the user picked could not be read as text
    if Inputs.UPLOAD_ERROR in st.session_state:
        st.error(f"{st.session_state[Inputs.UPLOAD_ERROR]} is not a valid text file. Please upload a text-based file.")
        return None

    # Nothing uploaded yet (or it was removed) - nothing to encode
    if Inputs.UPLOAD_FILE_PATH not in st.session_state:

        # Let the user know uploading will throw away the text they already submitted if we have some
        if Inputs.SUBMITTED_TEXT_PATH in st.session_state:
            st.warning("Currently using your submitted text. Uploading a file will replace it.")

        return None

    st.success(f"File uploaded successfully! Using {st.session_state[Inputs.UPLOAD_FILE_NAME]} for encoding.")

    return st.session_state[Inputs.UPLOAD_FILE_PATH]
    

def _area_input() -> Path | None:
    """
    Get text input from a text area inside a form, so nothing is used until the user presses submit. The submitted
    text is kept in session state, so it (and the success message) survives reruns from other widgets on the page.

    Returns:
        The path to the temporary file containing the submitted text, or None if nothing has been submitted yet.
    """
    # A form batches the widgets inside it, so typing in the text area does not rerun the page - only submit does
    with st.form(key="text_input_form"):
        input_text: str = st.text_area(
            label="Enter what you like to encode:",
            key=Inputs.TEXT_INPUT,
            required=True,                  # Browser blocks submitting an empty or whitespace only box
            # Streamlit  for some reason just throws away a widget's value when it isn't drawn, which empties the box 
            #   while the success message (from our own session state) stays.
            persist_state="session"
        )
        submitted: bool = st.form_submit_button("Submit")

    # Submit is only True on the one rerun right after it was pressed, so save what was submitted to session state
    if submitted:
        # The required check runs in the browser and can be bypassed, so check again here
        if not input_text.strip():
            st.error("Please enter some text before submitting.")
            return None

        # Only one input is used at a time, so submitted text replaces any uploaded file
        _clear_uploaded_file()

        # Update our session state
        st.session_state[Inputs.SUBMITTED_TEXT] = input_text
        st.session_state[Inputs.SUBMITTED_TEXT_PATH] = _contents_to_file(input_text, "text_input.txt")

    # Nothing submitted yet (or the last submit was empty) - nothing to encode
    if Inputs.SUBMITTED_TEXT_PATH not in st.session_state:
        
        # Let the user know submitting will throw away the file they already uploaded
        if Inputs.UPLOAD_FILE_PATH in st.session_state:
            st.warning(f"Currently using {st.session_state[Inputs.UPLOAD_FILE_NAME]}. Submitting text will replace it.")

        return None

    st.success("Text submitted successfully! Using inputted text for encoding.")
    
    return st.session_state[Inputs.SUBMITTED_TEXT_PATH]


def show_input_settings() -> Path | None:
    """
    Show the input settings to the user.

    Returns:
        The path to the temporary file containing the input to encode, or None if there is no valid input yet.
    """
    # Now show to the user the option to choose between file upload and text area input
    input_method = st.segmented_control(
        label="Choose an input method:",
        options=[Inputs.FILE_UPLOAD, Inputs.TEXT_INPUT],
        default=Inputs.FILE_UPLOAD,                # Default to file upload
        required=True                       # Must be either one
    )

    # What to show based on the input method selected. Either way we end up with a path to encode, or None if the user
    #   has not given us anything valid yet.
    in_path: Path | None = None
    # User wants to upload a file
    if input_method == Inputs.FILE_UPLOAD:
        in_path = _file_uploader()
    # User wants a text box
    elif input_method == Inputs.TEXT_INPUT:
        in_path = _area_input()

    return in_path



#------------------------------------------------------------------------------#
#                                                                              #
#   ENCODER SETTINGS                                                           #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants for the encoder settings - the options in the drop down, and the session state keys for their arguments
class EncoderNames(StrEnum):
    DNA_MAP = "DNA Map"
    DNA_MAP_HUFFMAN = "DNA Map + Huffman"

    QUATERNARY = "Quaternary Huffman"
    BITSCAN_QUATERNARY = "Bit-Scan Quaternary Huffman"

    TERNARY = "Ternary Huffman"
    GOLDMAN = "Goldman (Rotating Tree)"

    PGRAM = "P-gram Huffman"
    COMPLEMENTARY_TREE = "Complementary Tree"

    # Addtional session state settings
    BITSCAN_K = "bitscan_k"
    PGRAM_M = "pgram_m"


# Plain Huffman is left out as it outputs bits, not bases, so it can't be framed into oligos
ENCODER_OPTIONS: Final[list[EncoderNames]] = [
    EncoderNames.DNA_MAP,
    EncoderNames.DNA_MAP_HUFFMAN,

    EncoderNames.QUATERNARY,
    EncoderNames.BITSCAN_QUATERNARY,

    EncoderNames.TERNARY,
    EncoderNames.GOLDMAN,

    EncoderNames.PGRAM,
    EncoderNames.COMPLEMENTARY_TREE,    
]

# A short description of each encoder, shown under the drop down so the user knows what they picked
ENCODER_DESCRIPTIONS: Final[dict[EncoderNames, str]] = {
    EncoderNames.DNA_MAP: "Direct mapping, 00 → A, 01 → C, 10 → G, 11 → T. This method has no compression.",
    EncoderNames.DNA_MAP_HUFFMAN: "Binary Huffman compression first, then the direct 2 bits per base mapping.",

    EncoderNames.QUATERNARY: "Huffman with all four bases (A, C, G, T), each base holds up to 2 bits. Essentially works"
                             " like a binary Huffman but with a larger alphabet - so four children per node with their"
                             " respective DNA bases.",
    EncoderNames.BITSCAN_QUATERNARY: "Quaternary Huffman that scans k bits at a time, larger k packs better but the "
                                     " table grows quickly. It uses a quaternary tree like the one described in the "
                                     " Quaternary encoding method.",

    EncoderNames.TERNARY: "Huffman with three bases per branch (A, C, G), a child never repeats its parent's base.",
    EncoderNames.GOLDMAN: "Rotating tree Huffman. First the input is converted into trits using a Huffman tree with "
                          " three children per node. Then each trit is converted to a base using a rotating map, where"
                          " the base chosen depends on the previous base - so the three options are always the bases"
                          " that are different from the last one.",

    EncoderNames.PGRAM: "Huffman over variable length substrings (p-grams) up to length m. Larger m can compress better"
                        " but the table grows quickly - however this should be amortized over a large dataset and"
                        " frozen. The Huffman codes are built over the complementary tree, a binary tree where even"
                        " levels useG and T and odd levels use A and C, and each code word is complemented when needed"
                        " so thealternation carries on across code words.",
    EncoderNames.COMPLEMENTARY_TREE: "The Huffman codes are built over the complementary tree, a binary tree where even"
                                     " levels use G and T and odd levels use A and C, and each code word is "
                                     " complemented when needed so the alternation carries on across code words.",
}

PRODUCES_HOMOPOLYMERS: str = "This method produces homopolymers."
PREVENTS_HOMOPOLYMERS: str = "This method does not produce homopolymers."
# Whether each encoder produces homopolymers, kept apart from the description so it can be shown in bold
HOMOPOLYMER_ENCODER_NOTES: Final[dict[EncoderNames, str]] = {
    EncoderNames.DNA_MAP: PRODUCES_HOMOPOLYMERS,
    EncoderNames.DNA_MAP_HUFFMAN: PRODUCES_HOMOPOLYMERS,

    EncoderNames.QUATERNARY: PRODUCES_HOMOPOLYMERS,
    EncoderNames.BITSCAN_QUATERNARY: PRODUCES_HOMOPOLYMERS,

    EncoderNames.TERNARY: PREVENTS_HOMOPOLYMERS[:-1] + " of size >= 2.",
    EncoderNames.GOLDMAN: PREVENTS_HOMOPOLYMERS,

    EncoderNames.PGRAM: PREVENTS_HOMOPOLYMERS,
    EncoderNames.COMPLEMENTARY_TREE: PREVENTS_HOMOPOLYMERS,
}

# Ranges for the encoder arguments. k matches the measured bases per byte table (estimating needs an entry for k), m
#   past 16 is allowed by Pgram but the estimate is capped at 16, so no point going higher here.
BITSCAN_K_RANGE: Final[tuple[int, int]] = (min(BitScanQuaternary.BITSCAN_BASES_PER_BYTE),
                                           max(BitScanQuaternary.BITSCAN_BASES_PER_BYTE))
PGRAM_M_RANGE: Final[tuple[int, int]] = (min(Pgram.PGRAM_BASES_PER_BYTE), 
                                         max(Pgram.PGRAM_BASES_PER_BYTE))
BITSCAN_K_DEFAULT: Final[int] = 16
PGRAM_M_DEFAULT: Final[int] = 3


def _sync_widget(source_key: str, target_key: str) -> None:
    """
    Callback to copy one widget's value into another, so a slider and number input always show the same thing.

    Args:
        source_key: The key of the widget the user just changed.
        target_key: The key of the widget to update.
    """
    st.session_state[target_key] = st.session_state[source_key]


def _slider_with_input(label: str, key: str, min_value: int, max_value: int, default: int, help: str) -> int:
    """
    A slider with a number input beside it, kept in sync. The slider is quick to drag, the number input is easier for
    an exact value. Both keep their value for the whole session, so switching encoders and back doesn't reset them.

    Args:
        label: The label shown above the slider.
        key: The base key, the slider and number input use this with "_slider" and "_number" added.
        min_value: The lowest value allowed.
        max_value: The highest value allowed.
        default: The value used before the user changes anything.
        help: The tooltip shown beside the label.

    Returns:
        The value picked.
    """
    slider_key: str = f"{key}_slider"
    number_key: str = f"{key}_number"

    # Set the starting value through session state instead of value=, Streamlit warns if a widget gets both
    st.session_state.setdefault(slider_key, default)
    st.session_state.setdefault(number_key, default)

    # Create two columns, one for the slider and one for the number input, and align them at the bottom so they line up
    #   giving more space to the slider as it is bigger.
    slider_column, number_column = st.columns([3, 1], vertical_alignment="bottom")

    # Slider is the main widget, so it has the label and help. 
    with slider_column:
        value: int = st.slider(
            label=label,
            min_value=min_value,
            max_value=max_value,
            step=1,
            key=slider_key,
            help=help,
            on_change=_sync_widget,             # Ensure we keep keep the number we saved
            args=(slider_key, number_key),
            persist_state="session",            # Same here - keep the value for the whole session
        )

    # The number input is hidden, as the slider already has the label. It is only there to let the user type in a value
    with number_column:
        st.number_input(
            label=label,
            min_value=min_value,
            max_value=max_value,
            step=1,
            key=number_key,
            on_change=_sync_widget,            # Ensure we keep the slider we saved
            args=(number_key, slider_key),
            label_visibility="collapsed",      # Slider already has the label
            persist_state="session",           # Keep the value for the whole session
        )

    return value


def _find_optimal_k(in_path: str) -> int:
    """
    Tries every k and returns the one with the best coding density for the file. Cached, as it encodes the file once
    per k, so it can be slow on big files.

    Args:
        in_path: The path to the file to encode.

    Returns:
        The k with the best coding density.
    """
    return BitScanQuaternary().find_and_set_optimal_k(in_path, BITSCAN_K_RANGE[0], BITSCAN_K_RANGE[1])[0]


def _set_optimal_k(in_path: Path) -> None:
    """
    Callback for the find best k button, moves the slider and number input to the best k for the input.

    Args:
        in_path: The path to the file to encode.
    """
    # Just easier to use a toast here because a spinner needs to be drawn with session state and dont feel like messing
    #   with that rn. Toast is a small popup that goes away after a few seconds like finding k
    st.toast("Finding the best k...")
    optimal_k: int = _find_optimal_k(str(in_path))
    st.session_state[f"{EncoderNames.BITSCAN_K}_slider"] = optimal_k
    st.session_state[f"{EncoderNames.BITSCAN_K}_number"] = optimal_k


def show_encoder_settings(in_path: Path | None) -> Encoding:
    """
    Drop down to pick the encoder, plus the extra arguments for the encoders that have any.

    Args:
        in_path: The path to the input, used to find the best k for bit-scan. None if there is no input yet.

    Returns:
        The encoder built with the picked settings.
    """
    st.markdown("**Encoder**")

    # What the user wants
    choice: EncoderNames = st.selectbox(
        label="Encoding method:",
        options=ENCODER_OPTIONS,
        key="encoder_choice",
        persist_state="session",
    )
    # Captions take markdown, so ** makes the homopolymer note bold
    st.caption(f"{ENCODER_DESCRIPTIONS[choice]} **{HOMOPOLYMER_ENCODER_NOTES[choice]}**")

    # Only bit-scan and p-gram take any arguments, every other encoder is built as is
    match choice:
        case EncoderNames.BITSCAN_QUATERNARY:
            # Slider for the bit scan
            k: int = _slider_with_input(
                label="Bits scanned at a time (k):",
                key=EncoderNames.BITSCAN_K,
                min_value=BITSCAN_K_RANGE[0],
                max_value=BITSCAN_K_RANGE[1],
                default=BITSCAN_K_DEFAULT,
                help="How many bits are read as one symbol. Larger k can compress better but makes the Huffman table"
                     " much bigger.",
            )

            # Want to show a button to find best k - but only if there is an input
            st.button(
                "Find best k for my input",
                on_click=_set_optimal_k,            # me when event-driven
                args=(in_path,),
                disabled=in_path is None,           # Disable if there is no input yet
                help="Tries every k on your input and picks the one with the best coding density.",
            )
            return BitScanQuaternary(k=k)

        case EncoderNames.PGRAM:
            # Slider for the p-gram length
            m: int = _slider_with_input(
                label="Maximum p-gram length (m):",
                key=EncoderNames.PGRAM_M,
                min_value=PGRAM_M_RANGE[0],
                max_value=PGRAM_M_RANGE[1],
                default=PGRAM_M_DEFAULT,
                help="The longest substring that can get its own Huffman code.",
            )
            return Pgram(m=m)

        # Rest of the encoders have no arguments, so just return as is
        case EncoderNames.DNA_MAP:
            return DNAMap()
        case EncoderNames.DNA_MAP_HUFFMAN:
            return DNAMapHuffman()
        case EncoderNames.QUATERNARY:
            return Quaternary()
        case EncoderNames.TERNARY:
            return Ternary()
        case EncoderNames.GOLDMAN:
            return Goldman()
        case EncoderNames.COMPLEMENTARY_TREE:
            return ComplementaryTree()
        case _:
            raise ValueError(f"Unknown encoder: {choice}")


#------------------------------------------------------------------------------#
#                                                                              #
#   OLIGO SETTINGS                                                             #
#                                                                              #
#------------------------------------------------------------------------------#


# Defaults for the oligo, same as test_encoders.py
OLIGO_LENGTH_DEFAULT: Final[int] = 200
PRIMER_DEFAULT: Final[str] = "ACGTACGT"
OVERHEAD_MARGIN_DEFAULT: Final[float] = 0.1


def _clean_primer(primer: str) -> str | None:
    """
    Cleans up a primer the user typed, so lowercase and spaces are fine.

    Args:
        primer: The primer as typed.

    Returns:
        The primer in uppercase with no spaces, or None if it has anything other than A, C, G, and T.
    """
    primer = primer.replace(" ", "").upper()

    # Only bases allowed - an empty primer is fine, the framer just leaves it off
    if not set(primer) <= Encoding.DNA_BASES:
        return None

    return primer


def show_oligo_settings() -> tuple[int, str | None, str | None, float]:
    """
    Settings for the oligo itself - how long it is, the primers on each end, and how much room to leave for the index.

    Returns:
        The oligo length, start primer, end primer (None if a primer is not valid), and the overhead margin.
    """
    st.markdown("**Oligo**")

    # Oligo length input - feels weird adding a slider the upperbound is so large and unrealistic
    oligo_length: int = st.number_input(
        label="Oligo length:",
        min_value=100,
        max_value=1000,
        step=1,
        value=OLIGO_LENGTH_DEFAULT,
        key="oligo_length",
        help="The total length of each oligo in bases, including the primers.",
        persist_state="session",        # Keep value for whole session
    )

    # Primers as text, cleaned up after so lowercase or spaces don't count as wrong
    start_primer_text: str = st.text_input(
        label="Start primer:",
        value=PRIMER_DEFAULT,
        key="start_primer",
        help="Bases added to the start of every oligo. Leave empty for no primer.",
        persist_state="session",
    )
    end_primer_text: str = st.text_input(
        label="End primer:",
        value=PRIMER_DEFAULT,
        key="end_primer",
        help="Bases added to the end of every oligo. Leave empty for no primer.",
        persist_state="session",
    )

    start_primer: str | None = _clean_primer(start_primer_text)
    end_primer: str | None = _clean_primer(end_primer_text)

    # Let the user know right away if a primer is wrong, instead of when they press encode
    if start_primer is None:
        st.error(f"Start primer must be in set of {Encoding.DNA_BASES}")
    if end_primer is None:
        st.error(f"End primer must be in set of {Encoding.DNA_BASES}")

    # Overhead margin input, a float between 0 and. Not gonna use sliders for floats
    overhead_margin: float = st.number_input(
        label="Overhead margin:",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=OVERHEAD_MARGIN_DEFAULT,
        key="overhead_margin",
        help="Extra room left when guessing how many oligos are needed, used to pick the index width. A bigger margin"
             " is less likely to need a wider index part way through, but can waste a base of index.",
        persist_state="session",
    )

    return oligo_length, start_primer, end_primer, overhead_margin


#------------------------------------------------------------------------------#
#                                                                              #
#   INDEXER SETTINGS                                                           #
#                                                                              #
#------------------------------------------------------------------------------#


class IndexerNames(StrEnum):
    DNA_MAP = "DNA Map Indexer"
    GOLDMAN = "Goldman Indexer"
    ENUMERATIVE = "Enumerative Indexer"

    # Additional session state settings
    ENUMERATIVE_N = "enumerative_n"


# The indexers to choose from
INDEXER_OPTIONS: Final[list[IndexerNames]] = [
    IndexerNames.DNA_MAP,
    IndexerNames.GOLDMAN,
    IndexerNames.ENUMERATIVE
]

# A short description of each indexer, shown under the drop down
INDEXER_DESCRIPTIONS: Final[dict[IndexerNames, str]] = {
    IndexerNames.DNA_MAP: "Direct mapping, 00 → A, 01 → C, 10 → G, 11 → T. Has no run length limit, so the index can"
                          " have homopolymers of any length.",
    IndexerNames.GOLDMAN: "Rotating map, the index number is written in trits and each trit picks one of the three"
                          " base different from the last one. So the index has no homopolymers.",
    IndexerNames.ENUMERATIVE: "Enumerative indexing, counts every sequence that keeps runs under the maximum run length"
                              " and gives each index its own one. Looks at the bases before and after the index, so no"
                              " run form with the primer or payload either.",
}


def show_indexer_settings() -> tuple[type[Indexer], int, bool]:
    """
    Drop down to pick the indexer, plus the extra arguments for the indexers that have any.

    Returns:
        The indexer type picked and the max run length (-1 if the indexer doesn't use one) and if it produces 
        homopolymers.
    """
    st.markdown("**Indexer**")

    # Selectbock for the indexer
    choice: IndexerNames = st.selectbox(
        label="Indexing method:",
        options=INDEXER_OPTIONS,
        key="indexer_choice",
        help="The indexing method to use.",
        persist_state="session",        # Keep value for who session
    )
    st.caption(f"{INDEXER_DESCRIPTIONS[choice]}")

    # User pick
    pick: tuple[type[Indexer], int, bool]
    # Only enumerative takes any arguments, every other indexer is built as is
    match choice:
        case IndexerNames.ENUMERATIVE:
            # Slider for the enumerative n
            max_run_length: int = _slider_with_input(
                label="Maximum homopolymer runlength:",
                key=IndexerNames.ENUMERATIVE_N,
                min_value=1,
                max_value=10,
                default=1,
                help="The maximum homopolymer run in the indexer.",
            )
            homopolymer_free: bool = max_run_length == 1
            pick = (EnumerativeIndexer, max_run_length, homopolymer_free)

        # Rest of the indexers have no arguments (will set max_run_length to -1), so just return as is
        case IndexerNames.DNA_MAP:
            pick = (DNAMapIndexer, -1, False)
        case IndexerNames.GOLDMAN:
            pick = (GoldmanIndexer, -1, True)
        case _:
            raise ValueError(f"Unknown indexer: {choice}")

    return pick



#------------------------------------------------------------------------------#
#                                                                              #
#   INNER CODE SETTINGS                                                        #
#                                                                              #
#------------------------------------------------------------------------------#



class InnerCodeNames(StrEnum):
    NONE = "None"
    REED_SOLOMON = "Reed-Solomon"

    REED_SOLOMON_PARITY = "reed_solomon_parity"

INNER_CODES: Final[list[InnerCodeNames]] = [
    InnerCodeNames.NONE,
    InnerCodeNames.REED_SOLOMON,
]

INNER_CODE_DESCRIPTIONS: Final[dict[InnerCodeNames, str]] = {
    InnerCodeNames.NONE: "No inner error correction code is used. The checksum can still find a bad oligo, but can't"
                         " fix it.",
    InnerCodeNames.REED_SOLOMON: "Reed-Solomon parity is added to every oligo, so a few bad bases in an oligo can be"
                                 " fixed. Only corrects substitution errors, so indel errors are not corrected."
}


def show_inner_correction_settings(indexer: type[Indexer], max_run_length: int) -> RSInner | None:
    """
    Drop down to pick the inner error correction code, plus the extra arguments for the codes that have any.

    Args:
        indexer: The indexer type picked, the parity is written to DNA the same way as the index.
        max_run_length: The max run length from the indexer (-1 if it doesn't use one).

    Returns:
        The inner code built with the picked settings, or None for no inner code.
    """
    st.markdown("**Inner Error Correction Code**")

    choice: InnerCodeNames = st.selectbox(
        label="Inner error correction code:",
        options=INNER_CODES,
        key="inner_code_choice",
        help="The inner error correction code to use.",
        persist_state="session",        # Keep value for who session
    )

    st.caption(f"{INNER_CODE_DESCRIPTIONS[choice]}")

    match choice:
        case InnerCodeNames.NONE:
            return None
        
        case InnerCodeNames.REED_SOLOMON:
            # Slider for the number of parity bytes
            parity_bytes: int = _slider_with_input(
                label="Parity bytes:",
                key=InnerCodeNames.REED_SOLOMON_PARITY,
                min_value=2,
                max_value=16,
                default=4,
                help="Parity bytes added to each oligo. Every 2 parity 'bytes' can fix 1 bad 'byte', but takes up room " \
                     " the payload could have used. A 'byte' is 4 bases.",
            )
            return RSInner(indexer=indexer, parity_bytes=parity_bytes, max_run_length=max_run_length)

        case _:
            raise ValueError(f"Unknown inner error correction code: {choice}")



#------------------------------------------------------------------------------#
#                                                                              #
#   OUTER CODE SETTINGS                                                        #
#                                                                              #
#------------------------------------------------------------------------------#



class OuterCodeNames(StrEnum):
    NONE = "None"
    GOLDMAN_REDUNDANCY = "Goldman Redundancy"
    XOR = "XOR"
    REED_SOLOMON = "Reed-Solomon"

    # Additional session state settings
    REED_SOLOMON_REDUNDANCY = "reed_solomon_redundancy"

OUTER_CODES: Final[list[OuterCodeNames]] = [
    OuterCodeNames.NONE,
    OuterCodeNames.GOLDMAN_REDUNDANCY,
    OuterCodeNames.XOR,
    OuterCodeNames.REED_SOLOMON,
]

OUTER_CODE_DESCRIPTIONS: Final[dict[OuterCodeNames, str]] = {
    OuterCodeNames.NONE: "No outer error correction code is used, so a lost oligo stays lost.",
    OuterCodeNames.GOLDMAN_REDUNDANCY: "Four quarter shifted copies of the payload, so every quarter shows up in"
                                        " four different oligos. Simple, but takes about four times the oligos.",
    OuterCodeNames.XOR: "Adds one parity oligo for every two payload oligos, so one lost oligo in each block of"
                        " three can be rebuilt.",
    OuterCodeNames.REED_SOLOMON: "Adds parity oligos across all the oligos, so lost oligos can be rebuilt as"
                                 " long as not too many are missing.",
}



def show_outer_correction_settings(is_homopolymer_free: bool) -> OuterCodes | None:
    """
    Drop down to pick the outer error correction code, plus the extra arguments for the codes that have any.

    Args:
        is_homopolymer_free: True if the encoder is guaranteed to produce no homopolymers, False otherwise.

    Returns:
        The outer code built with the picked settings, or None for no outer code.
    """
    st.markdown("**Outer Error Correction Code**")

    # User picks what htey want
    choice: OuterCodeNames = st.selectbox(
        label="Outer error correction code:",
        options=OUTER_CODES,
        key="outer_code_choice",
        help="The outer error correction code to use. If the encoder or indexer produces homopolymers, XOR and RS Outer"
             " codes are switched to the direct DNA Map verisons. Done as normal ones don't work with homopolymers.",
        persist_state="session",        # Keep value for who session
    )

    st.caption(f"{OUTER_CODE_DESCRIPTIONS[choice]}")
    
    # Basically if the enum
    match choice:
        case OuterCodeNames.NONE:
            return None
        case OuterCodeNames.GOLDMAN_REDUNDANCY:
            return DirectDNAGoldmanRedundancy() if not is_homopolymer_free else GoldmanRedundancy()
        case OuterCodeNames.XOR:
            return DirectDNAXOR() if not is_homopolymer_free else XOR()
        
        case OuterCodeNames.REED_SOLOMON:
            # Slider is a whole number percent, the outer code wants it between 0 and 1
            redundancy_percent: int = _slider_with_input(
                label="Redundancy (%):",
                key=OuterCodeNames.REED_SOLOMON_REDUNDANCY,
                min_value=1,
                max_value=100,
                default=20,
                help="How much parity to add compared to the payload, e.g. 20% adds 1 parity oligo for every 5 payload"
                     " oligos. Does add filler bases if payload is not a multiple of 6.",
            )

            if not is_homopolymer_free:
                return DirectDNARSOuter(redundancy=redundancy_percent / 100)
            return RSOuter(redundancy=redundancy_percent / 100)

        case _:
            raise ValueError(f"Unknown outer error correction code: {choice}")



#------------------------------------------------------------------------------#
#                                                                              #
#   ENCODING                                                                   #
#                                                                              #
#------------------------------------------------------------------------------#


# Session state keys for the last run
class ResultKeys(StrEnum):
    FRAMER = "framer"
    ORIGINAL_SIZE = "original_size"
    JSON = "framer_json"
    ERROR = "encode_error"

# Runs of at least this length are counted, same as test_encoders.py
RUN_LENGTH_K: Final[int] = 3


def encode(in_path: Path, 
           oligo_length: int, 
           overhead_margin: float, 
           start_primer: str, 
           end_primer: str, 
           encoder: Encoding, 
           indexer: tuple[type[Indexer], int], 
           inner_codes: RSInner | None, 
           outer_codes: OuterCodes | None) -> None:
    """
    Builds the config from every setting, frames the oligos, and saves the results to session state. Any error is saved
    too, so it can be shown on the page instead of crashing it.

    Args:
        in_path: The path to the file to encode.
        oligo_length: The length of each oligo.
        overhead_margin: The extra room left when guessing how many oligos are needed, used to pick the index width.
        start_primer: The primer added to the start of every oligo.
        end_primer: The primer added to the end of every oligo.
        encoder: The encoder to use.
        indexer: The indexer type and max run length to use.
        inner_codes: The inner error correction code to use, or None for no inner code.
        outer_codes: The outer error correction code to use, or None for no outer code.
    """
    # Forget the last run first, so an error never shows next to old results
    for key in ResultKeys:
        st.session_state.pop(key, None)

    # The indexer and checksum are built here, each run needs its own new indexer as the framer changes its width
    indexer_type: type[Indexer] = indexer[0]
    max_run_length: int = indexer[1]
    new_indexer: Indexer
    if indexer_type is EnumerativeIndexer:
        new_indexer = EnumerativeIndexer(width=-1, max_run_length=max_run_length)
    else:
        new_indexer = indexer_type(width=-1)

    # Try to set up the config and frame - show big error if anything goes wrong
    try:
        config: FramerConfig = FramerConfig(
            in_path=str(in_path),
            oligo_length=oligo_length,
            overhead_margin=overhead_margin,
            start_primer=start_primer,
            end_primer=end_primer,
            encoder=encoder,
            indexer=new_indexer,
            checksum=CheckSum(indexer_type, inner_code=inner_codes is not None, max_run_length=max_run_length),
            inner_correction=inner_codes,
            outer_correction=outer_codes
        )

        framer: OligoFramer = OligoFramer(config)
        framer.frame_all_oligos()

        # Save the framer to a JSON file now, so it can be downloaded for the decode page
        json_path: Path = Path(mkdtemp(prefix="dna_codec_")) / f"{Path(in_path).stem}_framed.json"
        Serializer.save_framer(framer, str(json_path))

    # A setting the config doesn't accept
    except ValidationError as error:
        st.session_state[ResultKeys.ERROR] = f"A setting is not valid: {error.errors()[0]['msg']}"
        return
    
    # Our own errors (e.g. oligo too short for the payload) or anything else the framer hits
    except (Error, ValueError) as error:
        st.session_state[ResultKeys.ERROR] = f"{type(error).__name__}: {error}"
        return

    st.session_state[ResultKeys.FRAMER] = framer
    st.session_state[ResultKeys.ORIGINAL_SIZE] = Path(in_path).stat().st_size * 8      # Bytes to bits
    st.session_state[ResultKeys.JSON] = json_path


def show_oligo_layout(framer: OligoFramer) -> None:
    """
    Shows how many bases each part of the oligo uses, as one sideways bar split into the parts.

    Args:
        framer: The framer after frame_all_oligos.
    """
    config: FramerConfig = framer.config

    # The inner code is optional, so if it is not used it has no length
    inner_length: int = 0
    if config.inner_correction:
        inner_length = config.inner_correction.get_dna_bases()

    # Numbered so the parts keep this order, since the chart sorts them alphabetically
    layout: dict[str, int] = {
        "1. Start primer": len(config.start_primer),
        "2. Index": config.indexer.width,
        "3. Payload": config.payload_length,
        "4. Filler": config.filler_length,
        "5. Checksum": config.checksum.get_dna_bases(),
        "6. Inner RS": inner_length,
        "7. End primer": len(config.end_primer),
    }
    # Set the data up based on the layout. The one row is named Oligo, which is the label shown next to the bar
    #   (without a name it would just show 0)
    data: pd.DataFrame = pd.DataFrame(index=["Oligo"])
    for part, length in layout.items():
        data[part] = [length]

    st.markdown("**Oligo layout**")

    # One row named Oligo, a column per part, so it is stacked into one bar. With horizontal the labels swap axes, so
    #   y_label ends up along the bottom
    st.bar_chart(
        data=data,
        horizontal=True,
        x_label="",
        y_label="Number of Bases",
        height=250,
    )


def show_info(framer: OligoFramer, original_size: int) -> None:
    """
    Shows the same info as OligoFramer.print_info - the components, the layout, and measurements of the oligos.

    Args:
        framer: The framer after frame_all_oligos.
        original_size: The size of the original file in bits.
    """
    config: FramerConfig = framer.config

    # Total number of bases across every oligo, primers included
    total_bases: int = 0
    for oligo in framer.framed_oligos:
        total_bases += len(oligo)

    # Cut the primers off each oligo, so the stats only measure what the encoder made (same as print_info)
    start: int = len(config.start_primer)
    end: int = len(config.end_primer)
    bodies: list[str] = []
    for oligo in framer.framed_oligos:
        bodies.append(oligo[start:len(oligo) - end])

    # Already have a nice stats class to figure this stuff out for us, so just use it
    stats: EncodingStats = EncodingStats(config.encoder, bodies, framer.huffman_codes, RUN_LENGTH_K)

    # Main numbers up top, one per column - all the important things a user would want to know at a glance.
    oligos_column, bits_column, gc_column, runs_column = st.columns(4)
    oligos_column.metric(
        label="Oligos",
        value=f"{config.num_oligos:,}",
    )
    bits_column.metric(
        label="Effective Bits/base",
        value=f"{original_size / stats.num_bases:.3f}",
        help="How many bits of the original file are stored in each base (primers not included). Higher is better.",
    )
    gc_column.metric(
        label="GC",
        value=f"{stats.gc_content:.1f}%",
        help="GC content (primters not included)"
    )
    runs_column.metric(
        label=f"Runs ≥ {RUN_LENGTH_K}",
        value=f"{stats.runs:,}",
        help=f"Homopolymer runs of length {RUN_LENGTH_K} or more (primers not included)",
    )

    # Now show the components and sizes in two columns, so they are side by side.
    components_column, size_column = st.columns(2)
    with components_column:
        st.markdown(
            "**Components**",
            help="The settings used to encode the file."
        )

        # Everything else, same as print_info. Each table's index is the name of the row, so it shows down the left side
        components: dict[str, str] = {
            "Encoder": config.encoder.describe(),
            "Indexer": type(config.indexer).__name__,
            "Inner correction": type(config.inner_correction).__name__ if config.inner_correction else "None",
            "Outer correction": type(config.outer_correction).__name__ if config.outer_correction else "None",
            "Oligos": f"{config.num_oligos:,} of length {config.oligo_length}",
        }
        table: pd.DataFrame = pd.DataFrame({"Setting": components.values()}, index=list(components.keys()))

        st.table(table)

    with size_column:
        # The Huffman table isn't counted in the sizes, so let the user know why
        st.markdown(
            "**Sizes**",
            help="These sizes don't include the Huffman table. The table is presumed to be amortized over a large"
                 " dataset (built once and reused), so its cost per file is small.",
        )

        # The original size is in bits, the rest are in bases
        sizes: dict[str, str] = {
            "Original size": f"{original_size:,} bits",
            "With primers": f"{total_bases:,} bases ({original_size / total_bases:.3f} bits/base)",
            "Without primers": f"{stats.num_bases:,} bases",
        }
        table = pd.DataFrame({"Size": sizes.values()}, index=list(sizes.keys()))

        st.table(table)


def show_downloads(framer: OligoFramer, json_path: Path) -> None:
    """
    Shows the download buttons for the framer JSON and the oligos text file.

    Args:
        framer: The framer after frame_all_oligos.
        json_path: The path to the JSON file with the framer.
    """
    st.subheader("Download", divider="gray")

    # Downloads right next to each other - a horizontal container puts them in a row without spreading them out like
    #   columns do. The JSON is what the decode page needs, the text file is just the oligos
    with st.container(horizontal=True):
        st.download_button(
            label="Download settings (.json)",
            help="Needed on the decode page to decode the oligos.",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            icon=":material/download:",
        )

        # One oligo per line, same format as Serializer.save_segments_to_text, so it can be uploaded on the other pages
        st.download_button(
            label="Download oligos (.txt)",
            help="The oligos, one per line. Cannot be uploaded on the decode page.",
            data="\n".join(framer.framed_oligos) + "\n",
            file_name=f"{json_path.stem.removesuffix('_framed')}_oligos.txt",   # Little hacky but atp idc
            mime="text/plain",
            icon=":material/download:",
        )



#------------------------------------------------------------------------------#
#                                                                              #
#   SHOW UI                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#

# Constants
# Encoders whose payloads are homopolymer free, these can use the normal outer codes. Every other encoder can produce
#   homopolymers, so it needs the DirectDNA outer codes. Same as test_encoders.py
HOMOPOLYMER_FREE: Final[set[type[Encoding]]] = {Goldman, Pgram, ComplementaryTree}

# Page title and info
st.title("Encoding")
# TODO: add a description of the page
st.write("TODO: add a descirption of the page")


# Input section
st.subheader("Input", divider="gray")

# Where our temp file can be found - or None if nothing valid yet
in_path: Path | None = show_input_settings()


# Settings subsection
st.subheader("Settings", divider="gray")

# Three columns - encoder and indexer on the left, the oligo in the middle, and the error correction on the right
encoder_column, oligo_column, correction_column = st.columns(3)

# First column is encoder and indexer settings (on top of each other)
encoder: Encoding
indexer: tuple[type[Indexer], int, bool]
indexer_homopolymer_free: bool
with encoder_column:
    with st.container(border=True):
        encoder = show_encoder_settings(in_path)
    with st.container(border=True):
        indexer = show_indexer_settings()
is_homopolymer_free: bool = indexer[2] and type(encoder) in HOMOPOLYMER_FREE

# Second column is the oligo settings 
oligo_length: int
start_primer: str | None
end_primer: str | None
overhead_margin: float
with oligo_column:
    with st.container(border=True):
        oligo_length, start_primer, end_primer, overhead_margin = show_oligo_settings()

# Third and last are Inner and outer codes on top of each other
inner_codes: RSInner | None
outer_codes: OuterCodes | None
with correction_column:
    with st.container(border=True):
        inner_codes = show_inner_correction_settings(indexer[0], indexer[1])
    with st.container(border=True):
        outer_codes = show_outer_correction_settings(is_homopolymer_free)


# Encode section, only enabled if there is an input and the primers are valid
st.subheader("Encode", divider="gray")

# Can only encode once there is an input and the primers are valid
ready: bool = in_path is not None and start_primer is not None and end_primer is not None
if in_path is None:
    st.info("Add an input above to encode.")

if st.button("Encode", type="primary", disabled=not ready):
    # When pressed show our little spinner so we know that we are doing something
    with st.spinner("Encoding..."):
        # Ensure we can actually encode (can't resuse ready because typing doesn't seem to actually care about that)
        if in_path is not None and start_primer is not None and end_primer is not None:
            encode(
                in_path=in_path,
                oligo_length=oligo_length,
                overhead_margin=overhead_margin,
                start_primer=start_primer,
                end_primer=end_primer,
                encoder=encoder,
                indexer=(indexer[0], indexer[1]),   # Only need the type and max run length, the framer builds its own
                inner_codes=inner_codes,
                outer_codes=outer_codes,
            )

# If There was an error show
if ResultKeys.ERROR in st.session_state:
    st.error(f"Encoding failed. {st.session_state[ResultKeys.ERROR]}")

# Everything was great - show that it succeeded and add a subsection for downloading results
if ResultKeys.FRAMER in st.session_state:
    st.success("Encoded successfully!")

    # Show results
    show_oligo_layout(st.session_state[ResultKeys.FRAMER])
    show_info(st.session_state[ResultKeys.FRAMER], st.session_state[ResultKeys.ORIGINAL_SIZE])
    show_downloads(st.session_state[ResultKeys.FRAMER], st.session_state[ResultKeys.JSON])