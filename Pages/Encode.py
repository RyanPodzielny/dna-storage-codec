from enum import StrEnum
from pathlib import Path
from tempfile import mkdtemp
from typing import Final

from matplotlib.widgets import Slider
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from Encode.Methods.DNAMapping import DNAMap, DNAMapHuffman
from Encode.Methods.Encoding import Encoding
from Encode.Methods.HuffmanCoding import BitScanQuaternary, ComplementaryTree, Goldman, Pgram, Quaternary, Ternary


#------------------------------------------------------------------------------#
#                                                                              #
#   PAGE INFO                                                                  #
#                                                                              #
#------------------------------------------------------------------------------#



st.title("Encoding")
st.write("TODO: add a descirption of the page")



#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Section heading for everything to do with picking our input
st.subheader("Input", divider="gray")


# Constants
class Inputs(StrEnum):
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


def file_uploader() -> Path | None:
    """
    Upload a text-based file and return the path to a temporary file containing its contents. The file is kept in
    session state, so it is still used after switching to text input and back.

    Returns:
        The path to the temporary file containing the uploaded file's contents, or None if there is no valid file.
    """
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
    

def text_area_input() -> Path | None:
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


def input_settings() -> Path | None:

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
        in_path = file_uploader()
    # User wants a text box
    elif input_method == Inputs.TEXT_INPUT:
        in_path = text_area_input()

    return in_path


# Where our temp file can be found - or None if nothing valid yet
in_path: Path | None = input_settings()



#------------------------------------------------------------------------------#
#                                                                              #
#   ENCODER SETTINGS                                                           #
#                                                                              #
#------------------------------------------------------------------------------#



# Section for everything to do with the settings
st.subheader("Settings", divider="gray")


# Constants for the encoder settings - the options in the drop down, and the session state keys for their arguments
class Encoders(StrEnum):
    DNA_MAP = "DNA Map"
    DNA_MAP_HUFFMAN = "DNA Map + Huffman"

    QUATERNARY = "Quaternary Huffman"
    BITSCAN_QUATERNARY = "Bit-Scan Quaternary Huffman"

    TERNARY = "Ternary Huffman"
    GOLDMAN = "Goldman (Rotating Tree)"

    PGRAM = "P-gram Huffman"
    COMPLEMENTARY_TREE = "Complementary Tree"

    BITSCAN_K = "bitscan_k"
    PGRAM_M = "pgram_m"


# Plain Huffman is left out as it outputs bits, not bases, so it can't be framed into oligos
ENCODER_OPTIONS: Final[list[Encoders]] = [
    Encoders.DNA_MAP,
    Encoders.DNA_MAP_HUFFMAN,

    Encoders.QUATERNARY,
    Encoders.BITSCAN_QUATERNARY,

    Encoders.TERNARY,
    Encoders.GOLDMAN,

    Encoders.PGRAM,
    Encoders.COMPLEMENTARY_TREE,    
]


# A short description of each encoder, shown under the drop down so the user knows what they picked
ENCODER_DESCRIPTIONS: Final[dict[Encoders, str]] = {
    Encoders.DNA_MAP: "Direct mapping, 00 → A, 01 → C, 10 → G, 11 → T. This method has no compression.",
    Encoders.DNA_MAP_HUFFMAN: "Binary Huffman compression first, then the direct 2 bits per base mapping.",

    Encoders.QUATERNARY: "Huffman with all four bases (A, C, G, T), each base holds up to 2 bits. Essentially works"
                         " like a binary Huffman but with a larger alphabet - so four children per node with their"
                         " respective DNA bases.",
    Encoders.BITSCAN_QUATERNARY: "Quaternary Huffman that scans k bits at a time, larger k packs better but the table"
                                 " grows quickly. It uses a quaternary tree like the one described in the Quaternary"
                                 " encoding method.",

    Encoders.TERNARY: "Huffman with three bases per branch (A, C, G), a child never repeats its parent's base.",
    Encoders.GOLDMAN: "Rotating tree Huffman. First the input is converted into trits using a Huffman tree with three"
                      " children per node. Then each trit is converted to a base using a rotating map, where the base"
                      " chosen depends on the previous base - so the three options are always the bases that are"
                      " different from the last one.",

    Encoders.PGRAM: "Huffman over variable length substrings (p-grams) up to length m. Larger m can compress better but"
                    " the table grows quickly - however this should be amortized over a large dataset and frozen."
                    " The Huffman codes are built over the complementary tree, a binary tree where even levels use"
                    " G and T and odd levels use A and C, and each code word is complemented when needed so the"
                    " alternation carries on across code words.",
    Encoders.COMPLEMENTARY_TREE: "The Huffman codes are built over the complementary tree, a binary tree where even"
                                 " levels use G and T and odd levels use A and C, and each code word is complemented"
                                 "  when needed so the alternation carries on across code words.",
}

PRODUCES_HOMOPOLYMERS: str = "This method produces homopolymers."
PREVENTS_HOMOPOLYMERS: str = "This method does not produce homopolymers."
# Whether each encoder produces homopolymers, kept apart from the description so it can be shown in bold
HOMOPOLYMER_NOTES: Final[dict[Encoders, str]] = {
    Encoders.DNA_MAP: PRODUCES_HOMOPOLYMERS,
    Encoders.DNA_MAP_HUFFMAN: PRODUCES_HOMOPOLYMERS,

    Encoders.QUATERNARY: PRODUCES_HOMOPOLYMERS,
    Encoders.BITSCAN_QUATERNARY: PRODUCES_HOMOPOLYMERS,

    Encoders.TERNARY: PREVENTS_HOMOPOLYMERS[:-1] + " of size >= 2.",
    Encoders.GOLDMAN: PREVENTS_HOMOPOLYMERS,

    Encoders.PGRAM: PREVENTS_HOMOPOLYMERS,
    Encoders.COMPLEMENTARY_TREE: PREVENTS_HOMOPOLYMERS,
}

# Ranges for the encoder arguments. k matches the measured bases per byte table (estimating needs an entry for k), m
#   past 16 is allowed by Pgram but the estimate is capped at 16, so no point going higher here.
BITSCAN_K_RANGE: Final[tuple[int, int]] = (min(BitScanQuaternary.BITSCAN_BASES_PER_BYTE),
                                           max(BitScanQuaternary.BITSCAN_BASES_PER_BYTE))
PGRAM_M_RANGE: Final[tuple[int, int]] = (min(Pgram.PGRAM_BASES_PER_BYTE), max(Pgram.PGRAM_BASES_PER_BYTE))
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
    optimal_k: int = _find_optimal_k(str(in_path))
    st.session_state[f"{Encoders.BITSCAN_K}_slider"] = optimal_k
    st.session_state[f"{Encoders.BITSCAN_K}_number"] = optimal_k


def encoder_settings(in_path: Path | None) -> Encoding:
    """
    Drop down to pick the encoder, plus the extra arguments for the encoders that have any.

    Args:
        in_path: The path to the input, used to find the best k for bit-scan. None if there is no input yet.

    Returns:
        The encoder built with the picked settings.
    """
    st.markdown("**Encoder**")

    choice: Encoders = st.selectbox(
        label="Encoding method:",
        options=ENCODER_OPTIONS,
        key="encoder_choice",
        persist_state="session",
    )
    # Captions take markdown, so ** makes the homopolymer note bold
    st.caption(f"{ENCODER_DESCRIPTIONS[choice]} **{HOMOPOLYMER_NOTES[choice]}**")

    # Only bit-scan and p-gram take any arguments, every other encoder is built as is
    match choice:
        case Encoders.BITSCAN_QUATERNARY:
            # Slider for the bit scan
            k: int = _slider_with_input(
                label="Bits scanned at a time (k):",
                key=Encoders.BITSCAN_K,
                min_value=BITSCAN_K_RANGE[0],
                max_value=BITSCAN_K_RANGE[1],
                default=BITSCAN_K_DEFAULT,
                help="How many bits are read as one symbol. Larger k can compress better but makes the Huffman table"
                     " much bigger.",
            )

            # Want to show a button to find best k - but only if there is an input
            st.button(
                "Find best k for my input",
                on_click=_set_optimal_k,
                args=(in_path,),
                disabled=in_path is None,           # Disable if there is no input yet
                help="Tries every k on your input and picks the one with the best coding density.",
            )
            return BitScanQuaternary(k=k)

        case Encoders.PGRAM:
            # Slider for the p-gram length
            m: int = _slider_with_input(
                label="Maximum p-gram length (m):",
                key=Encoders.PGRAM_M,
                min_value=PGRAM_M_RANGE[0],
                max_value=PGRAM_M_RANGE[1],
                default=PGRAM_M_DEFAULT,
                help="The longest substring that can get its own Huffman code.",
            )
            return Pgram(m=m)

        # Rest of the encoders have no arguments, so just return as is
        case Encoders.DNA_MAP:
            return DNAMap()
        case Encoders.DNA_MAP_HUFFMAN:
            return DNAMapHuffman()
        case Encoders.QUATERNARY:
            return Quaternary()
        case Encoders.TERNARY:
            return Ternary()
        case Encoders.GOLDMAN:
            return Goldman()
        case Encoders.COMPLEMENTARY_TREE:
            return ComplementaryTree()
        case _:
            raise ValueError(f"Unknown encoder: {choice}")


# Left column is the encoder, right column is two settings stacked on top of each other
encoder_column, other_column = st.columns(2)

with encoder_column:
    with st.container(border=True):
        encoder: Encoding = encoder_settings(in_path)

with other_column:
    with st.container(border=True):
        st.write("Top right settings - probably gonna be index and error correction settings")

    with st.container(border=True):
        st.write("Bottom right settings - probably gonna be oligo framing settings (so the oligo length and primers)")
