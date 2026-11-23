"""
Module for shared UI elements, like a slider with a number input beside it or a file uploader that keeps its file in
session state. These are used in multiple places, so they are kept here to avoid duplication.
"""


from enum import StrEnum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, TypeVar

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import BaseModel, ValidationError

from Model.Encode.Methods.Encoding import Encoding
from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer


#------------------------------------------------------------------------------#
#                                                                              #
#   CONSTANTS                                                                  #
#                                                                              #
#------------------------------------------------------------------------------#



# Key so we can run encoding process at each page without having to re-upload. Basically just a session state to store
#   path to the newly encoded JSON file, where we can use in simulation, clustering and decoding.
ENCODED_JSON_PATH_KEY: str = "encoded_json_path"

# A slider with an input works on either ints or floats, but every number given to one has to be the same type. Would 
#   be nice if we could jsut do type Number = int | float, but typing check throws a fit since it doesnt implicit cast.
#   Kinda stupid so we are stuck with TypeVars
Number = TypeVar("Number", int, float)


class HelperInputs(StrEnum):
    # Need to make these keys as Streamlit doesn't keep the text_input box written, when user switches. Just easier.
    JSON_UPLOAD = "JSON Upload"
    TEXT_UPLOAD = "Raw Oligo Upload"



# Just using pydantic instead of dataclass to keep it consistent with everything else
class UploadKeys(BaseModel):
    """
    The session state keys for one file uploader. Session state is shared across every page, so each uploader needs
    its own keys or they would overwrite each other. Basically we just want to store name and path of each uploader's 
    file so we can use it throughout the pages. We use it for everything that needs a file (which is a lot, literally
    used on every page), so easier than just duplicating the code for each page.
    """
    widget: str     # The file uploader widget itself
    name: str       # Name of the uploaded file
    path: str       # Path to our temporary copy of the uploaded file
    error: str      # Name of the last file that could not be used



#------------------------------------------------------------------------------#
#                                                                              #
#   PRIVATE HELPER FUNCTIONS                                                   #
#                                                                              #
#------------------------------------------------------------------------------#



def _framer_file_uploader(json_keys: UploadKeys) -> Path | None:
    """
    Upload a JSON file saved from the Encoding page. The file is kept in session state, so it is still used after
    switching to text upload and back.

    Returns:
        The path to the temporary file containing the uploaded file's contents, or None if there is no valid file.
    """
    uploaded_json_path: Path | None = file_uploader(
        label="Upload a JSON file saved from the Encoding page:",
        keys=json_keys,
        error_message="is not a valid encoded JSON file. Please upload a JSON file saved from the Encoding page.",
        file_type="json",                   # Only the JSON files saved by the encoder
        is_valid=is_deframer,               # Only valid if it loads as a deframer
    )
    return uploaded_json_path


def _text_file_uploader(text_keys: UploadKeys) -> Path | None:
    """
    Upload a text file of oligos, one per line. Same as the JSON uploader, the file is kept in session state.

    Returns:
        The path to the temporary file containing the uploaded file's contents, or None if there is no valid file.
    """
    uploaded_text_path: Path | None = file_uploader(
        label="Upload a text file containing oligos separated by newlines:",
        keys=text_keys,
        error_message="is not a valid encoded oligo file. Please upload a file that contains only nucleotides.",
        file_type=["txt", "text"],          # Same format as the oligos download on the Encoding page
        is_valid=text_is_oligos,            # Only valid if every line is only bases
    )
    return uploaded_text_path



#------------------------------------------------------------------------------#
#                                                                              #
#   PUBLIC HELPER FUNCTIONS                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



def clean_primer(primer: str) -> str | None:
    """
    Cleans up a primer the user typed, so lowercase and spaces are fine.

    Args:
        primer: The primer as typed.

    Returns:
        The primer in uppercase with no spaces, or None if it has anything other than A, C, G, and T.
    """
    primer = primer.replace(" ", "").upper()

    # Only bases allowed - an empty primer is fine, it just means no primer
    if not set(primer) <= Encoding.DNA_BASES:
        return None

    return primer


def text_is_oligos(in_path: Path) -> bool:
    """
    Check the text is a list of oligos, one per line. Each line must be ACGT only, and at least 1 character long.

    Args:
        in_path: The path to the file to check.

    Returns:
        True if the file contains a list of valid oligos, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8 text.
    """
    with open(in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines, as they are not oligos but we don't want to fail on them
            if not line:
                continue

            # Check the line is ACGT only
            if not all(base in Encoding.DNA_BASES for base in line):
                return False

    return True


def is_deframer(json_path: Path) -> bool:
    """
    Check the uploaded JSON matches what the deframer expects. Runs once in the uploader's callback, not on every rerun.

    Args:
        json_path: The path to our copy of the uploaded JSON.

    Returns:
        True if the file loads as a deframer, False if not.
    """
    # Only need to know it loads, the deframer itself is made again when we decode
    try:
        Serializer.load_deframer(str(json_path))
    except (ValidationError, Error, KeyError, TypeError):
        return False

    return True


def contents_to_file(contents: str, filename: str) -> Path:
    """
    Write the contents to a temporary file and return the path to that file.

    Args:
        contents: The text to write to the file.
        filename: The name to give the temporary file.

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


def clear_upload(keys: UploadKeys) -> None:
    """
    Forget the uploaded file, so nothing from the file upload is used.

    Args:
        keys: The session state keys of the uploader to clear.
    """
    st.session_state.pop(keys.name, None)
    st.session_state.pop(keys.path, None)
    st.session_state.pop(keys.error, None)


def store_upload(keys: UploadKeys,
                 is_valid: Callable[[Path], bool] | None = None,
                 on_upload: Callable[[], None] | None = None) -> None:
    """
    Callback for a file uploader, runs only when the user uploads a new file or removes one. Streamlit seems to just
    drop the uploaded file when the uploader isn't drawn (switching input method or page), so we keep our own copy.
    Using a callback means we can tell user removed the file apart from uploader was hidden and came back empty.

    Args:
        keys: The session state keys of the uploader.
        is_valid: Optional extra check on our copy of the file (on top of it being text), e.g. that it loads.
        on_upload: Optional function to run once a valid file is stored, e.g. to clear another input.
    """
    # Remove any previous error or file, and get the current upload
    upload: UploadedFile | None = st.session_state[keys.widget]
    st.session_state.pop(keys.error, None)

    # User removed the file, so just clear it
    if upload is None:
        clear_upload(keys)
        return

    # Ensure we can read the file as text, and handle any errors - letting users know if the file is not valid text
    file_contents: str
    try:
        file_contents = upload.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        # Can't show the error from inside a callback (it would draw above everything), so flag it for the page
        clear_upload(keys)
        st.session_state[keys.error] = upload.name
        return

    # Convert to a temporary file to be used by the backend - done once here, not on every rerun
    file_path: Path = contents_to_file(file_contents, upload.name)

    # Page specific check, flagged the same way as not being text
    if is_valid is not None and not is_valid(file_path):
        clear_upload(keys)
        st.session_state[keys.error] = upload.name
        return

    # Let the page react to a new valid file (e.g. replace other inputs)
    if on_upload is not None:
        on_upload()

    # Store the file name and path in session state, so we can use it throughout the pages
    st.session_state[keys.name] = upload.name
    st.session_state[keys.path] = file_path


def sync_widget(source_key: str, target_key: str) -> None:
    """
    Callback to copy one widget's value into another, so a slider and number input always show the same thing.

    Args:
        source_key: The key of the widget the user just changed.
        target_key: The key of the widget to update.
    """
    st.session_state[target_key] = st.session_state[source_key]



#------------------------------------------------------------------------------#
#                                                                              #
#   UI HELPER FUNCTIONS                                                        #
#                                                                              #
#------------------------------------------------------------------------------#



def file_uploader(label: str,
                  keys: UploadKeys,
                  error_message: str,
                  file_type: str | list[str] | None = None,
                  is_valid: Callable[[Path], bool] | None = None,
                  on_upload: Callable[[], None] | None = None) -> Path | None:
    """
    A file uploader that keeps the upload in session state, so it is still used after it is hidden and drawn again.
    Shows a remove button if the box came back empty but the file is still used, and an error if the last file picked
    could not be used.

    Args:
        label: The label shown above the uploader.
        keys: The session state keys of the uploader, must be different for every uploader.
        error_message: Shown after the file name when a file can't be used, e.g. "is not a valid text file."
        file_type: The file extensions allowed, or None to allow any.
        is_valid: Optional extra check on our copy of the file (on top of it being text), e.g. that it loads.
        on_upload: Optional function to run once a valid file is stored, e.g. to clear another input.

    Returns:
        The path to the temporary file containing the uploaded file's contents, or None if there is no valid file.
    """
    # Actual file upload button for user to click
    st.file_uploader(
        label=label,
        key=keys.widget,
        type=file_type,
        max_upload_size=None,               # No limit on file size - probs a bad idea but thats on the user
        accept_multiple_files=False,        # Only one at a time
        on_change=store_upload,             # Save (or forget) the file only when the user changes it
        args=(keys, is_valid, on_upload),
    )

    # Streamlit empties the uploader when it isn't drawn (switching page or input method) and won't let us put the file
    #   back, so the box has no X to remove our copy with. Give our own remove button, only when the box is empty but
    #   we are still using a file - otherwise the box's own X does the same thing. Will also remove this button if 
    #   pressed.
    if st.session_state.get(keys.widget, None) is None and keys.path in st.session_state:
        st.button(
            label=f"Remove {st.session_state[keys.name]}",
            key=f"{keys.widget}_remove",
            icon=":material/close:",
            help="Remove the file so it is no longer used.",
            on_click=clear_upload,          # Forget our copy, same as the box's X
            args=(keys,),
        )

    # The last file the user picked could not be used
    if keys.error in st.session_state:
        st.error(f"{st.session_state[keys.error]} {error_message}")
        return None

    # Path to our copy, or None if nothing is uploaded yet (or it was removed)
    return st.session_state.get(keys.path, None)


def slider_with_input(label: str,
                      key: str,
                      min_value: Number,
                      max_value: Number,
                      default: Number,
                      help: str,
                      step: Number | None = None,
                      disabled: bool = False) -> Number:
    """
    A slider with a number input beside it, kept in sync. The slider is quick to drag, the number input is easier for
    an exact value. Both keep their value for the whole session, so switching encoders and back doesn't reset them.
    Works with ints or floats (e.g. a probability), as long as every number given is the same type.

    Args:
        label: The label shown above the slider.
        key: The base key, the slider and number input use this with "_slider" and "_number" added.
        min_value: The lowest value allowed.
        max_value: The highest value allowed.
        default: The value used before the user changes anything.
        help: The tooltip shown beside the label.
        step: How much the value moves per step, None for Streamlit's default (1 for ints, 0.01 for floats).
        disabled: Whether both widgets are greyed out, e.g. when the setting isn't used.

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
    slider_column: Any
    number_column: Any
    slider_column, number_column = st.columns([3, 1], vertical_alignment="bottom")

    # Slider is the main widget, so it has the label and help. 
    with slider_column:
        value: Number = st.slider(
            label=label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            help=help,
            disabled=disabled,
            on_change=sync_widget,              # Ensure we keep keep the number we saved
            args=(slider_key, number_key),
            persist_state="session",            # Same here - keep the value for the whole session
        )

    # The number input is hidden, as the slider already has the label. It is only there to let the user type in a value
    with number_column:
        st.number_input(
            label=label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_key,
            disabled=disabled,
            on_change=sync_widget,            # Ensure we keep the slider we saved
            args=(number_key, slider_key),
            label_visibility="collapsed",      # Slider already has the label
            persist_state="session",           # Keep the value for the whole session
        )

    return value


def show_input_settings(control_key: str, json_keys: UploadKeys, text_keys: UploadKeys) -> tuple[Path | None, bool]:
    """
    Show the input settings to the user. An uploaded file overrides the oligos encoded earlier in the session.

    Arsgs:
        control_key: The key for the segmented control, so it keeps its value when the page is opened again.
        json_keys: The session state keys of the JSON uploader.
        text_keys: The session state keys of the text uploader.

    Returns:
        The path to the file to use (None if there is no valid input yet), and whether it is a JSON file saved
        from the Encoding page (False if it is a text file of oligos).
    """
    # Now show to the user the option to choose between JSON upload and text upload
    input_method = st.segmented_control(
        label="Choose an input method:",
        options=[HelperInputs.JSON_UPLOAD, HelperInputs.TEXT_UPLOAD],
        default=HelperInputs.JSON_UPLOAD,                # Default to JSON upload
        required=True,                      # Must be either one
        key=control_key,
        persist_state="session",            # Keep the picked method when the page is opened again
    )

    # What to show based on the input method selected. Either way we end up with a path to cluster, or None if the user
    #   has not given us anything valid yet.
    in_path: Path | None = None
    keys: UploadKeys = json_keys
    # User wants to upload a JSON file
    if input_method == HelperInputs.JSON_UPLOAD:
        in_path = _framer_file_uploader(json_keys)
    # User wants to upload a text file
    elif input_method == HelperInputs.TEXT_UPLOAD:
        in_path = _text_file_uploader(text_keys)
        keys = text_keys

    # The last run on the Encoding page, or None if the user hasn't encoded anything yet
    encoded_json_path: Path | None = st.session_state.get(ENCODED_JSON_PATH_KEY, None)

    # Uploaded file always wins - let the user know if it is overriding what they encoded
    if in_path is not None:
        # Change to the uploaded file, and let user know
        st.success(f"File uploaded successfully! Using {st.session_state[keys.name]} for clustering and consensus.")
        return in_path, input_method == HelperInputs.JSON_UPLOAD

    # No upload, so use the latest oligos from the session (Encoding page, or the last consensus run) - warning that
    #   uploading will override them (the uploader already shows an error if the last file was not valid, so no need to
    #   show this too)
    if encoded_json_path is not None and keys.error not in st.session_state:
        st.warning(f"Currently using the latest oligos from this session ({Path(encoded_json_path).name}). Uploading a"
                   f" file will override them.")

    # Encoding page always saves a JSON
    return encoded_json_path, True



def show_downloads(in_path: Path,
                   strands: list[str], 
                   json_path: Path | None, 
                   json_help: str, 
                   strand_help: str, 
                   suffix: str,
                   error_path: Path | None = None,
                   error_help: str | None = None) -> None:
    """
    Shows the download buttons for the consensus strands as a text file, and the JSON with them in it if the input
    was a JSON file.

    Args:
        in_path: The path to the input, used to name the downloads.
        strands: The consensus strand of every cluster.
        json_path: The path to the JSON file with the consensus strands, or None if the input was a text file.
        json_help: The help text for the JSON download button.
        strand_help: The help text for the strands download button.
        error_path: The path to the JSON file with the errors, or None if there are no errors.
        error_help: The help text for the errors download button.
    """
    st.subheader("Download", divider="gray")

    # Downloads right next to each other - a horizontal container puts them in a row without spreading them out like
    #   columns do
    with st.container(horizontal=True):
        # Only a JSON input has settings to put the strands back in to
        if json_path is not None:
            st.download_button(
                label="Download settings (.json)",
                help=json_help,
                data=json_path.read_bytes(),
                file_name=json_path.name,
                mime="application/json",
                icon=":material/download:",
            )

        # One strand per line, same format as Serializer.save_segments_to_text, so it can be uploaded on the other pages
        st.download_button(
            label="Download strands (.txt)",
            help=strand_help,
            data="\n".join(strands) + "\n",
            file_name=f"{in_path.stem}-{suffix}.txt",
            mime="text/plain",
            icon=":material/download:",
        )

        if error_path is not None:
            st.download_button(
                label="Download errors (.json)",
                help=error_help,
                data=error_path.read_bytes(),
                file_name=error_path.name,
                mime="application/json",
                icon=":material/download:",
            )