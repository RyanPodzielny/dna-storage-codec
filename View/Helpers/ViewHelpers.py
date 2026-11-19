"""
Module for shared UI elements, like a slider with a number input beside it or a file uploader that keeps its file in
session state. These are used in multiple places, so they are kept here to avoid duplication.
"""


from pydantic import BaseModel
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


#------------------------------------------------------------------------------#
#                                                                              #
#   CONSTANTS                                                                  #
#                                                                              #
#------------------------------------------------------------------------------#



# Key so we can run encoding process at each page without having to re-upload. Basically just a session state to store
#   path to the newly encoded JSON file, where we can use in simulation, clustering and decoding.
ENCODED_JSON_PATH_KEY: str = "encoded_json_path"


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
#   HELPER FUNCTIONS                                                           #
#                                                                              #
#------------------------------------------------------------------------------#



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


def sync_widget(source_key: str, target_key: str) -> None:
    """
    Callback to copy one widget's value into another, so a slider and number input always show the same thing.

    Args:
        source_key: The key of the widget the user just changed.
        target_key: The key of the widget to update.
    """
    st.session_state[target_key] = st.session_state[source_key]


def slider_with_input(label: str, key: str, min_value: int, max_value: int, default: int, help: str) -> int:
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
            step=1,
            key=number_key,
            on_change=sync_widget,            # Ensure we keep the slider we saved
            args=(number_key, slider_key),
            label_visibility="collapsed",      # Slider already has the label
            persist_state="session",           # Keep the value for the whole session
        )

    return value