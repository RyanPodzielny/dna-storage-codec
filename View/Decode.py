"""
Module for the Decoding page, which handles the user interface for the DNA decoding process. This includes uploading a
JSON file saved from the Encoding page, or using the oligos encoded earlier in the session if there are any.
"""

from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from Model.Encode.OligoFramer import OligoDeframer
from Model.Helpers.Errors import Error
from Model.Serialize.EncoderSerializer import Serializer
from View.Helpers import ViewHelpers
from View.Helpers.ViewHelpers import UploadKeys, file_uploader


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class Inputs(StrEnum):
    # Session state is shared across every page, so these need to be different from the Encoding page's keys
    JSON_UPLOAD = "json_upload"

    UPLOAD_FILE_NAME = "decode_uploaded_file_name"
    UPLOAD_FILE_PATH = "decode_uploaded_file_path"
    UPLOAD_ERROR = "decode_upload_error"


# Every key the file uploader keeps its upload under
UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=Inputs.JSON_UPLOAD,
    name=Inputs.UPLOAD_FILE_NAME,
    path=Inputs.UPLOAD_FILE_PATH,
    error=Inputs.UPLOAD_ERROR,
)


def _is_deframer(json_path: Path) -> bool:
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


def show_input_settings() -> Path | None:
    """
    Show the input settings to the user. An uploaded JSON file overrides the oligos encoded earlier in the session.

    Returns:
        The path to the JSON file to decode, or None if there is no valid input yet.
    """
    # Shared uploader handles storing the file and showing an error if it is not a valid deframer JSON
    uploaded_json_path: Path | None = file_uploader(
        label="Upload a JSON file saved from the Encoding page to decode:",
        keys=UPLOAD_KEYS,
        error_message="is not a valid encoded JSON file. Please upload a JSON file saved from the Encoding page.",
        file_type="json",                   # Only the JSON files saved by the encoder
        is_valid=_is_deframer,              # Only valid if it loads as a deframer
    )

    # The last run on the Encoding page, or None if the user hasn't encoded anything yet
    encoded_json_path: Path | None = st.session_state.get(ViewHelpers.ENCODED_JSON_PATH_KEY, None)

    # Uploaded file always wins - let the user know if it is overriding what they encoded
    if uploaded_json_path is not None:
        # Change to the uploaded file, and let user know
        st.success(f"File uploaded successfully! Using {st.session_state[Inputs.UPLOAD_FILE_NAME]} for decoding.")

        return uploaded_json_path

    # No upload, so use the oligos from the Encoding page - warning that uploading will override them
    if encoded_json_path is not None:
        st.warning(f"Currently using the oligos from the Encoding page ({Path(encoded_json_path).name}). Uploading a"
                   f" JSON file will override them.")

    return encoded_json_path



#------------------------------------------------------------------------------#
#                                                                              #
#   DECODING                                                                   #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class ResultKeys(StrEnum):
    # Session state is shared across every page, so these need to be different from the Encoding page's keys
    IN_PATH = "decoded_in_path"
    RECOVERED = "decode_recovered"
    MATCHES = "decode_matches"
    ERROR_INDICES = "decode_error_indices"
    ERROR = "decode_error"


def decode(in_path: Path) -> None:
    """
    Loads the deframer from the JSON file, decodes every oligo, and saves the results to session state. Any error is
    saved too, so it can be shown on the page instead of crashing it.

    Args:
        in_path: The path to the JSON file to decode.
    """
    # Forget the last run first, so an error never shows next to old results
    for key in ResultKeys:
        st.session_state.pop(key, None)

    # Remember which file these results are for, so they are hidden if the user changes the input
    st.session_state[ResultKeys.IN_PATH] = in_path

    # Try to load and decode - show big error if anything goes wrong
    try:
        deframer: OligoDeframer = Serializer.load_deframer(str(in_path))
        decoded_oligos: list[str | None] = deframer.correct_all_oligos()
        content: str = deframer.decode_to_string(decoded_oligos)

    # The file no longer matches what the deframer expects (it is checked on upload, but the encoded one is not)
    except ValidationError as error:
        st.session_state[ResultKeys.ERROR] = f"The JSON file is not valid: {error.errors()[0]['msg']}"
        return

    # Our own errors or anything else the deframer hits
    except (Error, ValueError, KeyError, TypeError) as error:
        st.session_state[ResultKeys.ERROR] = f"{type(error).__name__}: {error}"
        return

    # Every segment that could not be recovered is None, so keep where they are to show the user
    error_indices: list[int] = []
    for index, segment in enumerate(decoded_oligos):
        if segment is None:
            error_indices.append(index)

    st.session_state[ResultKeys.RECOVERED] = deframer.calculate_recovered(decoded_oligos)
    st.session_state[ResultKeys.MATCHES] = deframer.matches_original(content)
    st.session_state[ResultKeys.ERROR_INDICES] = error_indices


def show_recovered(recovered: float, matches: bool) -> None:
    """
    Shows how much of the file was recovered as a metric in the middle of the page, and whether it matches the
    original file.

    Args:
        recovered: The percentage of segments that were recovered.
        matches: Whether the decoded content matches the original file's length and hash.
    """

    # Show how much of the file was recovered - tkaes up the full width kinda looks silly but whatever
    st.metric(
        label="Recovered",
        value=f"{recovered:.2f}%",
        border=True,
    )

    # Every segment came back and the hash matches - fully decoded
    if matches:
        st.success("Decoded successfully! The decoded content matches the original file.")
    # Can be 100% recovered but still not match, if a bad strand slipped past the checksum
    else:
        st.warning("The decoded content does not match the original file.")


def show_error_indices(error_indices: list[int]) -> None:
    """
    Shows the index of every segment that could not be recovered in its own table.

    Args:
        error_indices: The index of each segment that could not be recovered.
    """
    # Just make it a new section
    st.subheader("Errors", divider="gray")
    st.write(f"{len(error_indices)} segments could not be recovered:")

    # Index column is our own, so hide pandas' one
    st.dataframe(
        pd.DataFrame({"Index": error_indices}),
        hide_index=True,
        width="stretch",                    # Take up the full width of the page
    )



#------------------------------------------------------------------------------#
#                                                                              #
#   SHOW UI                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



# Page title and info
st.title("Decoding")
# TODO: add a description of the page
st.write("TODO: add a descirption of the page")


# Input section
st.subheader("Input", divider="gray")

# Where the JSON to decode can be found - or None if nothing valid yet
in_path: Path | None = show_input_settings()


# Decode section, only enabled if there is an input
st.subheader("Decode", divider="gray")

if in_path is None:
    st.info("Upload a JSON file above, or encode something on the Encoding page, to decode.")

# Only show decode button if we have valid input
if st.button("Decode", type="primary", disabled=in_path is None):
    # When pressed show our little spinner so we know that we are doing something
    with st.spinner("Decoding..."):
        # Ensure we can actually decode (can't reuse the disabled check - i hate typing)
        if in_path is not None:
            decode(in_path)

# Only show results for the input currently picked, so old results don't show next to a new file
if st.session_state.get(ResultKeys.IN_PATH, None) == in_path:
    # If there was an error show
    if ResultKeys.ERROR in st.session_state:
        st.error(f"Decoding failed. {st.session_state[ResultKeys.ERROR]}")

    # Decoded - show how much we got back, and which segments we lost if not all of them
    if ResultKeys.RECOVERED in st.session_state:
        show_recovered(st.session_state[ResultKeys.RECOVERED], st.session_state[ResultKeys.MATCHES])

        # Didn't recovre all segments - show which ones we lost
        if st.session_state[ResultKeys.ERROR_INDICES]:
            show_error_indices(st.session_state[ResultKeys.ERROR_INDICES])
