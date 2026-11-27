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
from View.Helpers.ViewHelpers import UploadKeys, file_uploader, is_deframer


#------------------------------------------------------------------------------#
#                                                                              #
#   INPUT                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class DecodeInputs(StrEnum):
    # Session state is shared across every page, so these need to be different from the Encoding page's keys
    UPLOAD_WIDGET = "decode_json_upload"
    UPLOAD_FILE_NAME = "decode_uploaded_file_name"
    UPLOAD_FILE_PATH = "decode_uploaded_file_path"
    UPLOAD_ERROR = "decode_upload_error"


# Every key the file uploader keeps its upload under
UPLOAD_KEYS: Final[UploadKeys] = UploadKeys(
    widget=DecodeInputs.UPLOAD_WIDGET,
    name=DecodeInputs.UPLOAD_FILE_NAME,
    path=DecodeInputs.UPLOAD_FILE_PATH,
    error=DecodeInputs.UPLOAD_ERROR,
)


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
        is_valid=is_deframer,               # Only valid if it loads as a deframer
    )

    # The last run on the Encoding page, or None if the user hasn't encoded anything yet
    encoded_json_path: Path | None = st.session_state.get(ViewHelpers.ENCODED_JSON_PATH_KEY, None)

    # Uploaded file always wins - let the user know if it is overriding what they encoded
    if uploaded_json_path is not None:
        # Change to the uploaded file, and let user know
        st.success(f"File uploaded successfully! Using {st.session_state[DecodeInputs.UPLOAD_FILE_NAME]} for decoding.")

        return uploaded_json_path

    # No upload, so use the latest oligos from the session (Encoding page, or the consensus strands from clustering) -
    #   warning that uploading will override them
    if encoded_json_path is not None:
        st.warning(f"Currently using the latest oligos from this session ({Path(encoded_json_path).name}). Uploading a"
                   f" JSON file will override them.")

    return encoded_json_path



#------------------------------------------------------------------------------#
#                                                                              #
#   DECODING                                                                   #
#                                                                              #
#------------------------------------------------------------------------------#



# Constants
class DecodeResultKeys(StrEnum):
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
    for key in DecodeResultKeys:
        st.session_state.pop(key, None)

    # Remember which file these results are for, so they are hidden if the user changes the input
    st.session_state[DecodeResultKeys.IN_PATH] = in_path

    # Try to load and decode - show big error if anything goes wrong
    try:
        deframer: OligoDeframer = Serializer.load_deframer(str(in_path))
        decoded_oligos: list[str | None] = deframer.correct_all_oligos()
        content: str = deframer.decode_to_string(decoded_oligos)

    # The file no longer matches what the deframer expects (it is checked on upload, but the encoded one is not)
    except ValidationError as error:
        st.session_state[DecodeResultKeys.ERROR] = f"The JSON file is not valid: {error.errors()[0]['msg']}"
        return

    # Our own errors or anything else the deframer hits
    except (Error, ValueError, KeyError, TypeError) as error:
        st.session_state[DecodeResultKeys.ERROR] = f"{type(error).__name__}: {error}"
        return

    # Every segment that could not be recovered is None, so keep where they are to show the user
    error_indices: list[int] = []
    for index, segment in enumerate(decoded_oligos):
        if segment is None:
            error_indices.append(index)

    # Save the results to session state so they can be shown on the page
    st.session_state[DecodeResultKeys.RECOVERED] = deframer.calculate_recovered(decoded_oligos)
    st.session_state[DecodeResultKeys.MATCHES] = deframer.matches_original(content)
    st.session_state[DecodeResultKeys.ERROR_INDICES] = error_indices


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
    st.write(f"{len(error_indices)} segments could not be recovered")

    # Use an expander like the simulator does
    with st.expander("Expand Errors"):
        # Index column is our own, so hide pandas' one
        data: pd.DataFrame = pd.DataFrame({"Index": error_indices})
        st.dataframe(
            data=data,
            hide_index=True,
            width="stretch",                    # Take up the full width of the page
        )



#------------------------------------------------------------------------------#
#                                                                              #
#   SHOW UI                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#



# Page title and info
PAGE_DESCRIPTION: Final[str] =\
"""Decode a JSON file saved from the Encoding page, or the oligos encoded earlier in the session. The decoded content is 
compared to the original file's length and hash, and a metric is shown for how much of the file was recovered. If any 
segments could not be recovered, their indices are shown in a table below. It is required for the user to upload or use 
the oligos from the Encoding page, as the decoder cannot know how to decode without the encoding scheme."""

st.title("Decoding")
st.write(PAGE_DESCRIPTION)


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
if st.session_state.get(DecodeResultKeys.IN_PATH, None) == in_path:
    # If there was an error show
    if DecodeResultKeys.ERROR in st.session_state:
        st.error(f"Decoding failed. {st.session_state[DecodeResultKeys.ERROR]}")

    # Decoded - show how much we got back, and which segments we lost if not all of them
    if DecodeResultKeys.RECOVERED in st.session_state:
        show_recovered(st.session_state[DecodeResultKeys.RECOVERED], st.session_state[DecodeResultKeys.MATCHES])

        # Didn't recovre all segments - show which ones we lost
        if DecodeResultKeys.ERROR_INDICES in st.session_state and\
            len(st.session_state[DecodeResultKeys.ERROR_INDICES]) > 0:
            
            show_error_indices(st.session_state[DecodeResultKeys.ERROR_INDICES])