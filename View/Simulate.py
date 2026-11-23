"""
"""

from enum import StrEnum
from pathlib import Path
from typing_extensions import Final

import streamlit as st

from Model.Simulate.ErrorConfig import AgingConfig, PCRConfig, SequencingConfig, SynthesisConfig
from View.Helpers.ViewHelpers import UploadKeys, show_input_settings


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
#   SIMULATOR SETTINGS                                                         #
#                                                                              #
#------------------------------------------------------------------------------#



# Defailts
SYNTEHSIS_DEFAULT: Final[SynthesisConfig] = SynthesisConfig()
AGING_DEFAULT: Final[AgingConfig] = AgingConfig()
PCR_DEFAULT: Final[PCRConfig] = PCRConfig()
SEQUENCING_DEFAULT: Final[SequencingConfig] = SequencingConfig()  



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
in_path, is_json = show_input_settings("simulator_input_method", JSON_UPLOAD_KEYS, TEXT_UPLOAD_KEYS)
