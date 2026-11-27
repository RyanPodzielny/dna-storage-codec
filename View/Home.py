"""
Home page for the DNA Storage Codec.
"""


from typing import Final

import streamlit as st

# Page title and info
PAGE_DESCRIPTION: Final[str] =\
"""Welcome to the DNA Storage Codec! This application allows you to encode and decode data into DNA sequences, using a 
variety of encoding schemes and error correction shcemes. It includes the full encode -> simulate -> cluster -> decode 
pipeline, as well as the ability to separetely encode, simulate, cluster, and decode data. Encoding states are saved
per session, so you can encode data, simulate errors, and decode the data all in continious browser session. You can 
also download and upload encoded data, simulation settings, and clustering settings to easily replicate your work.

A full detail of the tool is available in the thesis paper, which can be found in the PDF viewer below."""

st.title("DNA Storage Codec")
st.write(PAGE_DESCRIPTION)

# TODO: insert PDF viewer for paper 