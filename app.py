import streamlit as st

# Pages to navigate to 
pages = st.navigation(
    [
        # Home page is default, so it will be the first page to load
        st.Page("Pages/Home.py", title="Home", icon=":material/home:", default=True),

        st.Page("Pages/Encode.py", title="Encoding", icon=":material/genetics:"),
        st.Page("Pages/Simulate.py", title="Simulate Errors", icon=":material/instant_mix:"),
        st.Page("Pages/Convergence.py", title="Clustering and Consensus", icon=":material/hub:"),
        st.Page("Pages/Decode.py", title="Decoding", icon=":material/fact_check:"),
        st.Page("Pages/Pipeline.py", title="End-to-End Pipeline", icon=":material/route:"),
    ]
)


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

pages.run()