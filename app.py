import streamlit as st

# Pages to navigate to 
pages = st.navigation(
    [
        # Home page is default, so it will be the first page to load - probably just gonna be pdf view of the paper
        st.Page("View/Home.py", title="Home", icon=":material/home:", default=True),

        st.Page("View/Encode.py", title="Encoding", icon=":material/genetics:"),
        st.Page("View/Simulate.py", title="Simulate Errors", icon=":material/instant_mix:"),
        st.Page("View/Convergence.py", title="Clustering and Consensus", icon=":material/hub:"),
        st.Page("View/Decode.py", title="Decoding", icon=":material/fact_check:"),
    ]
)

# Make it so we can see the pages of the steps and make it wide so elements fill screen
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

pages.run()