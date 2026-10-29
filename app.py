import streamlit as st

st.set_page_config(
    page_title="FrameSeek",
    page_icon="🎬",
    layout="wide",
)

navigation = st.navigation([
    st.Page(
        "unified_search.py",
        title="Combined Search",
        icon="🔎",
        default=True,
    ),
    st.Page(
        "visual_app.py",
        title="Visual & Text Search",
        icon="🖼️",
    ),
    st.Page(
        "speech_app.py",
        title="Speech Search",
        icon="🎙️",
    ),
])

navigation.run()