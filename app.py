import streamlit as st

st.set_page_config(
    page_title="FrameSeek",
    page_icon="🎬",
    layout="wide",
)

navigation = st.navigation([
    st.Page(
        "visual_app.py",
        title="Visual & Text Search",
        icon="🖼️",
        default=True,
    ),
    st.Page(
        "speech_app.py",
        title="Speech Search",
        icon="🎙️",
    ),
])

with st.sidebar:
    st.caption(
        "Visual search finds similar scenes. "
        "Text search finds words on screen. "
        "Speech search finds spoken words."
    )

navigation.run()