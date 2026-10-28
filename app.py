import cProfile
import io
import pstats
import time

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
    diagnose = st.checkbox("Measure page speed", value=True)

if diagnose:
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()

    try:
        navigation.run()
    finally:
        profiler.disable()
        elapsed = time.perf_counter() - start

        report = io.StringIO()
        stats = pstats.Stats(profiler, stream=report)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)

        with st.sidebar:
            st.metric("Python page execution", f"{elapsed:.1f} seconds")

            with st.expander("Timing details", expanded=True):
                st.code(report.getvalue(), language="text")
else:
    navigation.run()