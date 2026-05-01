import streamlit as st


def run_app(router):
    st.title("TBRGS Route System")

    origin = st.text_input("Origin")
    destination = st.text_input("Destination")
    time = st.text_input("Time (e.g. 08:00)")

    if st.button("Find Routes"):
        result = router(origin, destination, time, None)

        st.subheader("Best Path")

        best = result["paths"][result["best_path_index"]]
        st.write("Nodes:", best["nodes"])
        st.write("Cost:", best["cost"])

        st.subheader("All Paths")

        for p in result["paths"]:
            st.write(p["nodes"], " | cost:", p["cost"])