import streamlit as st
import requests

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent AI Research System")
st.markdown("*Powered by LangGraph + Groq + RAG*")
st.divider()

query = st.text_input(
    "Enter your research question:",
    placeholder="e.g. Analyze the financial risks of Infosys based on their 2024 annual report"
)

if st.button("🚀 Run Analysis", type="primary"):
    if not query:
        st.warning("Please enter a research question first.")
    else:
        with st.spinner("Running multi-agent pipeline... this takes 30-60 seconds"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    json={"query": query}
                )
                data = response.json()

                st.success("✅ Pipeline Complete!")
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📋 Plan")
                    st.markdown(data["plan"])

                    st.subheader("🔍 Research Findings")
                    st.markdown(data["research"])

                    st.subheader("🧠 Analysis")
                    st.markdown(data["analysis"])

                with col2:
                    st.subheader("📄 Final Report")
                    st.markdown(data["report"])

                    score_color = "🟢" if "APPROVED" in data["critique"] else "🔴"
                    st.subheader(f"{score_color} Quality Review")
                    st.markdown(data["critique"])

                    st.metric(
                        label="Revision Rounds",
                        value=data["revision_count"]
                    )

            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                st.info("Make sure FastAPI is running: uvicorn api:app --reload")