import streamlit as st
from rag_pipeline import (
    load_and_process_urls,
    retrieve_docs,
    ask_llm,
    get_content_per_url
)
from utils import generate_summary, extract_insights, generate_questions

st.set_page_config(page_title="AI Web Knowledge Analyzer", layout="wide")

# 🎨 UI Styling
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("## 🌐 AI Web Knowledge Analyzer")
st.caption("Analyze, Summarize, Compare & Ask Questions from Web Content 🚀")

# =========================
# SESSION STATE FIX (TAB CONTROL)
# =========================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Q&A"

# =========================
# URL INPUT
# =========================
urls = st.text_area("Enter URLs (comma separated)")

if st.button("🚀 Process"):
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    if not url_list:
        st.warning("⚠️ Please enter valid URLs")
    else:
        try:
            with st.spinner("Processing URLs..."):
                vectorstore = load_and_process_urls(url_list)

            st.session_state.vectorstore = vectorstore
            st.session_state.urls = url_list
            st.success("✅ Ready!")

        except Exception as e:
            st.error(f"❌ Failed to process URLs: {e}")

# =========================
# MAIN FEATURES
# =========================
if "vectorstore" in st.session_state:

    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "🧠 Insights", "⚡ Advanced"])

    # =========================
    # 💬 Q&A TAB
    # =========================
    with tab1:
        query = st.text_input("Ask a question")

        if query:
            docs = retrieve_docs(st.session_state.vectorstore, query)
            context = " ".join([doc.page_content for doc in docs])

            answer = ask_llm(context, query)

            st.subheader("💬 Answer")
            st.write(answer)

    # =========================
    # 🧠 INSIGHTS TAB
    # =========================
    with tab2:
        if st.button("📄 Generate Summary"):
            st.write(generate_summary(str(st.session_state.urls)))

        if st.button("🧠 Extract Insights"):
            st.write(extract_insights(str(st.session_state.urls)))

    # =========================
    # ⚡ ADVANCED TAB (FIXED)
    # =========================
    with tab3:

        if st.button("⚖️ Compare URLs", key="compare_btn"):

            url_contents = get_content_per_url(st.session_state.urls)
            urls_list = list(url_contents.keys())

            if len(urls_list) < 2:
                st.warning("⚠️ Please provide at least 2 URLs")

            else:
                text1 = url_contents[urls_list[0]]
                text2 = url_contents[urls_list[1]]

                prompt = f"""
                Compare the following two web pages:

                URL 1: {urls_list[0]}
                Content:
                {text1}

                URL 2: {urls_list[1]}
                Content:
                {text2}

                Provide:
                1. Key similarities
                2. Key differences
                3. Final conclusion
                """

                result = ask_llm("", prompt)

                st.subheader("⚖️ Comparison Result")
                st.write(result)

        if st.button("❓ Generate Questions"):
            st.write(generate_questions(str(st.session_state.urls)))