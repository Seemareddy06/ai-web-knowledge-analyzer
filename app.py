import streamlit as st
from rag_pipeline import load_and_process_urls, retrieve_docs, ask_llm
from utils import generate_summary, extract_insights, compare_docs, generate_questions

st.set_page_config(page_title="AI Web Knowledge Analyzer", layout="wide")

# 🎨 Custom UI
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

# Input URLs
urls = st.text_area("Enter URLs (comma separated)")

if st.button("🚀 Process"):
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    with st.spinner("Processing..."):
        vectorstore = load_and_process_urls(url_list)

    st.session_state.vectorstore = vectorstore
    st.session_state.urls = url_list
    st.success("✅ Ready!")

# Tabs
if "vectorstore" in st.session_state:

    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "🧠 Insights", "⚡ Advanced"])

    # 💬 Q&A TAB
    with tab1:

        query = st.text_input(
            "Ask a question",
            placeholder="Type your question here..."
        )

        if query:
            docs = retrieve_docs(st.session_state.vectorstore, query)
            context = " ".join([doc.page_content for doc in docs])

            answer = ask_llm(context, query)

            st.subheader("💬 Answer")
            st.write(answer)

            # Confidence Score
            confidence = min(len(docs) * 20, 100)
            st.subheader("📊 Confidence Score")
            st.progress(confidence)
            st.write(f"{confidence}% confident")

            # Sources
            st.subheader("📚 Sources")
            for doc in docs:
                st.write(doc.page_content[:300])

    # 🧠 INSIGHTS TAB
    with tab2:
        if st.button("📄 Generate Summary"):
            st.write(generate_summary(str(st.session_state.urls)))

        if st.button("🧠 Extract Insights"):
            st.write(extract_insights(str(st.session_state.urls)))

    # ⚡ ADVANCED TAB
    with tab3:
        if st.button("⚖️ Compare URLs"):
            all_docs = []
            for url in st.session_state.urls:
                docs = retrieve_docs(st.session_state.vectorstore, url)
                all_docs.extend([doc.page_content for doc in docs])

            combined_text = " ".join(all_docs)
            st.write(compare_docs(combined_text))

        if st.button("❓ Generate Questions"):
            st.write(generate_questions(str(st.session_state.urls)))