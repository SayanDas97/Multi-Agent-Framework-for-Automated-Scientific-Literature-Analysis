
import streamlit as st

from rag.pdf_processor import extract_pdf_pages, chunk_documents
from rag.retriever import build_retriever
from agents.workflow import run_research_workflow


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

DEFAULT_MODEL = "gemini-3.5-flash-lite"

st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Agent AI Research Assistant")

st.caption(
    "Research Discovery | Paper Analysis | Comparative Analysis | "
    "Research Gap Identification | Fact Checking | Report Generation"
)


# ==================================================
# 2. SIDEBAR CONFIGURATION
# ==================================================

with st.sidebar:

    st.header("Configuration")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Enter your Google Gemini API key. Do not share it publicly."
    )

    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_MODEL
    )

    st.markdown(
        "[Get a Gemini API key](https://aistudio.google.com/apikey)"
    )

    st.divider()

    st.caption(
        "Uploaded PDFs are processed in memory "
        "and are not permanently stored by this app."
    )


# ==================================================
# 3. APPLICATION DESCRIPTION
# ==================================================

st.markdown("""
### Welcome to the Research Assistant

Upload academic research papers, enter a research question,
and execute the six-agent research analysis workflow.

The application retrieves relevant evidence from your PDFs
and uses Gemini to analyse the retrieved information.
""")


# ==================================================
# 4. PDF UPLOAD
# ==================================================

uploads = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload 2–10 text-based academic research papers."
)


# ==================================================
# 5. RESEARCH QUESTION
# ==================================================

query = st.text_area(
    "Enter your research question or topic",
    placeholder=(
        "Example: What deep learning methods are used "
        "for pneumonia detection from chest X-ray images?"
    ),
    height=110
)


# ==================================================
# 6. RETRIEVAL CONFIGURATION
# ==================================================

top_k = st.slider(
    "Number of evidence passages to retrieve",
    min_value=3,
    max_value=12,
    value=6
)


# ==================================================
# 7. SESSION STATE INITIALIZATION
# ==================================================

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "paper_names" not in st.session_state:
    st.session_state.paper_names = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==================================================
# 8. ACTION BUTTONS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    build_clicked = st.button(
        "1. Process Uploaded Papers",
        type="primary",
        use_container_width=True
    )

with col2:

    run_clicked = st.button(
        "2. Run Six-Agent Analysis",
        use_container_width=True
    )


# ==================================================
# 9. PROCESS UPLOADED PDF DOCUMENTS
# ==================================================

if build_clicked:

    if not uploads:

        st.error("Please upload at least one research paper.")

    else:

        with st.spinner(
            "Extracting PDF text and building the semantic search index..."
        ):

            docs = []
            errors = []

            for uploaded in uploads:

                try:

                    pages = extract_pdf_pages(
                        uploaded.getvalue(),
                        uploaded.name
                    )

                    docs.extend(pages)

                except Exception as exc:

                    errors.append(
                        f"{uploaded.name}: {str(exc)}"
                    )

            if not docs:

                st.error(
                    "No readable text was extracted. "
                    "Please upload text-based PDFs."
                )

            else:

                try:

                    chunks = chunk_documents(docs)

                    if not chunks:
                        raise ValueError(
                            "No text chunks were created from the uploaded PDFs."
                        )

                    retriever = build_retriever(chunks)

                    st.session_state.retriever = retriever

                    st.session_state.paper_names = sorted(
                        set(
                            document["source"]
                            for document in docs
                        )
                    )

                    st.session_state.chunk_count = len(chunks)

                    # Clear previous analysis
                    st.session_state.last_result = None

                    st.success(
                        f"Successfully processed "
                        f"{len(st.session_state.paper_names)} paper(s), "
                        f"{len(docs)} pages and "
                        f"{len(chunks)} text chunks."
                    )

                    if errors:

                        st.warning(
                            "\n".join(errors)
                        )

                    with st.expander("View processed paper names"):

                        for paper in st.session_state.paper_names:
                            st.write(f"• {paper}")

                except Exception as exc:

                    st.error("Error while building the knowledge base.")
                    st.exception(exc)


# ==================================================
# 10. KNOWLEDGE BASE STATUS
# ==================================================

if st.session_state.retriever is not None:

    st.success(
        f"Knowledge Base Ready | "
        f"Papers: {len(st.session_state.paper_names)} | "
        f"Chunks: {st.session_state.chunk_count}"
    )

    # Preview retrieved evidence
    if query.strip():

        with st.expander("Preview Retrieved Evidence"):

            try:

                preview = st.session_state.retriever.search(
                    query,
                    top_k=top_k
                )

                if not preview:

                    st.warning("No relevant evidence was retrieved.")

                else:

                    for index, item in enumerate(preview, start=1):

                        st.markdown(
                            f"**{index}. {item['source']} | "
                            f"Page {item['page']} | "
                            f"Score: {item['score']:.3f}**"
                        )

                        st.write(item["text"])

                        st.divider()

            except Exception as exc:

                st.error("Evidence retrieval failed.")
                st.exception(exc)


# ==================================================
# 11. EXECUTE SIX-AGENT RESEARCH WORKFLOW
# ==================================================

if run_clicked:

    if st.session_state.retriever is None:

        st.error("Please process your research papers first.")

    elif not query.strip():

        st.error("Please enter a research question.")

    elif not api_key.strip():

        st.error("Please enter your Gemini API key in the sidebar.")

    elif not model_name.strip():

        st.error("Please enter a valid Gemini model name.")

    else:

        try:

            with st.spinner(
                "The six research agents are analysing your papers. "
                "Please wait..."
            ):

                result = run_research_workflow(
                    question=query.strip(),
                    retriever=st.session_state.retriever,
                    api_key=api_key.strip(),
                    model_name=model_name.strip(),
                    top_k=top_k
                )

            st.session_state.last_result = result

            st.success(
                "The six-agent research workflow completed successfully!"
            )

        except Exception as exc:

            st.error(
                "The research workflow could not be completed. "
                "Please check the API key, model availability and quota."
            )

            st.exception(exc)


# ==================================================
# 12. DISPLAY FINAL RESEARCH REPORT
# ==================================================

if st.session_state.last_result is not None:

    result = st.session_state.last_result

    st.divider()

    st.header("📄 Final Research Report")

    st.markdown(
        result["final_report"]
    )

    # Download report
    st.download_button(
        label="Download Research Report",
        data=result["final_report"],
        file_name="research_analysis_report.md",
        mime="text/markdown"
    )

    # Individual agent outputs
    with st.expander("View Individual Agent Outputs"):

        for agent_name, content in result["agent_outputs"].items():

            st.subheader(agent_name)

            st.markdown(content)

            st.divider()

    # Retrieved evidence
    with st.expander("View Retrieved Evidence and References"):

        for index, item in enumerate(result["evidence"], start=1):

            st.markdown(
                f"**[{index}] {item['source']} | "
                f"Page {item['page']} | "
                f"Similarity: {item['score']:.3f}**"
            )

            st.write(item["text"])

            st.divider()

    st.caption(
        "The report is generated using evidence retrieved from "
        "the uploaded PDFs. Verify all findings and references "
        "against the original research papers before academic submission."
    )


# ==================================================
# 13. APPLICATION FOOTER
# ==================================================

st.divider()

st.caption(
    "Multi-Agent AI Research Assistant | "
    "PDF Processing | Semantic Retrieval | "
    "Gemini-Powered Research Workflow"
)

st.caption(
    "Fine-tuning, external research discovery and benchmark "
    "evaluation are planned extensions."
)