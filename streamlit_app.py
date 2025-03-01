import streamlit as st
from crewai_regulatory_analyzer import process_regulatory_obligation, DocumentAnalysis
from typing import Optional

# App Configuration
st.set_page_config(
    page_title="AI Regulatory Analyzer Pro",
    page_icon="🔍",
    layout="wide"
)

def main():
    # Header Section
    st.title("🔍 AI Regulatory Analyzer Pro")
    st.markdown("""
    **Analyze regulatory documents with multi-agent AI system**
    - Action item extraction
    - Compliance checking
    - Risk mitigation strategies
    - Timeline planning
    """)
    
    # Input Section
    with st.sidebar:
        st.header("Configuration")
        input_type = st.radio("Input Type", ["URL", "File Upload"])
        api_actions = st.selectbox("Action Item API", ["groq", "google_ai"])
        api_mitigation = st.selectbox("Mitigation API", ["groq", "google_ai"])
        temperature = st.slider("AI Temperature", 0.1, 1.0, 0.5)
        industry = st.selectbox("Industry", ["General", "Healthcare", "Finance", "Technology"])
    
    # Document Input
    document_content: Optional[str] = None
    if input_type == "URL":
        url = st.text_input("Document URL")
        if url:
            document_content = url
    else:
        uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf"])
        if uploaded_file:
            document_content = uploaded_file.read().decode("utf-8")
    
    # Analysis Execution
    if st.button("Analyze Document") and document_content:
        with st.spinner("Deep analysis in progress..."):
            try:
                result: DocumentAnalysis = process_regulatory_obligation(
                    input_type.lower(),
                    document_content,
                    api_actions,
                    api_mitigation,
                    temperature,
                    industry
                )
                display_results(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def display_results(result: DocumentAnalysis):
    # Context Summary
    with st.expander("🌐 Document Context"):
        st.json(result.context)
    
    # Paragraph Analysis
    st.subheader("📝 Detailed Analysis")
    for idx, para in enumerate(result.paragraphs):
        with st.expander(f"Paragraph {idx+1}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text**")
                st.caption(para.text[:500] + "..." if len(para.text) > 500 else para.text)
                
                st.markdown("**Identified Actions**")
                for action in para.actions:
                    st.markdown(f"- {action.description}")
                    st.caption(f"Priority: {action.priority} | Timeline: {action.timeline}")
            
            with col2:
                st.markdown("**Risk Mitigations**")
                for mitigation in para.mitigations:
                    st.markdown(f"**{mitigation.strategy}**")
                    st.caption(f"Stakeholders: {', '.join(mitigation.stakeholders)}")
                    st.markdown("Steps:")
                    for step in mitigation.steps:
                        st.markdown(f"- {step}")
    
    # Final Report
    st.subheader("📊 Executive Summary")
    st.markdown(result.report)
    
    # Export Options
    st.download_button(
        label="Download Full Report (JSON)",
        data=result.model_dump_json(),
        file_name="regulatory_analysis.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
