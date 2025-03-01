# streamlit_app.py
import streamlit as st
from crewai_regulatory_analyzer import process_regulatory_obligation, DocumentAnalysis

# Streamlit App Configuration
st.set_page_config(page_title="Advanced Regulatory Analyzer", page_icon="📊")

# App Title
st.title("📊 Advanced Regulatory Obligation Analyzer")
st.markdown("Analyze regulatory documents with advanced AI-powered insights.")

# Input Section
input_type = st.radio("Select Input Type", ["URL", "File Upload"])
input_source = None

if input_type == "URL":
    input_source = st.text_input("Enter Document URL")
elif input_type == "File Upload":
    uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf"])
    if uploaded_file:
        input_source = uploaded_file.read().decode("utf-8")

# Advanced Options
with st.expander("Advanced Options"):
    api_choice_actions = st.selectbox(
        "Select API for Action Item Extraction",
        ["groq", "google_ai"],
        index=0
    )
    api_choice_mitigation = st.selectbox(
        "Select API for Risk Mitigation",
        ["groq", "google_ai"],
        index=0
    )
    temperature = st.slider("AI Temperature (Creativity vs. Precision)", 0.1, 1.0, 0.5)
    industry = st.selectbox("Select Industry", ["General", "Healthcare", "Finance", "Technology"])

# Process Button
if st.button("Analyze Document"):
    if not input_source:
        st.error("Please provide a valid document URL or file.")
    else:
        with st.spinner("Analyzing document..."):
            try:
                output = process_regulatory_obligation(
                    input_type.lower(),
                    input_source,
                    api_choice_actions,
                    api_choice_mitigation,
                    temperature,
                    industry
                )
                st.success("Analysis Complete!")

                # Display Results
                st.subheader("Analysis Results")
                st.json(output.model_dump())  # Display results in JSON format

                # Export Options
                st.download_button(
                    label="Download Results as JSON",
                    data=output.model_dump_json(),
                    file_name="analysis_results.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
