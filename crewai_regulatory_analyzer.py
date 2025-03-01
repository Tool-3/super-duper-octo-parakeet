import os
import asyncio
import requests
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------
# Configuration Models
# --------------------------
class AnalysisConfig(BaseModel):
    industry: str = Field(default="General", description="Industry focus for analysis")
    jurisdiction: str = Field(default="Global", description="Regulatory jurisdiction")
    risk_tolerance: str = Field(default="Medium", description="Organization's risk tolerance level")
    temperature: float = Field(default=0.3, ge=0.1, le=1.0, description="LLM creativity vs precision")

class DocumentMetadata(BaseModel):
    source: str = Field(..., description="Document source information")
    document_type: str = Field(..., description="Type of regulatory document")
    effective_date: str = Field(..., description="Document effective date")

# --------------------------
# LLM Configuration
# --------------------------
class LLMFactory:
    @staticmethod
    def get_llm(provider: str, config: AnalysisConfig):
        if provider == "groq":
            return ChatGroq(
                temperature=config.temperature,
                model="mixtral-8x7b-32768",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif provider == "google_ai":
            return LLMFactory._init_google_llm(config)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _init_google_llm(config: AnalysisConfig):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=config.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        finally:
            loop.close()

# --------------------------
# Agent Definitions
# --------------------------
class RegulatoryAnalyst(Agent):
    def __init__(self, config: AnalysisConfig):
        super().__init__(
            role="Senior Regulatory Analyst",
            goal="Accurate interpretation of regulatory requirements",
            backstory=(
                "With over 15 years of experience in regulatory compliance across multiple industries, "
                "you specialize in breaking down complex legal texts into actionable organizational requirements. "
                "Your analyses are known for their precision and practical applicability."
            ),
            verbose=True,
            allow_delegation=False,
            llm=LLMFactory.get_llm("groq", config),
            memory=True
        )

class ComplianceArchitect(Agent):
    def __init__(self, config: AnalysisConfig):
        super().__init__(
            role="Compliance Framework Architect",
            goal="Develop implementable compliance strategies",
            backstory=(
                "As a certified compliance professional with expertise in multiple regulatory frameworks (GDPR, HIPAA, SOX), "
                "you transform regulatory requirements into operational workflows while maintaining audit readiness."
            ),
            verbose=True,
            allow_delegation=True,
            llm=LLMFactory.get_llm("google_ai", config),
            memory=True
        )

class RiskStrategist(Agent):
    def __init__(self, config: AnalysisConfig):
        super().__init__(
            role="Enterprise Risk Strategist",
            goal="Identify and mitigate regulatory risks",
            backstory=(
                "With a PhD in Risk Management and 12 years of consulting experience, "
                "you develop risk mitigation strategies that balance compliance costs with organizational objectives."
            ),
            verbose=True,
            allow_delegation=True,
            llm=LLMFactory.get_llm("groq", config),
            memory=True
        )

class QualityValidator(Agent):
    def __init__(self, config: AnalysisConfig):
        super().__init__(
            role="Quality Assurance Specialist",
            goal="Ensure analysis accuracy and completeness",
            backstory=(
                "As a meticulous former auditor with expertise in regulatory documentation, "
                "you catch inconsistencies and validate compliance recommendations against industry standards."
            ),
            verbose=True,
            allow_delegation=False,
            llm=LLMFactory.get_llm("google_ai", config),
            memory=True
        )

# --------------------------
# Task Definitions
# --------------------------
class AnalysisTasks:
    def __init__(self, config: AnalysisConfig, metadata: DocumentMetadata):
        self.config = config
        self.metadata = metadata

    def contextual_analysis_task(self, agent: Agent) -> Task:
        return Task(
            description=(
                f"Analyze the regulatory document from {self.metadata.source} "
                f"effective {self.metadata.effective_date} for {self.config.industry} industry. "
                "Identify key regulatory obligations and their applicability scope."
            ),
            expected_output=(
                "Structured analysis of document scope, key obligations, "
                "and applicability to different organizational functions."
            ),
            agent=agent,
            output_json={
                "document_summary": "Brief overview of document purpose and scope",
                "key_obligations": "List of critical regulatory requirements",
                "applicability_matrix": "Mapping of requirements to business units"
            }
        )

    def compliance_mapping_task(self, agent: Agent) -> Task:
        return Task(
            description=(
                "Develop compliance implementation roadmap considering "
                f"{self.config.jurisdiction} jurisdiction and {self.config.risk_tolerance} risk tolerance."
            ),
            expected_output=(
                "Prioritized compliance action plan with resource requirements "
                "and implementation timelines."
            ),
            agent=agent,
            output_json={
                "action_plan": "List of compliance actions with priorities",
                "resource_allocation": "Required resources for implementation",
                "timeline": "Realistic implementation schedule"
            }
        )

    def risk_assessment_task(self, agent: Agent) -> Task:
        return Task(
            description=(
                "Conduct risk impact analysis considering organizational risk tolerance "
                f"({self.config.risk_tolerance}) and {self.config.industry} industry standards."
            ),
            expected_output=(
                "Risk assessment matrix with mitigation strategies and contingency plans "
                "for high-probability/high-impact risks"
            ),
            agent=agent,
            output_json={
                "risk_matrix": "Assessment of identified risks",
                "mitigation_strategies": "Detailed risk mitigation approaches",
                "contingency_plans": "Fallback plans for residual risks"
            }
        )

    def validation_task(self, agent: Agent, context: List[Task]) -> Task:
        return Task(
            description=(
                "Validate all analysis outputs for consistency, completeness, "
                "and compliance with industry best practices."
            ),
            expected_output=(
                "Validation report highlighting any discrepancies, "
                "potential gaps, and improvement recommendations"
            ),
            agent=agent,
            context=context,
            output_json={
                "validation_summary": "Overall validation status",
                "identified_issues": "List of potential problems",
                "corrective_actions": "Recommended improvements"
            }
        )

# --------------------------
# Analysis Orchestrator
# --------------------------
class RegulatoryAnalysisOrchestrator:
    def __init__(self, config: AnalysisConfig, metadata: DocumentMetadata):
        self.config = config
        self.metadata = metadata
        self.tasks = AnalysisTasks(config, metadata)
        
        self.analyst = RegulatoryAnalyst(config)
        self.architect = ComplianceArchitect(config)
        self.risk_strategist = RiskStrategist(config)
        self.validator = QualityValidator(config)

    def create_crew(self) -> Crew:
        tasks = [
            self.tasks.contextual_analysis_task(self.analyst),
            self.tasks.compliance_mapping_task(self.architect),
            self.tasks.risk_assessment_task(self.risk_strategist),
            self.tasks.validation_task(self.validator, [
                self.tasks.contextual_analysis_task,
                self.tasks.compliance_mapping_task,
                self.tasks.risk_assessment_task
            ])
        ]

        return Crew(
            agents=[self.analyst, self.architect, self.risk_strategist, self.validator],
            tasks=tasks,
            verbose=2,
            memory=True,
            process="sequential"  # Ensures task order and dependency handling
        )

# --------------------------
# Main Execution Flow
# --------------------------
def analyze_regulatory_document(document_content: str, 
                               config: AnalysisConfig, 
                               metadata: DocumentMetadata) -> Dict:
    """Main analysis workflow executor"""
    orchestrator = RegulatoryAnalysisOrchestrator(config, metadata)
    crew = orchestrator.create_crew()
    
    try:
        results = crew.kickoff(inputs={'document_content': document_content})
        return {
            "status": "success",
            "analysis": results,
            "metadata": metadata.dict(),
            "config": config.dict()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "metadata": metadata.dict(),
            "config": config.dict()
        }

# --------------------------
# Streamlit Interface
# --------------------------
def streamlit_interface():
    import streamlit as st
    
    st.set_page_config(page_title="Enterprise Regulatory Analyzer", layout="wide")
    st.title("Enterprise Regulatory Analysis Platform")
    
    with st.sidebar:
        st.header("Analysis Configuration")
        industry = st.selectbox("Industry Focus", ["Healthcare", "Finance", "Technology", "Energy"])
        jurisdiction = st.selectbox("Jurisdiction", ["EU", "US Federal", "California", "Global"])
        risk_tolerance = st.select_slider("Risk Tolerance", ["Low", "Medium", "High"])
        temperature = st.slider("Analysis Creativity", 0.1, 1.0, 0.3)
        
    doc_source = st.radio("Document Source", ["URL", "File Upload"])
    document_content = ""
    
    if doc_source == "URL":
        url = st.text_input("Document URL")
        if url:
            try:
                response = requests.get(url)
                document_content = response.text
            except Exception as e:
                st.error(f"Failed to fetch document: {str(e)}")
    else:
        uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf"])
        if uploaded_file:
            document_content = uploaded_file.read().decode("utf-8")
    
    if st.button("Start Analysis"):
        if not document_content:
            st.error("Please provide document content")
            return
            
        config = AnalysisConfig(
            industry=industry,
            jurisdiction=jurisdiction,
            risk_tolerance=risk_tolerance,
            temperature=temperature
        )
        
        metadata = DocumentMetadata(
            source=doc_source,
            document_type="Regulatory Text",
            effective_date="2024-01-01"  # Should be extracted from document
        )
        
        with st.spinner("Performing comprehensive analysis..."):
            result = analyze_regulatory_document(document_content, config, metadata)
            
            if result['status'] == 'success':
                st.success("Analysis Complete")
                st.json(result)
            else:
                st.error(f"Analysis Failed: {result['message']}")

if __name__ == "__main__":
    streamlit_interface()
