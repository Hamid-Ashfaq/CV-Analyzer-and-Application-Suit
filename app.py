import io
import re
import streamlit as st
import pypdf
import docx
from google import genai
from google.genai import types

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Academic CV Analyzer & Application Builder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper Functions for File Text Extraction
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        extracted_text = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    try:
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def extract_text_from_txt(uploaded_file) -> str:
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

def call_gemini_with_retry(client, model_name: str, contents: str, max_retries: int = 3):
    """
    Calls client.models.generate_content with exponential backoff for 503/429 errors,
    and automatic fallback to alternative models if high demand persists.
    """
    import time
    
    candidate_models = [model_name]
    fallback_options = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    for fb in fallback_options:
        if fb not in candidate_models:
            candidate_models.append(fb)
            
    last_exception = None
    for current_model in candidate_models:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents
                )
                return response, current_model
            except Exception as e:
                err_str = str(e)
                last_exception = e
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = attempt * 3
                    if attempt < max_retries:
                        st.toast(f"⏳ Server busy ({current_model}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        st.toast(f"⚠️ Model `{current_model}` high demand. Trying alternate model...")
                else:
                    raise e
                    
    raise last_exception

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=64)
    st.title("Settings & Options")
    
    # Secrets / API Key Handling
    gemini_api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔑 Gemini API Key loaded from secrets")
    else:
        st.warning("⚠️ No `GEMINI_API_KEY` found in `st.secrets`.")
        gemini_api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="For automatic configuration, place your API key in `.streamlit/secrets.toml` under `GEMINI_API_KEY`."
        )
    
    st.divider()
    
    # Position Configuration
    position_type = st.selectbox(
        "Target Degree Level:",
        ["PhD Position", "Master's Position (Thesis/Research)", "Research Assistantship (RA)", "Postdoctoral Fellowship"],
        index=0
    )
    
    selected_model = st.selectbox(
        "Gemini Model:",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"],
        index=0,
        help="gemini-3.8-flash provides fast, high-quality analytical outputs."
    )
    
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
        1. **Fill position details** & upload your current CV.
        2. **Analyze CV** to get a match score, gap analysis & concrete tailoring advice.
        3. **Generate Materials** to get a custom cold email, cover letter, motivation letter, statement of interest & research proposal draft.
        """
    )

# App Header
st.markdown('<div class="main-header">🎓 PhD & Master\'s CV Analyzer & Application Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Tailor your CV and auto-generate application documents for competitive academic positions using AI.</div>', unsafe_allow_html=True)

# Session State Initialization
if "cv_text" not in st.session_state:
    st.session_state["cv_text"] = ""
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "materials_result" not in st.session_state:
    st.session_state["materials_result"] = None

# Input Section: Position Details & CV Upload
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Position Details & Requirements")
    position_title = st.text_input(
        "Position Title:",
        placeholder="e.g. PhD Candidate in Computer Vision & Medical Imaging"
    )
    institution_name = st.text_input(
        "University / Institute / Lab:",
        placeholder="e.g. ETH Zurich - Computer Vision Lab"
    )
    professor_name = st.text_input(
        "Professor / PI / Hiring Manager Name (Optional):",
        placeholder="e.g. Prof. Dr. Anna Schmidt"
    )
    research_field = st.text_input(
        "Research Field / Specialization:",
        placeholder="e.g. Machine Learning, Computational Biology, Physics"
    )
    position_description = st.text_area(
        "Position Description & Requirements:",
        height=220,
        placeholder="Paste the job posting, research topics, required qualifications, prerequisites, skills, and project description here..."
    )

with col2:
    st.subheader("2. Your Academic CV")
    uploaded_file = st.file_uploader(
        "Upload your CV (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"]
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = extract_text_from_txt(uploaded_file)
            
        if text:
            st.session_state["cv_text"] = text
            st.success(f"Successfully loaded {uploaded_file.name} ({len(text)} characters extracted).")
            
    cv_text_input = st.text_area(
        "CV Content (Review or Paste directly):",
        value=st.session_state["cv_text"],
        height=260,
        placeholder="Extracted CV content will appear here, or you can paste your CV text directly..."
    )
    st.session_state["cv_text"] = cv_text_input

st.divider()

# Analysis Action Button
st.subheader("3. Analyze CV & Recommendations")

can_analyze = bool(gemini_api_key and st.session_state["cv_text"].strip() and position_description.strip())

if not gemini_api_key:
    st.info("💡 Please ensure your Gemini API Key is configured in `.streamlit/secrets.toml` or entered in the sidebar to proceed.")

analyze_button = st.button(
    "🚀 Analyze CV against Position",
    type="primary",
    disabled=not can_analyze,
    use_container_width=True
)

if analyze_button:
    with st.spinner("Analyzing your CV against position requirements using Gemini..."):
        try:
            client = genai.Client(api_key=gemini_api_key)
            
            analysis_prompt = f"""
You are an expert academic admissions advisor and principal research investigator evaluating candidates for higher education and research positions ({position_type}).

Analyze the following CV against the provided Position Description and Requirements.

### POSITION DETAILS
- Target Degree / Level: {position_type}
- Title: {position_title if position_title else 'Academic Position'}
- University / Institute: {institution_name if institution_name else 'Specified Institution'}
- Professor / PI: {professor_name if professor_name else 'Not specified'}
- Field: {research_field if research_field else 'General Research'}
- Position Description & Requirements:
{position_description}

### CANDIDATE CV
{st.session_state['cv_text']}

---

Provide a comprehensive, objective, and highly actionable analysis formatted cleanly in Markdown:

1. **Overall Match Score**: Give an overall compatibility score from 0% to 100% with a short 2-sentence rationale. Format score header explicitly as: `### Match Score: X%`.
2. **Executive Summary of Fit**: A concise evaluation of how well the candidate aligns with the research agenda and requirements.
3. **Key Strengths & Position Alignments**: 4-6 bullet points highlighting relevant research experience, skills, publications, or academic achievements that match the position.
4. **Gaps & Weaknesses**: 3-5 bullet points identifying missing prerequisites, weak technical areas, lacking domain keywords, or unhighlighted experience.
5. **Actionable CV Tailoring Recommendations**:
   - **Summary / Profile Section**: Specific phrasing recommendations to align research goals.
   - **Skills & Methodologies**: Key technical skills, lab techniques, frameworks, or tools to emphasize or add.
   - **Research & Project Highlights**: Bullet point rewrites to reframe previous projects toward this specific position.
   - **Education & Publications**: Recommendations on presenting relevant coursework, thesis topics, or preprints.
"""
            
            response, used_model = call_gemini_with_retry(client, selected_model, analysis_prompt)
            st.session_state["analysis_result"] = response.text
            st.session_state["materials_result"] = None  # Reset materials on new analysis
            if used_model != selected_model:
                st.info(f"ℹ️ Generated using `{used_model}` due to high demand on `{selected_model}`.")
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")

# Render Analysis Results
if st.session_state["analysis_result"]:
    st.markdown("---")
    st.subheader("📊 Analysis Results & CV Tailoring Recommendations")
    
    analysis_text = st.session_state["analysis_result"]
    
    # Try parsing match score for display widget
    score_match = re.search(r"Match Score:\s*(\d+)%", analysis_text, re.IGNORECASE)
    if score_match:
        score_val = int(score_match.group(1))
        col_s1, col_s2 = st.columns([1, 4])
        with col_s1:
            st.metric("Overall Fit Score", f"{score_val}%")
            st.progress(score_val / 100)
    
    st.markdown(analysis_text)
    
    st.divider()
    
    # SECTION 4: Application Materials Generator Button
    st.subheader("✉️ Application Materials Generator")
    st.write("Generate a complete set of tailored academic application documents customized specifically for this position based on your CV and the analysis results.")
    
    gen_materials_button = st.button(
        "✨ Prepare Customized Application Materials (Email, Cover Letter, Motivation Letter, SOI, Research Proposal)",
        type="primary",
        use_container_width=True
    )
    
    if gen_materials_button:
        with st.spinner("Drafting tailored cold email, cover letter, motivation letter, statement of interest, and research proposal outline..."):
            try:
                client = genai.Client(api_key=gemini_api_key)
                
                materials_prompt = f"""
You are an elite academic career mentor and grant/admissions consultant.
Using the provided Position Details, Candidate CV, and the prior Analysis & Recommendations, draft high-quality, professional, customized application materials tailored for this specific position.

### POSITION DETAILS
- Target Degree Level: {position_type}
- Position Title: {position_title if position_title else 'Academic Position'}
- Institution / Lab: {institution_name if institution_name else 'Specified Institution'}
- Professor / PI: {professor_name if professor_name else 'Prof. / Hiring Committee'}
- Field / Topic: {research_field if research_field else 'Specified Research Area'}
- Description:
{position_description}

### CANDIDATE CV
{st.session_state['cv_text']}

### CV ANALYSIS & RECOMMENDATIONS SUMMARY
{st.session_state['analysis_result']}

---

Generate five distinct, fully customized, high-impact documents. Format each document under its exact section header as specified below:

---DOCUMENT: EMAIL---
[Draft a concise, professional cold outreach email to Professor/PI or hiring manager. Include: Subject line, formal salutation, compelling hook referencing their lab's research, 2-3 lines on how candidate background matches position, reference to attached CV, and polite call to action requesting a brief call/discussion.]

---DOCUMENT: COVER_LETTER---
[Draft a formal academic cover letter. Include header placeholders, motivation for applying to this specific position and institution, alignment of past research/education with job requirements, key technical strengths, and professional closing.]

---DOCUMENT: MOTIVATION_LETTER---
[Draft an inspiring, comprehensive motivation letter. Detail candidate's academic journey, core research curiosity, why this laboratory/department is the ideal environment, and long-term career ambition in academia/research.]

---DOCUMENT: STATEMENT_OF_INTEREST---
[Draft a structured Statement of Interest (SOI). Focus on research interests, theoretical & methodological competencies, past project impacts, and alignment with the proposed research area of this position.]

---DOCUMENT: RESEARCH_PROPOSAL---
[Draft a structured Research Proposal / Research Plan outline tailored to the specific position topic. Include:
1. Proposed Title
2. Abstract
3. Problem Statement & Research Gaps
4. Objectives & Key Research Questions
5. Proposed Methodology & Technical Approach
6. Expected Outcomes & Impact
7. Key Literature References Outline]
"""
                
                response_mat, used_model_mat = call_gemini_with_retry(client, selected_model, materials_prompt)
                
                st.session_state["materials_result"] = response_mat.text
                st.success("Application materials successfully generated!")
                if used_model_mat != selected_model:
                    st.info(f"ℹ️ Generated using `{used_model_mat}` due to high demand on `{selected_model}`.")
                
            except Exception as e:
                st.error(f"Error generating application materials: {e}")

# Display Application Materials in Tabs
if st.session_state.get("materials_result"):
    st.markdown("---")
    st.subheader("📑 Tailored Application Documents")
    
    mat_text = st.session_state["materials_result"]
    
    # Parse individual documents
    def parse_doc(tag, full_text):
        pattern = rf"---DOCUMENT:\s*{tag}---\s*(.*?)(?=---DOCUMENT:|\Z)"
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return full_text
        
    email_doc = parse_doc("EMAIL", mat_text)
    cover_doc = parse_doc("COVER_LETTER", mat_text)
    motiv_doc = parse_doc("MOTIVATION_LETTER", mat_text)
    soi_doc = parse_doc("STATEMENT_OF_INTEREST", mat_text)
    proposal_doc = parse_doc("RESEARCH_PROPOSAL", mat_text)
    
    tab_email, tab_cover, tab_motiv, tab_soi, tab_proposal = st.tabs([
        "📧 Cold Email",
        "📄 Cover Letter",
        "💡 Motivation Letter",
        "🎯 Statement of Interest",
        "🔬 Research Proposal"
    ])
    
    with tab_email:
        st.markdown("### Cold Outreach Email to Professor / Hiring Manager")
        st.markdown(email_doc)
        st.download_button(
            "💾 Download Email Draft (.txt)",
            data=email_doc,
            file_name="cold_email_draft.txt",
            mime="text/plain"
        )
        
    with tab_cover:
        st.markdown("### Academic Cover Letter")
        st.markdown(cover_doc)
        st.download_button(
            "💾 Download Cover Letter (.md)",
            data=cover_doc,
            file_name="cover_letter.md",
            mime="text/markdown"
        )
        
    with tab_motiv:
        st.markdown("### Motivation Letter")
        st.markdown(motiv_doc)
        st.download_button(
            "💾 Download Motivation Letter (.md)",
            data=motiv_doc,
            file_name="motivation_letter.md",
            mime="text/markdown"
        )
        
    with tab_soi:
        st.markdown("### Statement of Interest (SOI)")
        st.markdown(soi_doc)
        st.download_button(
            "💾 Download Statement of Interest (.md)",
            data=soi_doc,
            file_name="statement_of_interest.md",
            mime="text/markdown"
        )
        
    with tab_proposal:
        st.markdown("### Tailored Research Proposal Outline")
        st.markdown(proposal_doc)
        st.download_button(
            "💾 Download Research Proposal Outline (.md)",
            data=proposal_doc,
            file_name="research_proposal_outline.md",
            mime="text/markdown"
        )
