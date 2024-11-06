import streamlit as st
import anthropic
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import re
import pandas as pd
import pdfplumber

def process_pdf(pdf_file):
    """Extract text from PDF maintaining page numbers"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += f"\n[Page {page_num}]\n{extracted_text}"
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    return text

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Script QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title
st.title("Script QA Assistant")
st.sidebar.header("Settings")

# Initialize the AI
client = anthropic.Anthropic()

# Helper functions for script analysis
def extract_scene_info(text):
    """Extract scene information from script text"""
    scene_pattern = r"(\d+)\s+(INT\.|EXT\.)\s+(.*?)\s*-\s*(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT)"
    scenes = []
    for match in re.finditer(scene_pattern, text, re.MULTILINE):
        scenes.append({
            'scene_number': match.group(1),
            'int_ext': match.group(2),
            'location': match.group(3),
            'time': match.group(4),
            'full_header': match.group(0)
        })
    return scenes

def extract_characters(text):
    """Extract unique character names from scene text with better formatting"""
    # Look for character names in ALL CAPS before dialogue
    character_pattern = r'\n([A-Z][A-Z\s]+)(?:\s*\(.*?\))?\n'
    characters = set(re.findall(character_pattern, text))
    
    # Clean up character names
    cleaned_characters = set()
    for char in characters:
        # Remove extra whitespace and newlines
        char = ' '.join(char.split())
        # Skip stage directions that got caught
        if any(word in char for word in ['SMASH', 'CUT', 'FADE', 'DISSOLVE', 'CONTINUED']):
            continue
        # Skip if entire name is common stage direction
        if char in {'TO', 'AND', 'THE', 'WITH', 'AS', 'WE'}:
            continue
        cleaned_characters.add(char)
    
    return sorted(list(cleaned_characters))


def process_script(file_content):
    """Process script content and extract structured information"""
    scenes = {}
    current_scene = None
    current_text = []
    current_page = 1
    
    # Track page numbers using [Page X] markers
    for line in file_content.split('\n'):
        # Check for page markers
        page_match = re.match(r'\[Page (\d+)\]', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
            
        # Look for scene headers
        scene_match = re.match(r'(\d+)\s+(INT\.|EXT\.)', line)
        if scene_match:
            # Save previous scene if exists
            if current_scene:
                scenes[current_scene['number']] = {
                    'header': current_scene['header'],
                    'content': '\n'.join(current_text),
                    'page': current_scene['page']
                }
            # Start new scene
            scene_info = extract_scene_info(line)
            if scene_info:
                current_scene = {
                    'number': scene_info[0]['scene_number'],
                    'header': scene_info[0]['full_header'],
                    'page': current_page
                }
                current_text = [line]
        else:
            if current_scene:
                current_text.append(line)
    
    # Save last scene
    if current_scene:
        scenes[current_scene['number']] = {
            'header': current_scene['header'],
            'content': '\n'.join(current_text),
            'page': current_scene['page']
        }
    
    return scenes

def validate_qa_row(scene_data, qa_row):
    """Validate QA sheet row against script scene data"""
    discrepancies = []
    
    # Extract scene information
    scene_info = extract_scene_info(scene_data['header'])[0]
    
    # Check INT/EXT
    script_int_ext = scene_info['int_ext'].strip('.')
    qa_int_ext = qa_row.get('INT/EXT', '')
    if script_int_ext != qa_int_ext:
        discrepancies.append({
            'field': 'INT/EXT',
            'script_value': script_int_ext,
            'qa_value': qa_int_ext,
            'error': 'Mismatch in INT/EXT designation'
        })
    
    # Check scene header
    script_header = scene_info['full_header']
    qa_header = qa_row.get('Full scene header', '')
    if script_header != qa_header:
        discrepancies.append({
            'field': 'Scene Header',
            'script_value': script_header,
            'qa_value': qa_header,
            'error': 'Scene header does not match script exactly'
        })
    
    # Check characters with improved comparison
    script_characters = set(extract_characters(scene_data['content']))
    qa_characters = set(char.strip() for char in str(qa_row.get('Characters Present', '')).split(',') if char.strip())
    
    missing_chars = script_characters - qa_characters
    extra_chars = qa_characters - script_characters
    
    if missing_chars or extra_chars:
        discrepancies.append({
            'field': 'Characters Present',
            'script_value': ', '.join(sorted(script_characters)),
            'qa_value': ', '.join(sorted(qa_characters)),
            'error': f'Missing from QA sheet: {", ".join(sorted(missing_chars)) if missing_chars else "None"}\nExtra in QA sheet: {", ".join(sorted(extra_chars)) if extra_chars else "None"}'
        })
    
    return discrepancies


def format_analysis_report(scene_number, discrepancies):
    """Format the analysis results into a readable report"""
    report = f"\n### SCENE {scene_number} ANALYSIS:\n\n"
    
    if not discrepancies:
        report += "✅ All fields verified correct\n"
        return report
    
    report += "🚫 Discrepancies found:\n\n"
    for d in discrepancies:
        report += f"**{d['field']}**\n"
        report += f"- In Script: {d['script_value']}\n"
        report += f"- In QA Sheet: {d['qa_value']}\n"
        report += f"- Issue: {d['error']}\n\n"
    
    return report

# Initialize session storage
if "messages" not in st.session_state:
    st.session_state.messages = []
if "script_content" not in st.session_state:
    st.session_state.script_content = None
if "qa_content" not in st.session_state:
    st.session_state.qa_content = None

# File upload section
with st.sidebar:
    script_file = st.file_uploader(
        "Upload Script PDF",
        type=["pdf"],
        help="Upload the original script PDF"
    )
    
    qa_file = st.file_uploader(
        "Upload QA Sheet",
        type=["csv"],
        help="Upload the QA sheet in CSV format"
    )
    
    if script_file and qa_file and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Process script
            script_text = process_pdf(script_file)
            if script_text:
                st.session_state.script_content = process_script(script_text)
                
                # Process QA sheet
                try:
                    qa_data = pd.read_csv(qa_file)
                    st.session_state.qa_content = qa_data.to_dict('records')
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing QA sheet: {str(e)}")
            else:
                st.error("Error processing script PDF")

# Chat interface
st.markdown("### Script QA Analysis")
st.markdown("Upload both script PDF and QA sheet (CSV) to begin validation.")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Request analysis or ask specific questions about scenes"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            system_prompt = f"""You are a specialized Script QA expert focused on validating script information against QA sheets. 
            For each scene in the QA sheet, systematically analyze and verify every column against the script content.

            When analyzing each scene, verify:
            1. Scene Number matches script
            2. INT/EXT matches script header exactly
            3. Full scene header matches script exactly (excluding scene number)
            4. Scene start/end pages match actual script pages
            5. Location matches script header
            6. Time designations match script header
            7. Characters listed match all named characters with dialogue or action
            8. All vehicles mentioned in scene are documented
            9. All animals mentioned in scene are documented
            10. All content flags accurately reflect script content
            11. Scene length matches actual script content
            12. Interior/Exterior flags match header designation

            Format discrepancies as:
            SCENE [Number]:
            - Field: [field name]
            - In Script: [exact content]
            - In QA Sheet: [sheet content]
            - Correction Needed: [what needs to be changed]

            Script Content:
            {st.session_state.script_content}

            QA Sheet Content:
            {st.session_state.qa_content}"""

            if "analyze" in prompt.lower() or "check" in prompt.lower():
                # Perform full analysis
                full_report = []
                for scene_num, scene_data in st.session_state.script_content.items():
                    qa_row = next((row for row in st.session_state.qa_content 
                                 if str(row.get('Scene #', '')).strip() == str(scene_num).strip()), None)
                    if qa_row:
                        discrepancies = validate_qa_row(scene_data, qa_row)
                        report = format_analysis_report(scene_num, discrepancies)
                        full_report.append(report)
                
                response = "\n".join(full_report)
            else:
                # Use Claude for specific questions
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = response.content[0].text

            st.markdown(response)
            
            # Add to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

# Add a reset button
if st.sidebar.button("Reset Analysis"):
    st.session_state.messages = []
    st.session_state.script_content = None
    st.session_state.qa_content = None
    st.success("Analysis reset. Please upload new documents.")
