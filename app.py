import streamlit as st
import anthropic
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import re
import pandas as pd
import pdfplumber

# Load environment variables
load_dotenv()

# Validation rules dictionary
VALIDATION_RULES = {
    'Scene #': "Must match script scene numbering exactly",
    'Has Multiple Setups': "True if scene changes location/time within the scene",
    'INT/EXT': "Must match script header exactly (INT. or EXT.)",
    'Full scene header': "Must match script exactly, excluding scene number",
    'Scene start page (PDF)': "Must match actual PDF page number",
    'Scene end page (PDF)': "Must match actual PDF page number",
    'Scene start page (Script)': "Must match actual script page number",
    'Scene end page (Script)': "Must match actual script page number",
    'Has interior?': "True if scene header contains INT.",
    'Has exterior?': "True if scene header contains EXT.",
    'Location': "Must match location from script header",
    'Set': "Must list all distinct locations mentioned in scene",
    'Time (from header)': "Must match time designation from script header exactly",
    'Time of Day': "Must be derived from script context and header",
    'Is night?': "True if scene occurs at night based on header/context",
    'Scene length': "Must match actual length in eighths based on script",
    'Characters Present': "Must list all named characters with dialogue or action",
    'Vehicles Present': "Must list all vehicles mentioned in scene",
    'Animals Present': "Must list all animals mentioned in scene",
    'Countries': "Must list all countries mentioned in scene",
    'Contains sex/nudity?': "True if scene contains sexual content or nudity",
    'Contains violence?': "True if scene contains violent content/actions",
    'Contains profanity?': "True if scene contains profanity/strong language",
    'Contains alcohol/drugs/smoking?': "True if scene mentions substance use",
    'Contains frightening/intense moment?': "True if scene has intense/frightening content",
    'Notes': "Must accurately describe key production considerations"
}

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

def extract_scene_info(text):
    """Extract scene information from script text"""
    scene_pattern = r"(\d+)\s+(INT\.|EXT\.)\s+(.*?)\s*-\s*(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT)"
    scenes = []
    for match in re.finditer(scene_pattern, text, re.MULTILINE):
        scenes.append({
            'scene_number': match.group(1),
            'int_ext': match.group(2).strip('.'),
            'location': match.group(3).strip(),
            'time': match.group(4),
            'full_header': match.group(0)
        })
    return scenes

def extract_characters(text):
    """Extract unique character names from scene text"""
    character_pattern = r'\n([A-Z][A-Z\s]+)(?:\s*\(.*?\))?\n'
    characters = set(re.findall(character_pattern, text))
    
    cleaned_characters = set()
    for char in characters:
        char = ' '.join(char.split())
        if any(word in char for word in ['SMASH', 'CUT', 'FADE', 'DISSOLVE', 'CONTINUED']):
            continue
        if char in {'TO', 'AND', 'THE', 'WITH', 'AS', 'WE'}:
            continue
        cleaned_characters.add(char)
    
    return sorted(list(cleaned_characters))

def validate_scene(scene_data, qa_row):
    """Validate each cell in the QA row against script data"""
    validations = []
    
    # Extract scene information
    scene_info = extract_scene_info(scene_data['content'])[0]
    characters = extract_characters(scene_data['content'])
    
    # Basic Information Validation
    if scene_info['scene_number'] != str(qa_row.get('Scene #', '')):
        validations.append({
            'field': 'Scene #',
            'current': qa_row.get('Scene #', ''),
            'correct': scene_info['scene_number'],
            'status': 'Error',
            'correction': f"Update to {scene_info['scene_number']}"
        })

    # INT/EXT Validation
    script_int_ext = scene_info['int_ext']
    qa_int_ext = qa_row.get('INT/EXT', '')
    if script_int_ext != qa_int_ext:
        validations.append({
            'field': 'INT/EXT',
            'current': qa_int_ext,
            'correct': script_int_ext,
            'status': 'Error',
            'correction': f"Change to {script_int_ext}"
        })

    # Header Validation
    script_header = scene_info['full_header']
    qa_header = qa_row.get('Full scene header', '')
    if script_header != qa_header:
        validations.append({
            'field': 'Full scene header',
            'current': qa_header,
            'correct': script_header,
            'status': 'Error',
            'correction': "Update to match script header exactly"
        })

    # Interior/Exterior Validation
    has_interior = 'INT' in script_int_ext
    has_exterior = 'EXT' in script_int_ext
    if str(qa_row.get('Has interior?', '')).lower() != str(has_interior).lower():
        validations.append({
            'field': 'Has interior?',
            'current': qa_row.get('Has interior?', ''),
            'correct': has_interior,
            'status': 'Error',
            'correction': f"Change to {has_interior}"
        })
    if str(qa_row.get('Has exterior?', '')).lower() != str(has_exterior).lower():
        validations.append({
            'field': 'Has exterior?',
            'current': qa_row.get('Has exterior?', ''),
            'correct': has_exterior,
            'status': 'Error',
            'correction': f"Change to {has_exterior}"
        })

    # Time Validation
    script_time = scene_info['time']
    qa_time = qa_row.get('Time (from header)', '')
    if script_time != qa_time:
        validations.append({
            'field': 'Time (from header)',
            'current': qa_time,
            'correct': script_time,
            'status': 'Error',
            'correction': f"Change to {script_time}"
        })

    # Night Scene Validation
    is_night = any(word in script_time.upper() for word in ['NIGHT', 'MIDDLE OF THE NIGHT'])
    if str(qa_row.get('Is night?', '')).lower() != str(is_night).lower():
        validations.append({
            'field': 'Is night?',
            'current': qa_row.get('Is night?', ''),
            'correct': is_night,
            'status': 'Error',
            'correction': f"Change to {is_night}"
        })

    # Characters Validation
    qa_characters = set(char.strip() for char in str(qa_row.get('Characters Present', '')).split(',') if char.strip())
    script_characters = set(characters)
    missing_chars = script_characters - qa_characters
    extra_chars = qa_characters - script_characters
    if missing_chars or extra_chars:
        validations.append({
            'field': 'Characters Present',
            'current': ', '.join(sorted(qa_characters)),
            'correct': ', '.join(sorted(script_characters)),
            'status': 'Error',
            'correction': f"Add missing characters: {', '.join(sorted(missing_chars)) if missing_chars else 'None'}\nRemove extra characters: {', '.join(sorted(extra_chars)) if extra_chars else 'None'}"
        })

    # Content Flags Validation
    content_flags = {
        'Contains sex/nudity?': lambda x: any(word in x.lower() for word in ['nude', 'naked', 'sex', 'love scene', 'kiss']),
        'Contains violence?': lambda x: any(word in x.lower() for word in ['kill', 'shot', 'blood', 'fight', 'punch', 'hit']),
        'Contains profanity?': lambda x: any(word in x.lower() for word in ['fuck', 'shit', 'damn', 'hell', 'ass']),
        'Contains alcohol/drugs/smoking?': lambda x: any(word in x.lower() for word in ['drink', 'drunk', 'beer', 'wine', 'smoke']),
        'Contains frightening/intense moment?': lambda x: any(word in x.lower() for word in ['scream', 'terror', 'horror', 'frighten', 'intense'])
    }

    for flag, check_func in content_flags.items():
        script_has_content = check_func(scene_data['content'])
        qa_has_content = str(qa_row.get(flag, '')).lower() == 'true'
        if script_has_content != qa_has_content:
            validations.append({
                'field': flag,
                'current': qa_has_content,
                'correct': script_has_content,
                'status': 'Error',
                'correction': f"Change to {script_has_content}"
            })

    return validations

def format_validation_report(scene_number, validations):
    """Format validation results into a readable report"""
    report = f"\n### SCENE {scene_number} ANALYSIS:\n\n"
    
    if not validations:
        report += "✅ All fields verified correct\n"
        return report
    
    report += "🚫 The following fields need correction:\n\n"
    for v in validations:
        report += f"**{v['field']}**\n"
        report += f"- Current Value: {v['current']}\n"
        report += f"- Correct Value: {v['correct']}\n"
        report += f"- Correction Needed: {v['correction']}\n\n"
    
    return report

# Configure Streamlit page
st.set_page_config(
    page_title="Script QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the AI
client = anthropic.Anthropic()

# Initialize session storage
if "messages" not in st.session_state:
    st.session_state.messages = []
if "script_content" not in st.session_state:
    st.session_state.script_content = None
if "qa_content" not in st.session_state:
    st.session_state.qa_content = None

# Main interface
st.title("Script QA Assistant")
st.sidebar.header("Settings")

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
                st.session_state.script_content = {'content': script_text}
                
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
if prompt := st.chat_input("Request analysis or ask about specific scenes"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            if "analyze" in prompt.lower() or "check" in prompt.lower():
                # Perform full analysis
                full_report = []
                for qa_row in st.session_state.qa_content:
                    scene_num = qa_row.get('Scene #', '')
                    validations = validate_scene(st.session_state.script_content, qa_row)
                    report = format_validation_report(scene_num, validations)
                    full_report.append(report)
                
                response = "\n".join(full_report)
            else:
                # Use Claude for specific questions
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0,
                    system="You are a Script QA expert. Analyze the script and QA sheet to answer questions accurately.",
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
