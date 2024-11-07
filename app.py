import streamlit as st
import anthropic
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import re
import pandas as pd
import pdfplumber
import traceback
import csv
from difflib import SequenceMatcher
import spacy

# Load environment variables
load_dotenv()

# Get API key and initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    st.error("No Anthropic API key found. Please make sure ANTHROPIC_API_KEY is set in your .env file.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Script QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

def process_pdf(pdf_file):
    """Extract text from PDF maintaining page numbers"""
    text = ""
    page_mapping = {}
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += f"\n[Page {page_num}]\n{extracted_text}"
                    page_mapping[page_num] = extracted_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None
    return text, page_mapping

def process_csv(csv_file):
    """Extract text from CSV maintaining row order"""
    text = ""
    try:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            text += " ".join(row) + "\n"
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None
    return text

# In the main content area
if script_file and qa_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Process Documents", type="primary", key="process_docs"):
            with st.spinner("Processing documents..."):
                # Check file type
                if script_file.type == "application/pdf":
                    script_text, page_mapping = process_pdf(script_file)
                elif script_file.type == "text/csv":
                    script_text = process_csv(script_file)
                    page_mapping = None
                else:
                    st.error("Unsupported file type. Please upload a PDF or CSV file.")
                    st.stop()

                if script_text:
                    try:
                        # Split script into scenes
                        scenes = split_into_scenes(script_text)
                        st.session_state.script_content = scenes
                        
                        # Process QA sheet
                        qa_data = pd.read_csv(qa_file)
                        qa_rows = qa_data.to_dict('records')
                        st.session_state.qa_content = qa_rows
                        
                        # Debug information in expander
                        with st.expander("Analysis Details"):
                            st.info(f"Found {len(scenes)} scenes")
                            st.text(f"Scene numbers: {', '.join(sorted(scenes.keys()))}")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["Scene Analysis", "Summary"])
                        
                        with tab1:
                            st.markdown("### Scene-by-Scene Analysis")
                            for qa_row in qa_rows:
                                scene_num = str(qa_row.get('Scene #', '')).strip()
                                scene_content = scenes.get(scene_num)
                                
                                # Create expander for each scene
                                with st.expander(f"Scene {scene_num}", expanded=True):
                                    if scene_content:
                                        validations = validate_scene(scene_content, qa_row)
                                        report = format_validation_report(scene_num, validations)
                                        st.markdown(report)
                                    else:
                                        st.markdown(f"⚠️ Scene not found in script")
                        
                        with tab2:
                            st.markdown("### Analysis Summary")
                            total_scenes = len(qa_rows)
                            scenes_with_issues = sum(1 for row in qa_rows if scenes.get(str(row.get('Scene #', '')).strip()))
                            st.metric("Total Scenes Processed", total_scenes)
                            st.metric("Scenes with Issues", scenes_with_issues)
                        
                        st.success(f"Analysis complete! Processed {len(scenes)} scenes.")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                else:
                    st.error("Error processing script PDF")
    with col2:
        if st.button("Reset Analysis", type="secondary"):
            st.session_state.messages = []
            st.session_state.script_content = None
            st.session_state.qa_content = None
            st.rerun()

def extract_scene_header(text):
    """Extract scene header information"""
    scene_pattern = r"(\d+)\s+(INT\.|EXT\.)\s+(.*?)\s*-\s*(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT)"
    match = re.search(scene_pattern, text)
    if match:
        return {
            'scene_number': match.group(1),
            'int_ext': match.group(2).strip('.'),
            'location': match.group(3).strip(),
            'time': match.group(4),
            'full_header': match.group(0)
        }
    return None

def split_into_scenes(script_text):
    """Split script text into individual scenes"""
    scene_pattern = r'(\d+[A-Z]?)\s+(INT\.|EXT\.)\s+.*?\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT)'
    scenes = {}
    
    matches = list(re.finditer(scene_pattern, script_text))
    
    for i in range(len(matches)):
        start = matches[i].start()
        if i == len(matches) - 1:
            end = len(script_text)
        else:
            end = matches[i + 1].start()
        
        scene_text = script_text[start:end].strip()
        scene_num = matches[i].group(1)
        
        if len(scene_text) > 50:
            scenes[scene_num] = scene_text
    
    return scenes

def extract_scene_characters(text):
    """Extract only characters that actually appear in the scene"""
    characters = set()
    
    # First pass - look for character names before dialogue or scene directions
    character_pattern = r'\n([A-Z][A-Z\s\'\-]+)(?:\s*\([^)]+\))?\s*\n(?=[\s\w])'
    
    # Additional words to exclude
    excluded_words = {
        'INT', 'EXT', 'CONTINUED', 'CONTINUOUS', 'CUT TO', 'FADE IN', 'FADE OUT',
        'DISSOLVE TO', 'SMASH CUT', 'BACK TO', 'FLASHBACK', 'END', 'THE', 'BUT',
        'THEN', 'WHEN', 'AND', 'SCENE', 'WE', 'NOW', 'LATER', 'DAY', 'NIGHT',
        'MORNING', 'EVENING', 'AFTERNOON', 'CHYRON', 'SUBTITLE', 'MORE', 'CONT',
        'NOTE', 'LEVIATHAN', 'PINK', 'GREEN', 'YELLOW', 'BLUE', 'WHITE',
        'MOMENTS', 'HELP', 'STOP', 'SEES', 'SUDDENLY', 'SHOOTS'
    }
    
    # Process pattern
    matches = re.finditer(character_pattern, text, re.MULTILINE)
    for match in matches:
        name = match.group(1).strip()
        # Only add if it's not in excluded words and looks like a name
        if (name not in excluded_words and 
            len(name) > 1 and
            not any(word in name for word in ['VOICES', 'CROWD', 'VARIOUS']) and
            not name.endswith('S HOME') and
            not name.startswith('CONT') and
            not any(char.isdigit() for char in name)):
            characters.add(name)
    
    return sorted(list(characters))
def check_scene_content(text):
    """Analyze scene content for various flags"""
    content_flags = {
        'sex/nudity': {
            'status': False,
            'keywords': ['nude', 'naked', 'sex', 'breast', 'tit', 'motorboat', 'love scene', 'kiss', 'jiggle'],
            'evidence': []
        },
        'violence': {
            'status': False,
            'keywords': ['kill', 'shot', 'blood', 'fight', 'punch', 'hit', 'slaughter', 'death', 'die', 'gun', 'shoot', 'bullet'],
            'evidence': []
        },
        'profanity': {
            'status': False,
            'keywords': ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard', 'kike'],
            'evidence': []
        },
        'alcohol/drugs': {
            'status': False,
            'keywords': ['drink', 'drunk', 'beer', 'wine', 'liquor', 'gimlet', 'mai tai', 'smoking', 'cigarette'],
            'evidence': []
        },
        'frightening': {
            'status': False,
            'keywords': ['scream', 'terror', 'horror', 'frighten', 'intense', 'violent', 'blood', 'kill', 'death'],
            'evidence': []
        }
    }
    
    text_lower = text.lower()
    for flag, data in content_flags.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                data['status'] = True
                if keyword not in data['evidence']:
                    data['evidence'].append(keyword)
    
    return content_flags

def validate_scene(scene_text, qa_row):
    """Validate scene against QA sheet data"""
    validations = {}
    scene_header = extract_scene_header(scene_text)
    
    if not scene_header:
        return {"error": "Could not parse scene header"}
    
    # Basic Information
    validations['Scene Number'] = {
        'current': str(qa_row.get('Scene #', '')),
        'correct': scene_header['scene_number'],
        'status': str(qa_row.get('Scene #', '')).strip() == scene_header['scene_number'].strip()
    }

    # Multiple Setups
    location_changes = len(re.findall(r'(INT\.|EXT\.)', scene_text))
    time_changes = len(re.findall(r'(CONTINUOUS|LATER|MOMENTS LATER)', scene_text))
    has_multiple = location_changes > 1 or time_changes > 0
    
    validations['Has Multiple Setups'] = {
        'current': str(qa_row.get('Has Multiple Setups', '')),
        'correct': 'YES' if has_multiple else 'NO',
        'status': str(qa_row.get('Has Multiple Setups', '')).upper() == ('YES' if has_multiple else 'NO')
    }

    # Scene Header
    qa_header = str(qa_row.get('Full scene header', ''))
    similarity = SequenceMatcher(None, scene_header['full_header'], qa_header).ratio()
    
    validations['Full scene header'] = {
        'current': qa_header,
        'correct': scene_header['full_header'],
        'status': similarity > 0.9  # adjust the threshold as needed
    }

    # INT/EXT Settings
    is_int = 'INT' in scene_header['int_ext']
    is_ext = 'EXT' in scene_header['int_ext']
    
    validations['Interior/Exterior'] = {
        'has_interior': {
            'current': str(qa_row.get('Has interior?', '')),
            'correct': 'YES' if is_int else 'NO',
            'status': str(qa_row.get('Has interior?', '')).upper() == ('YES' if is_int else 'NO')
        },
        'has_exterior': {
            'current': str(qa_row.get('Has exterior?', '')),
            'correct': 'YES' if is_ext else 'NO',
            'status': str(qa_row.get('Has exterior?', '')).upper() == ('YES' if is_ext else 'NO')
        }
    }

    # Characters
    scene_characters = extract_scene_characters(scene_text)
    qa_characters = [char.strip() for char in str(qa_row.get('Characters Present', '')).split(',') if char.strip()]
    
    validations['Characters'] = {
        'current': qa_characters,
        'correct': scene_characters,
        'missing': list(set(qa_characters) - set(scene_characters)),
        'extra': list(set(scene_characters) - set(qa_characters)),
        'status': set(qa_characters) == set(scene_characters)
    }

    # Content Flags
    content_flags = check_scene_content(scene_text)
    validations['Content Flags'] = {}
    
    for flag, data in content_flags.items():
        flag_key = f'Contains {flag}?'
        current_value = str(qa_row.get(flag_key, '')).upper() == 'YES'
        validations['Content Flags'][flag] = {
            'current': 'YES' if current_value else 'NO',
            'correct': 'YES' if data['status'] else 'NO',
            'status': current_value == data['status'],
            'evidence': data['evidence']
        }

    return validations

def format_validation_report(scene_number, validations):
    """Format validation results into a readable report"""
    report = f"\nSCENE {scene_number} ANALYSIS:\n"
    
    if "error" in validations:
        return report + f"Error: {validations['error']}\n"
    
    # Check if any corrections are needed
    needs_corrections = False
    for key, value in validations.items():
        if isinstance(value, dict) and 'status' in value and not value['status']:
            needs_corrections = True
            break
        if isinstance(value, dict) and 'Characters' in key and (value.get('missing') or value.get('extra')):
            needs_corrections = True
            break

    if not needs_corrections:
        report += "✅ All fields verified correct\n"
        return report

    report += "Required Updates:\n"

    # Has Multiple Setups
    if not validations['Has Multiple Setups']['status']:
        report += f"\nHas Multiple Setups\n"
        report += f"Current: {validations['Has Multiple Setups']['current']}\n"
        report += f"Should be: {validations['Has Multiple Setups']['correct']}\n"

    # Scene Header
    if not validations['Full scene header']['status']:
        report += f"\nFull scene header\n"
        report += f"Current: {validations['Full scene header']['current']}\n"
        report += f"Should be: \"{validations['Full scene header']['correct']}\"\n"

    # Interior/Exterior Settings
    if not validations['Interior/Exterior']['has_interior']['status'] or \
       not validations['Interior/Exterior']['has_exterior']['status']:
        report += "\nInterior/Exterior Settings\n"
        if not validations['Interior/Exterior']['has_interior']['status']:
            report += f"Set \"Has interior?\" to: {validations['Interior/Exterior']['has_interior']['correct']}\n"
        if not validations['Interior/Exterior']['has_exterior']['status']:
            report += f"Set \"Has exterior?\" to: {validations['Interior/Exterior']['has_exterior']['correct']}\n"

    # Characters
    if validations['Characters']['missing'] or validations['Characters']['extra']:
        report += "\nCharacters Present\n"
        if validations['Characters']['missing']:
            report += f"Add: {', '.join(validations['Characters']['missing'])}\n"
        if validations['Characters']['extra']:
            report += f"Remove: {', '.join(validations['Characters']['extra'])}\n"

    # Content Flags
    needs_flag_updates = False
    flag_updates = []
    
    for flag, data in validations['Content Flags'].items():
        if not data['status']:
            needs_flag_updates = True
            evidence = f" ({', '.join(data['evidence'])})" if data['evidence'] else ""
            flag_updates.append(f"- Contains {flag}{evidence}")

    if needs_flag_updates:
        report += "\nContent Flags (Change to YES):\n"
        report += '\n'.join(flag_updates)

    return report
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

# File upload section
with st.sidebar:
    st.header("Settings")
    script_file = st.file_uploader(
        "Upload Script PDF or CSV",
        type=["pdf", "csv"],
        help="Upload the original script PDF or CSV"
    )
    
    qa_file = st.file_uploader(
        "Upload QA Sheet",
        type=["csv"],
        help="Upload the QA sheet in CSV format"
    )

# Main content area
if script_file and qa_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Process Documents", type="primary", key="process_docs"):
            with st.spinner("Processing documents..."):
                # Check file type
                if script_file.type == "application/pdf":
                    script_text, page_mapping = process_pdf(script_file)
                elif script_file.type == "text/csv":
                    script_text = process_csv(script_file)
                    page_mapping = None
                else:
                    st.error("Unsupported file type. Please upload a PDF or CSV file.")
                    st.stop()

                if script_text:
                    try:
                        # Split script into scenes
                        scenes = split_into_scenes(script_text)
                        st.session_state.script_content = scenes
                        
                        # Process QA sheet
                        qa_data = pd.read_csv(qa_file)
                        qa_rows = qa_data.to_dict('records')
                        st.session_state.qa_content = qa_rows
                        
                        # Debug information in expander
                        with st.expander("Analysis Details"):
                            st.info(f"Found {len(scenes)} scenes")
                            st.text(f"Scene numbers: {', '.join(sorted(scenes.keys()))}")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["Scene Analysis", "Summary"])
                        
                        with tab1:
                            st.markdown("### Scene-by-Scene Analysis")
                            for qa_row in qa_rows:
                                scene_num = str(qa_row.get('Scene #', '')).strip()
                                scene_content = scenes.get(scene_num)
                                
                                # Create expander for each scene
                                with st.expander(f"Scene {scene_num}", expanded=True):
                                    if scene_content:
                                        validations = validate_scene(scene_content, qa_row)
                                        report = format_validation_report(scene_num, validations)
                                        st.markdown(report)
                                    else:
                                        st.markdown(f"⚠️ Scene not found in script")
                        
                        with tab2:
                            st.markdown("### Analysis Summary")
                            total_scenes = len(qa_rows)
                            scenes with issues = sum(1 for row in qa_rows if scenes.get(str(row.get('Scene #', '')).strip()))
                            st.metric("Total Scenes Processed", total_scenes)
                            st.metric("Scenes with Issues", scenes_with_issues)
                        
                        st.success(f"Analysis complete! Processed {len(scenes)} scenes.")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                else:
                    st.error("Error processing script file")

    with col2:
        if st.button("Reset Analysis", type="secondary"):
            st.session_state.messages = []
            st.session_state.script_content = None
            st.session_state.qa_content = None
            st.rerun()

# Chat interface for follow-up questions
st.markdown("---")
st.markdown("### Ask Questions About the Analysis")
st.markdown("You can ask specific questions about scenes or discrepancies.")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about specific scenes or issues"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Processing question..."):
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0,
                system="You are a Script QA expert. Help explain the analysis results and answer questions about specific scenes or discrepancies.",
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text

            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
