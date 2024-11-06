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
                    # Store text chunks with their page numbers
                    page_mapping[page_num] = extracted_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None
    return text, page_mapping

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
    character_pattern = r'\n([A-Z][A-Z\s]+)(?:\s*\(.*?\))?\s*\n(?=\S)'
    characters = set()
    for match in re.finditer(character_pattern, text):
        char = ' '.join(match.group(1).split())
        # Skip stage directions and common words
        if not any(word in char for word in ['SMASH', 'CUT', 'FADE', 'DISSOLVE', 'CONTINUED', 'CHYRON']):
            if not char in {'TO', 'AND', 'THE', 'WITH', 'AS', 'WE', 'SCENE', 'INT', 'EXT'}:
                characters.add(char)
    return sorted(list(characters))

def get_scene_length(text):
    """Calculate scene length in eighths"""
    # Basic estimation - can be improved based on your specific needs
    lines = len(text.split('\n'))
    eighths = max(1, round(lines / 8))  # Minimum 1/8
    return f"{eighths}/8"
def validate_scene(scene_data, qa_row, page_mapping):
    """Comprehensive scene validation"""
    validations = {}
    scene_text = scene_data.get('content', '')
    scene_info = extract_scene_info(scene_text)[0] if extract_scene_info(scene_text) else {}
    
    # Basic Information
    validations['Scene Number'] = {
        'current': str(qa_row.get('Scene #', '')),
        'correct': scene_info.get('scene_number', ''),
        'status': 'correct' if str(qa_row.get('Scene #', '')) == scene_info.get('scene_number', '') else 'incorrect'
    }

    # Multiple Setups
    location_changes = len(re.findall(r'(INT\.|EXT\.)', scene_text))
    has_multiple = location_changes > 1 or 'CONTINUOUS' in scene_text
    validations['Has Multiple Setups'] = {
        'current': str(qa_row.get('Has Multiple Setups', '')),
        'correct': 'YES' if has_multiple else 'NO',
        'status': 'correct' if (str(qa_row.get('Has Multiple Setups', '')).upper() == ('YES' if has_multiple else 'NO')) else 'incorrect'
    }

    # INT/EXT
    validations['INT/EXT'] = {
        'current': str(qa_row.get('INT/EXT', '')),
        'correct': scene_info.get('int_ext', ''),
        'status': 'correct' if str(qa_row.get('INT/EXT', '')) == scene_info.get('int_ext', '') else 'incorrect'
    }

    # Scene Header
    validations['Full scene header'] = {
        'current': str(qa_row.get('Full scene header', '')),
        'correct': scene_info.get('full_header', ''),
        'status': 'correct' if str(qa_row.get('Full scene header', '')) == scene_info.get('full_header', '') else 'incorrect'
    }

    # Interior/Exterior Settings
    is_int = 'INT' in scene_info.get('int_ext', '')
    is_ext = 'EXT' in scene_info.get('int_ext', '')
    validations['Has interior?'] = {
        'current': str(qa_row.get('Has interior?', '')),
        'correct': 'YES' if is_int else 'NO',
        'status': 'correct' if str(qa_row.get('Has interior?', '')).upper() == ('YES' if is_int else 'NO') else 'incorrect'
    }
    validations['Has exterior?'] = {
        'current': str(qa_row.get('Has exterior?', '')),
        'correct': 'YES' if is_ext else 'NO',
        'status': 'correct' if str(qa_row.get('Has exterior?', '')).upper() == ('YES' if is_ext else 'NO') else 'incorrect'
    }

    # Location and Set
    validations['Location'] = {
        'current': str(qa_row.get('Location', '')),
        'correct': scene_info.get('location', ''),
        'status': 'correct' if str(qa_row.get('Location', '')) == scene_info.get('location', '') else 'incorrect'
    }

    # Time Information
    is_night = any(word in scene_info.get('time', '').upper() for word in ['NIGHT', 'MIDDLE OF THE NIGHT'])
    validations['Time (from header)'] = {
        'current': str(qa_row.get('Time (from header)', '')),
        'correct': scene_info.get('time', ''),
        'status': 'correct' if str(qa_row.get('Time (from header)', '')) == scene_info.get('time', '') else 'incorrect'
    }
    validations['Is night?'] = {
        'current': str(qa_row.get('Is night?', '')),
        'correct': 'YES' if is_night else 'NO',
        'status': 'correct' if str(qa_row.get('Is night?', '')).upper() == ('YES' if is_night else 'NO') else 'incorrect'
    }

    # Characters
    script_characters = extract_characters(scene_text)
    qa_characters = [char.strip() for char in str(qa_row.get('Characters Present', '')).split(',') if char.strip()]
    missing_chars = set(script_characters) - set(qa_characters)
    extra_chars = set(qa_characters) - set(script_characters)
    validations['Characters Present'] = {
        'current': ', '.join(qa_characters),
        'correct': ', '.join(script_characters),
        'status': 'correct' if not (missing_chars or extra_chars) else 'incorrect',
        'missing': sorted(list(missing_chars)),
        'extra': sorted(list(extra_chars))
    }

    # Content Flags
    content_flags = {
        'Contains sex/nudity?': lambda x: any(word in x.lower() for word in ['nude', 'naked', 'sex', 'love scene', 'kiss', 'motorboat']),
        'Contains violence?': lambda x: any(word in x.lower() for word in ['kill', 'shot', 'blood', 'fight', 'punch', 'hit', 'die', 'dead']),
        'Contains profanity?': lambda x: any(word in x.lower() for word in ['fuck', 'shit', 'damn', 'hell', 'ass']),
        'Contains alcohol/drugs/smoking?': lambda x: any(word in x.lower() for word in ['drink', 'drunk', 'beer', 'wine', 'smoke', 'mai tai', 'gimlet']),
        'Contains frightening/intense moment?': lambda x: any(word in x.lower() for word in ['scream', 'terror', 'horror', 'frighten', 'intense', 'kill', 'die', 'dead'])
    }

    for flag, check_func in content_flags.items():
        script_has_content = check_func(scene_text)
        qa_has_content = str(qa_row.get(flag, '')).upper() == 'YES'
        validations[flag] = {
            'current': 'YES' if qa_has_content else 'NO',
            'correct': 'YES' if script_has_content else 'NO',
            'status': 'correct' if qa_has_content == script_has_content else 'incorrect'
        }

    return validations
def format_validation_report(scene_number, validations):
    """Format validation results into a clean, readable report"""
    report = f"\nSCENE {scene_number} ANALYSIS:\n"
    
    # Check if any corrections are needed
    needs_corrections = any(v['status'] == 'incorrect' for v in validations.values())
    if not needs_corrections:
        report += "✓ All fields verified correct\n"
        return report

    report += "Required Updates:\n"
    
    # Basic Information
    if validations['Scene Number']['status'] == 'incorrect':
        report += f"1. Scene Number\n   - Should be: {validations['Scene Number']['correct']}\n"

    # Multiple Setups
    if validations['Has Multiple Setups']['status'] == 'incorrect':
        report += f"2. Has Multiple Setups\n   - Current: {validations['Has Multiple Setups']['current']}\n   - Should be: {validations['Has Multiple Setups']['correct']}\n"

    # INT/EXT and Header
    if validations['INT/EXT']['status'] == 'incorrect':
        report += f"3. INT/EXT\n   - Current: {validations['INT/EXT']['current']}\n   - Should be: {validations['INT/EXT']['correct']}\n"

    if validations['Full scene header']['status'] == 'incorrect':
        report += f"4. Full scene header\n   - Current: {validations['Full scene header']['current']}\n   - Should be: \"{validations['Full scene header']['correct']}\"\n"

    # Interior/Exterior Settings
    if validations['Has interior?']['status'] == 'incorrect' or validations['Has exterior?']['status'] == 'incorrect':
        report += "5. Interior/Exterior Settings\n"
        if validations['Has interior?']['status'] == 'incorrect':
            report += f"   - Set \"Has interior?\" to: {validations['Has interior?']['correct']}\n"
        if validations['Has exterior?']['status'] == 'incorrect':
            report += f"   - Set \"Has exterior?\" to: {validations['Has exterior?']['correct']}\n"

    # Characters
    if validations['Characters Present']['status'] == 'incorrect':
        report += "6. Characters Present\n"
        if validations['Characters Present']['missing']:
            report += f"   - Add: {', '.join(validations['Characters Present']['missing'])}\n"
        if validations['Characters Present']['extra']:
            report += f"   - Remove: {', '.join(validations['Characters Present']['extra'])}\n"

    # Content Flags
    content_flags_need_update = False
    content_flag_updates = []
    for flag in ['Contains sex/nudity?', 'Contains violence?', 'Contains profanity?', 
                 'Contains alcohol/drugs/smoking?', 'Contains frightening/intense moment?']:
        if validations[flag]['status'] == 'incorrect':
            content_flags_need_update = True
            content_flag_updates.append(f"   - {flag.replace('?', '')}")

    if content_flags_need_update:
        report += "\n7. Content Flags (Change to YES):\n"
        report += '\n'.join(content_flag_updates) + '\n'

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
            script_text, page_mapping = process_pdf(script_file)
            if script_text:
                st.session_state.script_content = {'content': script_text, 'pages': page_mapping}
                
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
                    validations = validate_scene(st.session_state.script_content, qa_row, 
                                              st.session_state.script_content.get('pages', {}))
                    report = format_validation_report(scene_num, validations)
                    full_report.append(report)
                
                response = "\n".join(full_report)
            else:
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
