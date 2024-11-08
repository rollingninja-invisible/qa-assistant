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

# Load environment variables
load_dotenv()

# Get API key and initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    st.error("No Anthropic API key found. Please make sure ANTHROPIC_API_KEY is set in your .env file.")
    st.stop()

try:
    client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing Anthropic client: {str(e)}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Script QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Content flag definitions
CONTENT_FLAGS = {
    'Contains sex/nudity?': ['nude', 'naked', 'sex', 'breast', 'tit', 'motorboat', 'love scene', 'kiss', 'jiggle', 'buxom', 'breasts', 'lingerie', 'undress', 'strip', 'topless', 'intimate'],
    'Contains violence?': ['kill', 'shot', 'blood', 'fight', 'punch', 'hit', 'slaughter', 'death', 'die', 'gun', 'shoot', 'bullet', 'stab', 'wound', 'dead', 'murder', 'strangle', 'choke', 'slash', 'knife', 'weapon', 'assault'],
    'Contains profanity?': ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard', 'kike', 'goddamn', 'christ', 'jesus', 'crap', 'piss', 'cock', 'dick', 'whore', 'slut'],
    'Contains alcohol/drugs/smoking?': ['drink', 'drunk', 'beer', 'wine', 'liquor', 'gimlet', 'mai tai', 'smoking', 'cigarette', 'alcohol', 'booze', 'weed', 'joint', 'pill', 'needle', 'inject', 'high', 'bottle', 'bar'],
    'Contains a frightening/intense moment?': ['scream', 'terror', 'horror', 'frighten', 'intense', 'violent', 'blood', 'kill', 'death', 'panic', 'fear', 'traumatic', 'shock', 'disturbing', 'graphic', 'gory', 'brutal']
}

# Content flag column mappings
FLAG_TO_COLUMN = {
    'Contains sex/nudity?': 'Contains sex / nudity?',
    'Contains violence?': 'Contains violence?',
    'Contains profanity?': 'Contains profanity?',
    'Contains alcohol/drugs/smoking?': 'Contains alcohol / drugs / smoking?',
    'Contains a frightening/intense moment?': 'Contains a frightening / intense moment?'
}

def normalize_header(header):
    """Normalize scene header for comparison"""
    # Remove script formatting artifacts
    header = re.sub(r'\s+', ' ', header).strip()
    # Remove scene numbers
    header = re.sub(r'^\d+[A-Z]?\s*', '', header)
    # Standardize INT./EXT.
    header = header.replace('INT ', 'INT. ').replace('EXT ', 'EXT. ')
    return header

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

def extract_scene_header(text):
    scene_pattern = r'(\d+[A-Z]?)\s+((?:INT\.|EXT\.)|(?:INT\./EXT\.)|(?:I/E\.?))\s+(.*?)\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT|PRE-DAWN|DUSK|DAWN|EVENING|MORNING|AFTERNOON)'
    match = re.search(scene_pattern, text)
    if match:
        header_text = match.group(0).split('\n')[0].strip()  # Get only first line
        return {
            'scene_number': match.group(1),
            'int_ext': match.group(2).strip('.'),
            'location': match.group(3).strip(),
            'time': match.group(4),
            'full_header': header_text
        }
    return None

def split_into_scenes(script_text):
    """Split script text into individual scenes"""
    scene_pattern = r'(\d+[A-Z]?)\s+((?:INT\.|EXT\.)|(?:INT\./EXT\.)|(?:I/E\.?))\s+(.*?)\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT|PRE-DAWN|DUSK|DAWN|EVENING|MORNING|AFTERNOON)'
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
        
        if len(scene_text) > 50:  # Minimum length to be considered a scene
            scenes[scene_num] = scene_text
    
    return scenes

def extract_scene_characters(text):
    characters = set()
    lines = text.split('\n')
    
    # Process character names in parentheses
    char_pattern = r'([A-Z][A-Z\s\'\-]+)(?:\s*\([^)]+\))'
    for match in re.finditer(char_pattern, text):
        name = match.group(1).strip()
        if len(name.split()) <= 3:  # Most character names are 1-3 words
            characters.add(name)
    
    # Process dialogue speakers
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(r'^[A-Z][A-Z\s\'\-]+$', line):
            name = line.strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # Only add if followed by dialogue or parenthetical
            if (next_line.startswith('(') or 
                (not next_line.isupper() and next_line and 
                 not any(x in name for x in ['CONTINUED', 'CONT']))):
                characters.add(name)
    
    return sorted(list(characters))

def calculate_scene_length(text):
    """Calculate scene length in eighths"""
    content_lines = [line for line in text.split('\n') 
                    if line.strip() and not any(x in line for x in 
                    ['INT.', 'EXT.', 'CONTINUED', 'LEVIATHAN', 'CHYRON', '--', '(CONT', 'SCENE', 'PINK'])]
    
    # Count meaningful lines (dialogue, action, etc.)
    meaningful_lines = len([line for line in content_lines 
                          if not (line.isupper() and len(line) < 50) and  # Skip character cues
                          not line.startswith('(') and  # Skip parentheticals
                          not any(x in line for x in ['Draft', 'Revised', 'Page'])])  # Skip script metadata
    
    # Return just the number
    return str(max(1, round(meaningful_lines / 8)))

def validate_scene(scene_text, qa_row):
    validations = {}
    scene_header = extract_scene_header(scene_text)
    
    if not scene_header:
        return {"error": "Could not parse scene header"}
    
    # Compare headers exactly as they appear in QA sheet
    qa_header = str(qa_row.get('Full scene header (excluding scene number)', '')).strip()  
    script_header = scene_header['full_header'].replace(f"{scene_header['scene_number']} ", '').strip()  # Add space after number

    if script_header != qa_header:
        validations['Full scene header'] = {
            'current': qa_header,
            'correct': script_header,
            'status': False
        }

    # Multiple Setups - verify actual location/time changes
    location_changes = len(set(re.findall(r'(?:INT\.|EXT\.)\s+([^-]+)', scene_text)))
    time_changes = len(set(re.findall(r'-\s+(CONTINUOUS|LATER|MOMENTS LATER)', scene_text)))
    has_multiple = location_changes > 1 or time_changes > 0
    qa_multiple = str(qa_row.get('Has Multiple Setups', '')).upper() == 'YES'
    
    if has_multiple != qa_multiple:
        validations['Has Multiple Setups'] = {
            'current': str(qa_row.get('Has Multiple Setups', '')),
            'correct': 'YES' if has_multiple else 'NO',
            'status': False
        }

    # INT/EXT Settings
    is_int = 'INT' in scene_header['int_ext']
    is_ext = 'EXT' in scene_header['int_ext']
    qa_int = str(qa_row.get('Has interior?', '')).upper() == 'YES'
    qa_ext = str(qa_row.get('Has exterior?', '')).upper() == 'YES'
    
    int_ext_validations = {}
    if qa_int != is_int:
        int_ext_validations['has_interior'] = {
            'current': 'YES' if qa_int else 'NO',
            'correct': 'YES' if is_int else 'NO',
            'status': False
        }
    if qa_ext != is_ext:
        int_ext_validations['has_exterior'] = {
            'current': 'YES' if qa_ext else 'NO',
            'correct': 'YES' if is_ext else 'NO',
            'status': False
        }
    if int_ext_validations:
        validations['Interior/Exterior'] = int_ext_validations

    # Characters - strict comparison of actual characters
    script_characters = set(extract_scene_characters(scene_text))
    qa_characters = {char.strip() for char in str(qa_row.get('Characters Present in Scene', '')).split(',') if char.strip()}  # Update column name

    
   # Clean up character names before comparison
    script_characters = {name.strip().replace('  ', ' ') for name in script_characters}
    qa_characters = {name.strip().replace('  ', ' ') for name in qa_characters}
    
    # Only report if there are actual differences
    missing_chars = script_characters - qa_characters
    extra_chars = qa_characters - script_characters
    
    if missing_chars or extra_chars:
        validations['Characters'] = {
            'missing': sorted(list(missing_chars)),
            'extra': sorted(list(extra_chars)),
            'status': False
        }
   
    scene_length = str(qa_row.get('Scene length\n(in eighths)', '')).strip()  # Update column name
    script_length = calculate_scene_length(scene_text)
    if scene_length != script_length:
        validations['Scene length'] = {
            'current': scene_length,
            'correct': script_length,
            'status': False
        }   
   
    # Content Flags - careful validation of actual content
    def check_content(text, keywords):
        """Check for actual content matches, not just keyword presence"""
        text_lower = text.lower()
        matches = []
        for keyword in keywords:
            if keyword in text_lower:
                # Verify it's not part of a larger word
                if re.search(rf'\b{keyword}\b', text_lower):
                    matches.append(keyword)
        return bool(matches), matches

    # Content Flags validation
    flag_validations = {}
    for flag, keywords in CONTENT_FLAGS.items():
        has_content, evidence = check_content(scene_text, keywords)
        qa_value = str(qa_row.get(FLAG_TO_COLUMN[flag], '')).upper()  # Use mapping
        
        # Only show discrepancy if QA doesn't match actual content
        if qa_value == 'YES' and not has_content:
            flag_validations[flag] = {
                'current': qa_value,
                'correct': 'NO',
                'status': False,
                'evidence': []
            }
        elif qa_value != 'YES' and has_content:
            flag_validations[flag] = {
                'current': qa_value,
                'correct': 'YES',
                'status': False,
                'evidence': evidence
            }

    if flag_validations:
        validations['Content Flags'] = flag_validations
    return validations  
        
def format_validation_report(scene_number, validations):
    """Format validation results to show only actual discrepancies"""
    report = f"\nSCENE {scene_number} ANALYSIS:\n"
    
    if "error" in validations:
        return report + f"Error: {validations['error']}\n"
    
    if not validations:
        report += "✅ All fields verified correct\n"
        return report

    report += "Required Updates:\n"
    
    # Sort validations by field type
    if 'Full scene header' in validations:
        report += f"\nFull scene header\n"
        report += f"Current: {validations['Full scene header']['current']}\n"
        report += f"Should be: {validations['Full scene header']['correct']}\n"

    if 'Has Multiple Setups' in validations:
        report += f"\nHas Multiple Setups\n"
        report += f"Current: {validations['Has Multiple Setups']['current']}\n"
        report += f"Should be: {validations['Has Multiple Setups']['correct']}\n"

    if 'Interior/Exterior' in validations:
        report += "\nInterior/Exterior Settings\n"
        if 'has_interior' in validations['Interior/Exterior']:
            report += f"Set \"Has interior?\" to: {validations['Interior/Exterior']['has_interior']['correct']}\n"
        if 'has_exterior' in validations['Interior/Exterior']:
            report += f"Set \"Has exterior?\" to: {validations['Interior/Exterior']['has_exterior']['correct']}\n"

    if 'Scene length' in validations:
        report += f"\nScene length\n"
        report += f"Current: {validations['Scene length']['current']}\n"
        report += f"Should be: {validations['Scene length']['correct']}\n"
        
    if 'Characters' in validations:
        if validations['Characters']['missing'] or validations['Characters']['extra']:
            report += "\nCharacters Present\n"
            if validations['Characters']['missing']:
                report += f"Add: {', '.join(validations['Characters']['missing'])}\n"
            if validations['Characters']['extra']:
                report += f"Remove: {', '.join(validations['Characters']['extra'])}\n"

    if 'Content Flags' in validations:
        report += "\nContent Flags:\n"
        for flag, data in validations['Content Flags'].items():
            if data['status'] is False:
                if data['correct'] == 'YES':
                    report += f"- Change {flag.replace('?', '')} to YES ({', '.join(data['evidence'])})\n"
                else:
                    report += f"- Change {flag.replace('?', '')} to NO\n"

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
        "Upload Script PDF",
        type=["pdf"],
        help="Upload the original script PDF"
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
                # Process script
                script_text, page_mapping = process_pdf(script_file)
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
                            scenes_with_issues = sum(1 for row in qa_rows 
                                                   if scenes.get(str(row.get('Scene #', '')).strip()) 
                                                   and validate_scene(scenes.get(str(row.get('Scene #', '')).strip()), row))
                            st.metric("Total Scenes", total_scenes)
                            st.metric("Scenes Needing Updates", scenes_with_issues)
                        
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
