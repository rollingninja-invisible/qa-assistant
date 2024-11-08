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
    'Contains sex / nudity?': ['nude', 'naked', 'sex', 'breast', 'tit', 'motorboat', 'love scene', 'kiss', 'jiggle', 'buxom', 'breasts', 'lingerie', 'undress', 'strip', 'topless', 'intimate'],
    'Contains violence?': ['kill', 'shot', 'blood', 'fight', 'punch', 'hit', 'slaughter', 'death', 'die', 'gun', 'shoot', 'bullet', 'stab', 'wound', 'dead', 'murder', 'strangle', 'choke', 'slash', 'knife', 'weapon', 'assault'],
    'Contains profanity?': ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard', 'kike', 'goddamn', 'christ', 'jesus', 'crap', 'piss', 'cock', 'dick', 'whore', 'slut'],
    'Contains alcohol / drugs / smoking?': ['drink', 'drunk', 'beer', 'wine', 'liquor', 'gimlet', 'mai tai', 'smoking', 'cigarette', 'alcohol', 'booze', 'weed', 'joint', 'pill', 'needle', 'inject', 'high', 'bottle', 'bar'],
    'Contains a frightening / intense moment?': ['scream', 'terror', 'horror', 'frighten', 'intense', 'violent', 'blood', 'kill', 'death', 'panic', 'fear', 'traumatic', 'shock', 'disturbing', 'graphic', 'gory', 'brutal']
}

# Content flag column mappings (keys must match CONTENT_FLAGS exactly)
FLAG_TO_COLUMN = {
    'Contains sex / nudity?': 'Contains sex / nudity?',
    'Contains violence?': 'Contains violence?',
    'Contains profanity?': 'Contains profanity?',
    'Contains alcohol / drugs / smoking?': 'Contains alcohol / drugs / smoking?',
    'Contains a frightening / intense moment?': 'Contains a frightening / intense moment?'
}

def normalize_header(header):
    """Normalize scene header for comparison"""
    # Remove script formatting artifacts
    header = re.sub(r'\s+', ' ', header).strip()
    
    # Remove scene numbers
    header = re.sub(r'^\d+[A-Z]?\s*', '', header)
    
    # Remove trailing qualifiers in parentheses
    header = re.sub(r'\s*\([^)]+\)\s*$', '', header)
    
    # Standardize INT./EXT.
    header = header.replace('INT ', 'INT. ').replace('EXT ', 'EXT. ')
    header = header.replace('INT./', 'INT./').replace('EXT./', 'EXT./')
    
    # Remove extra time qualifiers
    header = re.sub(r'\s*-\s*(?:CONTINUOUS|MOMENTS LATER|LATER)\s*(?:-|$)', '', header)
    
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
    
    # First check for OMITTED scenes
    omitted_pattern = r'(\d+[A-Z]?)\s+(?:OMITTED)'
    omitted_matches = re.finditer(omitted_pattern, script_text)
    for match in omitted_matches:
        scene_num = match.group(1)
        scenes[scene_num] = "OMITTED"
    
    # Then process regular scenes
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
    """Extract character names from script text"""
    characters = set()
    
    # Words that should not be considered character names
    excluded_words = {
        # Scene elements
        'INT', 'EXT', 'CONTINUED', 'CONT', 'SCENE', 'CHYRON',
        'CUT', 'FADE', 'DISSOLVE', 'SMASH', 'FLASHBACK',
        # Technical terms
        'VOICE', 'O.S.', 'V.O.', 'SUBTITLE', 'SERIES',
        # Directions
        'ANGLE', 'CAMERA', 'POV', 'VIEW', 'CLOSER', 'BACK',
        # Scene descriptions
        'LATER', 'NOW', 'THEN', 'WHEN', 'SUDDENLY',
        # Places
        'BEDROOM', 'KITCHEN', 'HALLWAY', 'ROOM', 'OFFICE',
        # Objects
        'TAPE', 'PHONE', 'BOOK', 'CHAIR', 'TABLE', 'DOOR',
        # Actions
        'SHOOTS', 'FIRES', 'FALLS', 'MOVES', 'LOOKS'
    }
    
    def is_valid_character(name):
        """Check if a name is a valid character name"""
        if not name or len(name.split()) > 3:
            return False
        if any(word in name for word in excluded_words):
            return False
        if name.endswith('S HOME'):
            return False
        if re.search(r'\d', name):  # Contains numbers
            return False
        if any(x in name for x in ['CONT', 'VOICE', 'O.S.', 'V.O.']):
            return False
        return True

    # Extract names from dialogue headers
    dialogue_pattern = r'(?:^|\n)([A-Z][A-Z\s\'\-]+)(?=\s*[\n\(])'
    for match in re.finditer(dialogue_pattern, text, re.MULTILINE):
        name = match.group(1).strip()
        if is_valid_character(name):
            characters.add(name)
    
    # Extract names from character introductions
    intro_pattern = r'([A-Z][A-Z\s\'\-]+?)(?:\s*\([^)]+\))'
    for match in re.finditer(intro_pattern, text):
        name = match.group(1).strip()
        if is_valid_character(name):
            characters.add(name)
            
    # Clean up character names
    clean_characters = set()
    for name in characters:
        # Remove multiple spaces
        name = ' '.join(name.split())
        # Remove trailing 'THE' or leading articles
        name = re.sub(r'^(?:THE|A|AN)\s+', '', name)
        name = re.sub(r'\s+(?:THE|A|AN)$', '', name)
        if is_valid_character(name):
            clean_characters.add(name)
            
    return sorted(list(clean_characters))

def calculate_scene_length(text):
    """Calculate scene length in eighths based on QA sheet format"""
    content_lines = [line for line in text.split('\n') 
                    if line.strip() and not any(x in line for x in 
                    ['INT.', 'EXT.', 'CONTINUED:', 'CONT:', '--', 'CHYRON:', 
                     'CUT TO:', 'FADE TO:', 'Draft', 'Page', 'TITLE:', 'OMITTED'])]
    
    meaningful_lines = len([line for line in content_lines 
                          if not line.strip().startswith('(') and
                          not (line.isupper() and len(line) < 50) and
                          not any(x in line for x in ['CONTINUED', 'CONT', 'LEVIATHAN'])])
    
    # Minimum length of 1/8 for very short scenes
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
   
def check_content(text, keywords):
    """Check for actual content matches with context"""
    text_lower = text.lower()
    matches = []
    
    for keyword in keywords:
        # Use word boundaries for exact matches
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, text_lower):
            # Get surrounding context to verify relevance
            start = max(0, text_lower.find(keyword.lower()) - 50)
            end = min(len(text_lower), text_lower.find(keyword.lower()) + len(keyword) + 50)
            context = text_lower[start:end]
            
            # Exclude matches in stage directions or technical notes
            if not any(x in context for x in ['(cont', '(continued', 'scene heading', 'title card']):
                matches.append(keyword)
                
    return bool(matches), matches

     # Content flag column mappings
    flag_to_column = {
        'Contains sex/nudity?': 'Contains sex / nudity?',
        'Contains violence?': 'Contains violence?',
        'Contains profanity?': 'Contains profanity?',
        'Contains alcohol/drugs/smoking?': 'Contains alcohol / drugs / smoking?',
        'Contains a frightening/intense moment?': 'Contains a frightening / intense moment?'
    }

        # Content Flags validation
    flag_validations = {}
    for flag in CONTENT_FLAGS.keys():  # Use the keys from CONTENT_FLAGS
        qa_value = str(qa_row.get(FLAG_TO_COLUMN[flag], '')).upper()
        has_content, evidence = check_content(scene_text, CONTENT_FLAGS[flag])
        
        if qa_value != ('YES' if has_content else 'NO'):
            flag_validations[flag] = {
                'current': qa_value,
                'correct': 'YES' if has_content else 'NO',
                'status': False,
                'evidence': evidence if has_content else []
            }
    
    if flag_validations:
        validations['Content Flags'] = flag_validations
    return validations  # This line was missing
        
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
