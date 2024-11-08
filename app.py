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
from collections import defaultdict

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
    'Contains alcohol / drugs / smoking?': ['drink', 'drunk', 'beer', 'wine', 'liquor', 'gimlet', 'mai tai', 'smoking', 'cigarette', 'alcohol', 'booze', 'weed', 'bottle', 'bar'],
    'Contains a frightening / intense moment?': ['scream', 'terror', 'horror', 'frighten', 'intense', 'violent', 'blood', 'kill', 'death', 'panic', 'fear', 'traumatic', 'shock', 'disturbing', 'graphic', 'gory', 'brutal']
}

# Content flag column mappings (keys must match CONTENT_FLAGS exactly)
FLAG_TO_COLUMN = {
    'Contains sex / nudity?': 'Contains sex / nudity? ',
    'Contains violence?': 'Contains violence? ',
    'Contains profanity?': 'Contains profanity? ',
    'Contains alcohol / drugs / smoking?': 'Contains alcohol / drugs / smoking? ',
    'Contains a frightening / intense moment?': 'Contains a frightening / intense moment? '
}

def check_content(text, keywords):
    """Check for actual content matches with context"""
    text_lower = text.lower()
    matches = []
    for keyword in keywords:
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, text_lower):
            start = max(0, text_lower.find(keyword.lower()) - 100)
            end = min(len(text_lower), text_lower.find(keyword.lower()) + len(keyword) + 100)
            context = text_lower[start:end]
            if not any(x in context for x in ['(cont', '(continued', 'scene heading', 'title card']):
                matches.append(keyword)
    return bool(matches), matches

def normalize_header(header):
    """Normalize scene header for comparison"""
    header = re.sub(r'\s+', ' ', header).strip()
    header = re.sub(r'^\d+[A-Z]?\s*', '', header)
    header = re.sub(r'\s*\([^)]+\)\s*$', '', header)
    header = header.replace('INT ', 'INT. ').replace('EXT ', 'EXT. ')
    header = header.replace('INT./', 'INT./').replace('EXT./', 'EXT./')
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
    """Enhanced scene header extraction with strict validation"""
    scene_pattern = r'(?m)^(\d+[A-Z]?)\s+((?:INT\.|EXT\.)|(?:INT\./EXT\.)|(?:I/E\.?))\s+(.*?)\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT|PRE-DAWN|DUSK|DAWN|EVENING|MORNING|AFTERNOON)(?:\s*\([^)]+\))?'
    match = re.search(scene_pattern, text)
    if match:
        header_text = match.group(0).split('\n')[0].strip()
        location = match.group(3).strip()
        if ' - ' in location:
            locations = [loc.strip() for loc in location.split(' - ')]
        if len(header_text) > 200:
            return None
        return {
            'scene_number': match.group(1),
            'int_ext': match.group(2).strip('.'),
            'location': locations,
            'time': match.group(4).strip(),
            'full_header': header_text,
            'raw_match': match.group(0)
        }
    return None

def process_with_error_recovery(script_file, qa_file):
    """
    Process script and QA files with error handling and recovery
    Returns: (script_scenes, qa_rows, warnings, errors)
    """
    warnings = []
    errors = []
    script_scenes = {}
    qa_rows = []
    try:
        # Process script PDF
        script_text, page_mapping = process_pdf(script_file)
        if not script_text:
            errors.append("Failed to extract text from script PDF")
            return None, None, warnings, errors
            
        # Extract scenes from script
        script_scenes = split_into_scenes(script_text)
        if not script_scenes:
            errors.append("No valid scenes found in script")
            return None, None, warnings, errors

        # Process QA sheet
        try:
            qa_data = pd.read_csv(qa_file)
            validate_qa_sheet(qa_data)
            qa_rows = qa_data.to_dict('records')
        except Exception as e:
            errors.append(f"Error processing QA sheet: {str(e)}")
            return None, None, warnings, errors

        # Basic validation
        if len(script_scenes) == 0:
            warnings.append("No scenes found in script")
        if len(qa_rows) == 0:
            warnings.append("No rows found in QA sheet")

        return script_scenes, qa_rows, warnings, errors

    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return None, None, warnings, errors

def split_into_scenes(script_text):
    """Split script text into individual scenes"""
    scene_pattern = r'(\d+[A-Z]?)\s+((?:INT\.|EXT\.)|(?:INT\./EXT\.)|(?:I/E\.?))\s+(.*?)\s+-\s+(DAY|NIGHT|CONTINUOUS|LATER|MOMENTS LATER|MIDDLE OF THE NIGHT|PRE-DAWN|DUSK|DAWN|EVENING|MORNING|AFTERNOON)'
    scenes = {}
    omitted_pattern = r'(\d+[A-Z]?)\s+(?:OMITTED)'
    omitted_matches = re.finditer(omitted_pattern, script_text)
    for match in omitted_matches:
        scene_num = match.group(1)
        scenes[scene_num] = "OMITTED"
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
    """Enhanced character extraction with contextual validation"""
    characters = set()
    character_groups = {
        'MOVIEGOERS': ['MOVIEGOERS of all ages', 'MOVIEGOERS'],
        'MOURNERS': ['MOURNERS', 'GROUP OF MOURNERS'],
        'COPS': ['COPS', 'POLICE OFFICERS', 'OFFICERS']
    }
    character_pattern = r'(?:^|\n)([A-Z][A-Z\s\'\-]+)(?:\s*\((?:[^)]+)\))?\s*\n'
    for match in re.finditer(character_pattern, text, re.MULTILINE):
        name = match.group(1).strip()
        if (len(name.split()) <= 3 and
            not any(fp in name for fp in ['FP']) and
            not re.search(r'\d', name) and
            len(name) >= 2 and
            len(name) <= 30):
            characters.add(name)
    return sorted(list(characters))

def calculate_scene_length(text):
    """Enhanced scene length calculation with more accurate line counting"""
    def count_action_lines(lines):
        count = 0
        for line in lines:
            if not re.match(r'^CONTINUED:|^CONT\'D:', line):
                if not line.strip().startswith('(') and not re.match(r'^[A-Z\s]+$', line):
                    count += 1
        return count
    lines = [l for l in text.split('\n') if l.strip() and not any(x in l for x in ['FADE OUT', 'CUT TO:', 'DISSOLVE TO:'])]
    action_lines = count_action_lines(lines)
    total_lines = action_lines + (len(lines) - action_lines) * 0.75
    eighths = max(1, round((total_lines / 8) * 8) / 8)
    return str(int(eighths))

def validate_content_flags(scene_text, qa_row):
    """Enhanced content flag validation with context awareness"""
    validations = {}
    for flag, keywords in CONTENT_FLAGS.items():
        qa_value = str(qa_row.get(FLAG_TO_COLUMN[flag], '')).upper()
        has_content = False
        evidence = []
        for keyword in keywords:
            matches = re.finditer(rf'\b{re.escape(keyword.lower())}\b', scene_text.lower())
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(scene_text), match.end() + 100)
                context = scene_text[start:end]
                if not re.search(r'\([^)]*' + re.escape(keyword.lower()) + r'[^)]*\)', context):
                    has_content = True
                    evidence.append({
                        'keyword': keyword,
                        'context': context.strip()
                    })
        if qa_value != ('YES' if has_content else 'NO'):
            validations[flag] = {
                'current': qa_value,
                'correct': 'YES' if has_content else 'NO',
                'status': False,
                'evidence': evidence
            }
    return validations

def validate_scene(scene_text, qa_row):
    """Validate scene against QA sheet"""
    validations = {}
    if scene_text == "OMITTED":
        return {"status": "OMITTED"}
    try:
        scene_header = extract_scene_header(scene_text)
        if not scene_header:
            return {"error": "Could not parse scene header"}
        qa_header = str(qa_row.get('Full scene header (excluding scene number)', '')).strip()
        script_header = scene_header['full_header'].replace(f"{scene_header['scene_number']} ", '').strip()
        script_header = re.sub(r'\s*-\s*(?:CONTINUOUS|MOMENTS LATER|LATER)\s*(?:-|$)', '', script_header)
        if script_header != qa_header:
            validations['Full scene header'] = {
                'current': qa_header,
                'correct': script_header,
                'status': False
            }
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
        script_characters = set(extract_scene_characters(scene_text))
        qa_characters = {char.strip() for char in str(qa_row.get('Characters Present in Scene', '')).split(',') if char.strip()}
        script_characters = {re.sub(r'\s+', ' ', name).strip() for name in script_characters}
        qa_characters = {re.sub(r'\s+', ' ', name).strip() for name in qa_characters}
        missing_chars = qa_characters - script_characters
        extra_chars = script_characters - qa_characters
        if missing_chars or extra_chars:
            validations['Characters'] = {
                'missing': sorted(list(missing_chars)),
                'extra': sorted(list(extra_chars)),
                'status': False
            }
        scene_length = str(qa_row.get('Scene length (in eighths)', '')).strip()
        script_length = calculate_scene_length(scene_text)
        if scene_length != script_length:
            validations['Scene length'] = {
                'current': scene_length,
                'correct': script_length,
                'status': False
            }
        flag_validations = validate_content_flags(scene_text, qa_row)
        if flag_validations:
            validations['Content Flags'] = flag_validations
        return validations
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}

def format_validation_report(scene_number, validations):
    """Format validation results to show only actual discrepancies"""
    report = f"\nSCENE {scene_number} ANALYSIS:\n"
    if "error" in validations:
        return report + f"Error: {validations['error']}\n"
    if not validations:
        report += "✅ All fields verified correct\n"
        return report
    report += "Required Updates:\n"
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

def generate_validation_summary(validations_by_scene):
    """Generate detailed validation summary with statistics"""
    summary = {
        'total_scenes': len(validations_by_scene),
        'scenes_with_issues': 0,
        'issues_by_type': defaultdict(int),
        'critical_issues': [],
        'validation_evidence': {}
    }
    for scene_num, validations in validations_by_scene.items():
        if validations and not validations.get('status') == 'OMITTED':
            summary['scenes_with_issues'] += 1
            for issue_type in validations:
                if issue_type != 'error':
                    summary['issues_by_type'][issue_type] += 1
                    if issue_type in ['Characters', 'Full scene header']:
                        summary['critical_issues'].append({
                            'scene': scene_num,
                            'type': issue_type,
                            'details': validations[issue_type]
                        })
    return summary

# Modify the validate_qa_sheet function to handle variations in column names
def validate_qa_sheet(qa_data):
    """Validate QA sheet structure and content with flexible column matching"""
    column_mappings = {
        'Scene #': ['Scene #', 'Scene Number', 'Scene'],
        'Full scene header (excluding scene number)': [
            'Full scene header (excluding scene number)',
            'Scene Header',
            'Full Header'
        ],
        'Characters Present in Scene': [
            'Characters Present in Scene',
            'Characters',
            'Characters Present'
        ],
        'Scene length (in eighths)': [
            'Scene length (in eighths)',
            'Scene length\n(in eighths)',
            'Scene Length',
            'Length in eighths'
        ],
        'Has Multiple Setups': ['Has Multiple Setups', 'Multiple Setups'],
        'Has interior?': ['Has interior?', 'Interior'],
        'Has exterior? ': ['Has exterior? ', 'Exterior'],
        'Contains sex / nudity? ': ['Contains sex / nudity? ', 'Sex/Nudity'],
        'Contains violence? ': [
            'Contains violence? ',
            'Violence',
            'Contains violence'
        ],
        'Contains profanity? ': ['Contains profanity? ', 'Profanity'],
        'Contains alcohol / drugs / smoking? ': [
            'Contains alcohol / drugs / smoking? ',
            'Alcohol/Drugs/Smoking'
        ],
        'Contains a frightening / intense moment? ': [
            'Contains a frightening / intense moment? ',
            'Frightening/Intense',
            'Frightening/Intense Moment'
        ]
    }

    missing_columns = []
    for required_col, variations in column_mappings.items():
        if not any(var in qa_data.columns for var in variations):
            missing_columns.append(required_col)
    if missing_columns:
        raise ValueError(f"Missing required columns in QA sheet: {missing_columns}")
    # Rename columns to standard names if needed
    column_rename = {}
    for standard_name, variations in column_mappings.items():
        for var in variations:
            if var in qa_data.columns:
                column_rename[var] = standard_name
                break
    if column_rename:
        qa_data.rename(columns=column_rename, inplace=True)
    # Validate scene numbers
    for idx, row in qa_data.iterrows():
        scene_num = str(row['Scene #']).strip()
        if not re.match(r'^\d+[A-Z]?$', scene_num):
            raise ValueError(f"Invalid scene number format in row {idx + 1}: {scene_num}")

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
                script_scenes, qa_rows, warnings, errors = process_with_error_recovery(script_file, qa_file)
                if script_scenes and qa_rows:
                    st.session_state.script_content = script_scenes
                    st.session_state.qa_content = qa_rows
                    with st.expander("Analysis Details"):
                        st.info(f"Found {len(script_scenes)} scenes")
                        st.text(f"Scene numbers: {', '.join(sorted(script_scenes.keys()))}")
                    tab1, tab2 = st.tabs(["Scene Analysis", "Summary"])
                    with tab1:
                        st.markdown("### Scene-by-Scene Analysis")
                        validations_by_scene = {}
                        for qa_row in qa_rows:
                            scene_num = str(qa_row.get('Scene #', '')).strip()
                            scene_content = script_scenes.get(scene_num)
                            with st.expander(f"Scene {scene_num}", expanded=True):
                                try:
                                    if scene_content:
                                        validations = validate_scene(scene_content, qa_row)
                                        validations_by_scene[scene_num] = validations
                                        if validations:
                                            report = format_validation_report(scene_num, validations)
                                            st.markdown(report)
                                        else:
                                            st.markdown(f"⚠️ Error validating scene {scene_num}")
                                    else:
                                        st.markdown(f"⚠️ Scene not found in script")
                                except Exception as e:
                                    st.markdown(f"⚠️ Error processing scene {scene_num}: {str(e)}")
                    with tab2:
                        st.markdown("### Analysis Summary")
                        summary = generate_validation_summary(validations_by_scene)
                        st.metric("Total Scenes", summary['total_scenes'])
                        st.metric("Scenes Needing Updates", summary['scenes_with_issues'])
                        st.success(f"Analysis complete! Processed {len(script_scenes)} scenes.")
                        if warnings:
                            st.warning("Warnings:")
                            for warning in warnings:
                                st.write(warning)
                        if errors:
                            st.error("Errors:")
                            for error in errors:
                                st.write(error)
                else:
                    st.error("Error processing script PDF or QA sheet")
                    if errors:
                        st.write("Errors:")
                        for error in errors:
                            st.write(error)
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
