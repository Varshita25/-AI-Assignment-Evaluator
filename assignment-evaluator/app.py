import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import os
from datetime import datetime
import zipfile
import tempfile
import shutil
from pathlib import Path
import nbformat
from docx import Document
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Assignment Evaluator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Azure OpenAI Configuration
ENDPOINT = "https://varshitharesource.services.ai.azure.com/openai/v1/"
MODEL_NAME = "gpt-oss-120b"
DEPLOYMENT_NAME = "gpt-oss-120b"

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

def configure_azure_openai(api_key):
    """Configure Azure OpenAI with the provided key"""
    try:
        client = OpenAI(
            base_url=ENDPOINT,
            api_key=api_key
        )
        return client
    except Exception as e:
        st.error(f"Error configuring Azure OpenAI: {str(e)}")
        return None

def extract_text_from_file(file, file_type):
    """Extract text content from different file types"""
    try:
        if file_type == "txt":
            return file.read().decode('utf-8')
        
        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_type == "docx":
            doc = Document(BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif file_type == "ipynb":
            notebook = nbformat.read(BytesIO(file.read()), as_version=4)
            text = ""
            
            for i, cell in enumerate(notebook.cells):
                text += f"=== CELL {i+1} ===\n"
                
                if cell.cell_type == "code":
                    text += f"CODE CELL:\n{cell.source}\n"
                    
                    # Add execution count if available
                    if hasattr(cell, 'execution_count') and cell.execution_count:
                        text += f"EXECUTION COUNT: {cell.execution_count}\n"
                    
                    # Add outputs if available
                    if hasattr(cell, 'outputs') and cell.outputs:
                        text += "\nOUTPUTS:\n"
                        for output in cell.outputs:
                            # Handle different output types
                            if output.output_type == 'execute_result':
                                # Code execution results
                                if 'text/plain' in output.data:
                                    text += f"RESULT: {output.data['text/plain']}\n"
                                if 'text/html' in output.data:
                                    text += f"HTML OUTPUT: {output.data['text/html']}\n"
                                if 'image/png' in output.data:
                                    text += "IMAGE OUTPUT: [PNG Image Generated]\n"
                                if 'image/jpeg' in output.data:
                                    text += "IMAGE OUTPUT: [JPEG Image Generated]\n"
                                if 'application/json' in output.data:
                                    text += f"JSON OUTPUT: {output.data['application/json']}\n"
                            
                            elif output.output_type == 'stream':
                                # Print statements, stdout
                                text += f"PRINT OUTPUT: {output.text}\n"
                            
                            elif output.output_type == 'error':
                                # Error messages
                                text += f"ERROR: {output.ename}: {output.evalue}\n"
                                if hasattr(output, 'traceback'):
                                    text += f"TRACEBACK: {' '.join(output.traceback)}\n"
                            
                            elif output.output_type == 'display_data':
                                # Display outputs (plots, widgets, etc.)
                                if 'text/plain' in output.data:
                                    text += f"DISPLAY: {output.data['text/plain']}\n"
                                if 'image/png' in output.data:
                                    text += "PLOT/CHART: [PNG Plot Generated]\n"
                                if 'image/jpeg' in output.data:
                                    text += "PLOT/CHART: [JPEG Plot Generated]\n"
                                if 'text/html' in output.data:
                                    text += f"HTML DISPLAY: {output.data['text/html']}\n"
                    else:
                        text += "NO OUTPUT (Cell not executed or no output generated)\n"
                
                elif cell.cell_type == "markdown":
                    text += f"MARKDOWN CELL:\n{cell.source}\n"
                
                elif cell.cell_type == "raw":
                    text += f"RAW CELL:\n{cell.source}\n"
                
                text += "\n" + "="*50 + "\n\n"
            
            return text
        
        else:
            return "Unsupported file type"
    
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def evaluate_assignment_with_file(client, question_paper, uploaded_file, evaluation_criteria):
    """Evaluate a single assignment by sending the file content directly to AI"""
    
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    
    # Read file content
    try:
        file_content = uploaded_file.read()
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
    except Exception as e:
        return f"Error reading file {file_name}: {str(e)}"
    
    # Extract content based on file type
    file_content_text = ""
    
    try:
        if file_extension == 'ipynb':
            # Parse Jupyter notebook
            import json as json_lib
            notebook_data = json_lib.loads(file_content.decode('utf-8'))
            
            file_content_text = "JUPYTER NOTEBOOK CONTENT:\n\n"
            for i, cell in enumerate(notebook_data.get('cells', [])):
                file_content_text += f"=== CELL {i+1} ===\n"
                cell_type = cell.get('cell_type', 'unknown')
                source = ''.join(cell.get('source', []))
                
                if cell_type == 'code':
                    file_content_text += f"CODE CELL:\n{source}\n"
                    
                    # Add execution count
                    exec_count = cell.get('execution_count')
                    if exec_count:
                        file_content_text += f"EXECUTION COUNT: {exec_count}\n"
                    
                    # Add outputs
                    outputs = cell.get('outputs', [])
                    if outputs:
                        file_content_text += "\nOUTPUTS:\n"
                        for output in outputs:
                            output_type = output.get('output_type', '')
                            
                            if output_type == 'execute_result':
                                data = output.get('data', {})
                                if 'text/plain' in data:
                                    file_content_text += f"RESULT: {data['text/plain']}\n"
                                if 'text/html' in data:
                                    file_content_text += f"HTML: {data['text/html']}\n"
                                if 'image/png' in data:
                                    file_content_text += "IMAGE: [PNG plot/chart generated]\n"
                                    
                            elif output_type == 'stream':
                                text = ''.join(output.get('text', []))
                                file_content_text += f"PRINT: {text}\n"
                                
                            elif output_type == 'error':
                                ename = output.get('ename', '')
                                evalue = output.get('evalue', '')
                                file_content_text += f"ERROR: {ename}: {evalue}\n"
                                
                            elif output_type == 'display_data':
                                data = output.get('data', {})
                                if 'text/plain' in data:
                                    file_content_text += f"DISPLAY: {data['text/plain']}\n"
                                if 'image/png' in data:
                                    file_content_text += "PLOT: [PNG visualization generated]\n"
                    else:
                        file_content_text += "NO OUTPUT\n"
                        
                elif cell_type == 'markdown':
                    file_content_text += f"MARKDOWN CELL:\n{source}\n"
                    
                elif cell_type == 'raw':
                    file_content_text += f"RAW CELL:\n{source}\n"
                    
                file_content_text += "\n" + "="*50 + "\n\n"
                
        elif file_extension in ['txt', 'py', 'md', 'r']:
            # Text-based files
            file_content_text = file_content.decode('utf-8')
            
        elif file_extension in ['pdf', 'docx']:
            # Use the existing extraction function
            file_content_text = extract_text_from_file(uploaded_file, file_extension)
            
        else:
            file_content_text = f"Binary file of type {file_extension.upper()}"
            
    except Exception as e:
        file_content_text = f"Error extracting content from {file_extension} file: {str(e)}"
    
    prompt = f"""
You are an expert academic evaluator. Please evaluate the student's assignment file against the specific question paper requirements.

QUESTION PAPER:
{question_paper}

GENERAL EVALUATION GUIDELINES:
{evaluation_criteria}

ASSIGNMENT FILE: {file_name} (Type: {file_extension.upper()})

COMPLETE FILE CONTENT:
{file_content_text}

EVALUATION INSTRUCTIONS:
1. First, analyze the QUESTION PAPER to identify specific requirements, tasks, and expected deliverables
2. Create custom evaluation criteria based on what the question paper asks for
3. Evaluate the assignment against these specific requirements
4. For each requirement in the question paper, check if it's addressed in the assignment
5. Score based on how well each specific requirement is met

Please provide a comprehensive evaluation that:
- Identifies specific requirements from the question paper
- Evaluates how well each requirement is addressed
- Uses scoring criteria relevant to this specific assignment
- Provides targeted feedback based on what was actually asked

Respond with a detailed evaluation in the following JSON format:
{{
    "overall_score": "X/100",
    "question_paper_analysis": {{
        "identified_requirements": ["requirement1", "requirement2", "requirement3"],
        "assignment_type": "Description of what type of assignment this is",
        "key_deliverables": ["deliverable1", "deliverable2", "deliverable3"]
    }},
    "requirement_evaluation": {{
        "requirement1_name": {{
            "score": "X/Y",
            "status": "Met/Partially Met/Not Met",
            "feedback": "Specific feedback for this requirement"
        }},
        "requirement2_name": {{
            "score": "X/Y", 
            "status": "Met/Partially Met/Not Met",
            "feedback": "Specific feedback for this requirement"
        }}
    }},
    "detailed_scores": {{
        "requirement_fulfillment": "X/40",
        "technical_implementation": "X/30",
        "presentation_quality": "X/20",
        "documentation_clarity": "X/10"
    }},
    "strengths": ["strength1", "strength2", "strength3"],
    "areas_for_improvement": ["improvement1", "improvement2", "improvement3"],
    "detailed_feedback": "Comprehensive feedback analyzing how well the assignment meets the specific question paper requirements",
    "grade": "A/B/C/D/F",
    "recommendations": ["recommendation1", "recommendation2"],
    "file_analysis": {{
        "file_type": "{file_extension}",
        "file_name": "{file_name}",
        "content_summary": "Brief summary of what was found in the file",
        "technical_quality": "Assessment of technical implementation",
        "presentation_quality": "Assessment of formatting and presentation"
    }}
}}

IMPORTANT: Base your evaluation entirely on what the question paper specifically asks for. Different assignments should have different evaluation criteria based on their unique requirements.
"""
    
    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic evaluator. Analyze the provided file content thoroughly and provide comprehensive, fair evaluations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error during AI evaluation: {str(e)}"

def process_multiple_files(uploaded_files, question_paper, client, evaluation_criteria):
    """Process multiple assignment files by sending them directly to AI"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"AI analyzing {uploaded_file.name}...")
        
        # Reset file pointer to beginning
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        
        # Send file directly to AI for evaluation
        evaluation = evaluate_assignment_with_file(client, question_paper, uploaded_file, evaluation_criteria)
        
        results.append({
            "filename": uploaded_file.name,
            "evaluation": evaluation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": uploaded_file.name.split('.')[-1].lower()
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("AI analysis complete!")
    return results

def generate_pdf_report(evaluation_results, question_paper_text=""):
    """Generate a comprehensive PDF report of all evaluations"""
    
    # Create a BytesIO buffer to hold the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.darkgreen
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.spaceAfter = 6
    
    # Build the PDF content
    story = []
    
    # Title page
    story.append(Paragraph("🎓 AI Assignment Evaluation Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary information
    story.append(Paragraph("📊 Evaluation Summary", heading_style))
    
    summary_data = [
        ['Total Assignments Evaluated', str(len(evaluation_results))],
        ['Evaluation Date', datetime.now().strftime("%B %d, %Y")],
        ['Evaluation Time', datetime.now().strftime("%I:%M %p")],
        ['AI Model Used', 'Azure OpenAI GPT-OSS-120B']
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Question paper section (if available)
    if question_paper_text:
        story.append(Paragraph("📄 Question Paper", heading_style))
        # Truncate if too long
        qp_text = question_paper_text[:1000] + "..." if len(question_paper_text) > 1000 else question_paper_text
        story.append(Paragraph(qp_text.replace('\n', '<br/>'), normal_style))
        story.append(Spacer(1, 20))
    
    story.append(PageBreak())
    
    # Individual evaluation results
    for i, result in enumerate(evaluation_results):
        story.append(Paragraph(f"📝 Assignment {i+1}: {result['filename']}", heading_style))
        story.append(Paragraph(f"File Type: {result.get('file_type', 'Unknown').upper()}", normal_style))
        story.append(Paragraph(f"Evaluated: {result['timestamp']}", normal_style))
        story.append(Spacer(1, 12))
        
        try:
            # Try to parse JSON evaluation
            if result['evaluation'].strip().startswith('{'):
                eval_data = json.loads(result['evaluation'])
                
                # Overall score section
                story.append(Paragraph("🎯 Overall Performance", subheading_style))
                
                score_data = [
                    ['Overall Score', eval_data.get('overall_score', 'N/A')],
                    ['Grade', eval_data.get('grade', 'N/A')]
                ]
                
                score_table = Table(score_data, colWidths=[2*inch, 1*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(score_table)
                story.append(Spacer(1, 12))
                
                # Detailed scores
                if 'detailed_scores' in eval_data:
                    story.append(Paragraph("📊 Detailed Breakdown", subheading_style))
                    
                    detailed_data = [['Criterion', 'Score']]
                    for criterion, score in eval_data['detailed_scores'].items():
                        detailed_data.append([criterion.replace('_', ' ').title(), str(score)])
                    
                    detailed_table = Table(detailed_data, colWidths=[3*inch, 1*inch])
                    detailed_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(detailed_table)
                    story.append(Spacer(1, 12))
                
                # Question paper analysis (if available)
                if 'question_paper_analysis' in eval_data:
                    qp_analysis = eval_data['question_paper_analysis']
                    story.append(Paragraph("📋 Question Paper Analysis", subheading_style))
                    
                    if 'identified_requirements' in qp_analysis:
                        story.append(Paragraph("<b>Identified Requirements:</b>", normal_style))
                        for req in qp_analysis['identified_requirements']:
                            story.append(Paragraph(f"• {req}", normal_style))
                    
                    if 'assignment_type' in qp_analysis:
                        story.append(Paragraph(f"<b>Assignment Type:</b> {qp_analysis['assignment_type']}", normal_style))
                    
                    story.append(Spacer(1, 8))
                
                # Requirement evaluation (if available)
                if 'requirement_evaluation' in eval_data:
                    story.append(Paragraph("✅ Requirement Analysis", subheading_style))
                    
                    req_data = [['Requirement', 'Score', 'Status']]
                    for req_name, req_info in eval_data['requirement_evaluation'].items():
                        req_data.append([
                            req_name.replace('_', ' ').title(),
                            req_info.get('score', 'N/A'),
                            req_info.get('status', 'N/A')
                        ])
                    
                    req_table = Table(req_data, colWidths=[2*inch, 1*inch, 1*inch])
                    req_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(req_table)
                    story.append(Spacer(1, 12))
                
                # Strengths
                if 'strengths' in eval_data and eval_data['strengths']:
                    story.append(Paragraph("💪 Strengths", subheading_style))
                    for strength in eval_data['strengths']:
                        story.append(Paragraph(f"• {strength}", normal_style))
                    story.append(Spacer(1, 8))
                
                # Areas for improvement
                if 'areas_for_improvement' in eval_data and eval_data['areas_for_improvement']:
                    story.append(Paragraph("🔄 Areas for Improvement", subheading_style))
                    for improvement in eval_data['areas_for_improvement']:
                        story.append(Paragraph(f"• {improvement}", normal_style))
                    story.append(Spacer(1, 8))
                
                # Detailed feedback
                if 'detailed_feedback' in eval_data:
                    story.append(Paragraph("📝 Detailed Feedback", subheading_style))
                    feedback_text = eval_data['detailed_feedback'].replace('\n', '<br/>')
                    story.append(Paragraph(feedback_text, normal_style))
                    story.append(Spacer(1, 8))
                
                # Recommendations
                if 'recommendations' in eval_data and eval_data['recommendations']:
                    story.append(Paragraph("💡 Recommendations", subheading_style))
                    for recommendation in eval_data['recommendations']:
                        story.append(Paragraph(f"• {recommendation}", normal_style))
                    story.append(Spacer(1, 8))
                
                # File analysis
                if 'file_analysis' in eval_data:
                    file_analysis = eval_data['file_analysis']
                    story.append(Paragraph("🔍 File Analysis", subheading_style))
                    
                    if 'content_summary' in file_analysis:
                        story.append(Paragraph(f"<b>Content Summary:</b> {file_analysis['content_summary']}", normal_style))
                    if 'technical_quality' in file_analysis:
                        story.append(Paragraph(f"<b>Technical Quality:</b> {file_analysis['technical_quality']}", normal_style))
                    if 'presentation_quality' in file_analysis:
                        story.append(Paragraph(f"<b>Presentation Quality:</b> {file_analysis['presentation_quality']}", normal_style))
            
            else:
                # Raw evaluation text
                story.append(Paragraph("📄 Evaluation Response", subheading_style))
                eval_text = result['evaluation'].replace('\n', '<br/>')
                story.append(Paragraph(eval_text, normal_style))
        
        except json.JSONDecodeError:
            # Raw evaluation text
            story.append(Paragraph("📄 Evaluation Response", subheading_style))
            eval_text = result['evaluation'].replace('\n', '<br/>')
            story.append(Paragraph(eval_text, normal_style))
        
        # Add page break between assignments (except for the last one)
        if i < len(evaluation_results) - 1:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 20))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("Generated by AI Assignment Evaluator", 
                          ParagraphStyle('Footer', parent=styles['Normal'], 
                                       fontSize=8, alignment=TA_CENTER, 
                                       textColor=colors.grey)))
    
    # Build PDF
    doc.build(story)
    
    # Get the PDF data
    buffer.seek(0)
    return buffer.getvalue()

def main():
    st.title("🎓 AI Assignment Evaluator")
    st.markdown("### Automated assignment evaluation using Azure OpenAI")
    
    # Configure Azure OpenAI (hidden from UI)
    api_key = st.session_state.api_key
    if not api_key:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        st.session_state.api_key = api_key
    
    client = None
    if api_key:
        client = configure_azure_openai(api_key)
    
    # Sidebar for uploads and configuration
    with st.sidebar:
        st.header("📄 Question Paper")
        
        # Question paper upload
        question_file = st.file_uploader(
            "Upload Question Paper",
            type=['txt', 'pdf', 'docx'],
            help="Upload the question paper in TXT, PDF, or DOCX format"
        )
        
        question_paper_text = ""
        if question_file:
            file_extension = question_file.name.split('.')[-1].lower()
            question_paper_text = extract_text_from_file(question_file, file_extension)
            st.success(f"✅ Question paper loaded: {question_file.name}")
            
            with st.expander("📖 Preview"):
                st.text_area("Content", question_paper_text[:500] + "..." if len(question_paper_text) > 500 else question_paper_text, height=150, disabled=True)
        
        st.markdown("---")
        
        st.header("📚 Student Submissions")
        
        # Upload mode selection
        upload_mode = st.radio(
            "Upload Mode",
            ["Single File", "Multiple Files", "Select Folder", "Folder (ZIP)"],
            horizontal=False
        )
        
        uploaded_files = []
        
        if upload_mode == "Single File":
            single_file = st.file_uploader(
                "Upload Single Assignment",
                type=['txt', 'pdf', 'docx', 'ipynb'],
                help="Upload a single assignment file"
            )
            if single_file:
                uploaded_files = [single_file]
        
        elif upload_mode == "Multiple Files":
            multiple_files = st.file_uploader(
                "Upload Multiple Assignments",
                type=['txt', 'pdf', 'docx', 'ipynb'],
                accept_multiple_files=True,
                help="Upload multiple assignment files"
            )
            if multiple_files:
                uploaded_files = multiple_files
        
        elif upload_mode == "Select Folder":
            st.markdown("**Select Assignment Folder:**")
            
            # Initialize session state for folder path
            if 'selected_folder_path' not in st.session_state:
                st.session_state.selected_folder_path = ""
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display current selected folder
                if st.session_state.selected_folder_path:
                    st.text_input(
                        "Selected Folder",
                        value=st.session_state.selected_folder_path,
                        disabled=True,
                        help="Currently selected folder path"
                    )
                else:
                    st.info("No folder selected yet")
            
            with col2:
                if st.button("📁 Browse", help="Click to select a folder"):
                    # Create a simple folder selection script
                    folder_script = """
import tkinter as tk
from tkinter import filedialog
import sys

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

# Open folder dialog
folder_path = filedialog.askdirectory(title="Select Assignment Folder")

if folder_path:
    print(folder_path)
else:
    print("CANCELLED")

root.destroy()
"""
                    
                    # Write script to temp file and execute
                    import subprocess
                    import tempfile
                    
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                            f.write(folder_script)
                            script_path = f.name
                        
                        # Run the folder selection script
                        result = subprocess.run([
                            'python', script_path
                        ], capture_output=True, text=True, timeout=30)
                        
                        # Clean up temp file
                        os.unlink(script_path)
                        
                        if result.returncode == 0:
                            folder_path = result.stdout.strip()
                            if folder_path and folder_path != "CANCELLED":
                                st.session_state.selected_folder_path = folder_path
                                st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error opening folder dialog: {str(e)}")
                        st.info("💡 **Alternative:** You can also manually enter the folder path below:")
                        
                        # Fallback to manual input
                        manual_path = st.text_input(
                            "Manual Folder Path",
                            placeholder="C:/path/to/assignments/folder",
                            help="Enter the full path to the folder containing assignment files"
                        )
                        if manual_path:
                            st.session_state.selected_folder_path = manual_path
            
            # Process selected folder
            if st.session_state.selected_folder_path and os.path.exists(st.session_state.selected_folder_path):
                try:
                    # Find all supported files in the folder
                    supported_extensions = ['.txt', '.pdf', '.docx', '.ipynb']
                    found_files = []
                    
                    for root, dirs, files in os.walk(st.session_state.selected_folder_path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in supported_extensions):
                                file_path = os.path.join(root, file)
                                found_files.append(file_path)
                    
                    if found_files:
                        st.success(f"✅ Found {len(found_files)} assignment files")
                        
                        # Show file list in expander
                        with st.expander("📋 Files Found", expanded=False):
                            for file_path in found_files:
                                st.write(f"• {os.path.basename(file_path)}")
                        
                        # Create file-like objects for processing
                        for file_path in found_files:
                            try:
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                    file_name = os.path.basename(file_path)
                                    
                                    # Create a proper file-like object with correct content binding
                                    class FileWrapper:
                                        def __init__(self, name, content):
                                            self.name = name
                                            self._content = content
                                            self._position = 0
                                        
                                        def read(self):
                                            return self._content
                                        
                                        def seek(self, position):
                                            self._position = position
                                            return self._position
                                    
                                    uploaded_files.append(FileWrapper(file_name, file_content))
                            except Exception as e:
                                st.warning(f"Could not read {file_path}: {str(e)}")
                    else:
                        st.warning("⚠️ No supported files found in the selected folder")
                        st.info("Supported formats: .txt, .pdf, .docx, .ipynb")
                        
                except Exception as e:
                    st.error(f"Error accessing folder: {str(e)}")
            
            elif st.session_state.selected_folder_path:
                st.error("❌ Selected folder path does not exist")
                if st.button("🔄 Clear Selection"):
                    st.session_state.selected_folder_path = ""
                    st.rerun()
        
        elif upload_mode == "Folder (ZIP)":
            zip_file = st.file_uploader(
                "Upload ZIP Folder",
                type=['zip'],
                help="Upload a ZIP file containing multiple assignments"
            )
            
            if zip_file:
                # Extract ZIP file
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = os.path.join(temp_dir, "assignments.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.read())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find all supported files
                    supported_extensions = ['.txt', '.pdf', '.docx', '.ipynb']
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in supported_extensions):
                                file_path = os.path.join(root, file)
                                with open(file_path, 'rb') as f:
                                    # Create a file-like object
                                    file_content = f.read()
                                    
                                    # Create a proper file-like object with correct content binding
                                    class ZipFileWrapper:
                                        def __init__(self, name, content):
                                            self.name = name
                                            self._content = content
                                            self._position = 0
                                        
                                        def read(self):
                                            return self._content
                                        
                                        def seek(self, position):
                                            self._position = position
                                            return self._position
                                    
                                    uploaded_files.append(ZipFileWrapper(file, file_content))
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.write(f"• {file.name}")
        
        st.markdown("---")
        
        # Evaluation criteria
        st.subheader("📋 Evaluation Criteria")
        evaluation_criteria = st.text_area(
            "Custom Evaluation Criteria",
            value=""" """,
            height=120
        )
    
    # Main content area - Evaluation Process
    if not client:
        st.error("🔑 **API Configuration Required**")
        st.markdown("Please add your Azure OpenAI API key to the `.env` file:")
        st.code("AZURE_OPENAI_API_KEY=your_api_key_here")
        st.markdown("Then restart the application.")
        return
    
    # Evaluation Process
    st.header("🔍 Evaluation Process")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if question_paper_text:
            st.success("✅ Question Paper Ready")
        else:
            st.warning("⏳ Upload Question Paper")
    
    with col2:
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} Assignment(s) Ready")
        else:
            st.warning("⏳ Upload Assignments")
    
    with col3:
        if evaluation_criteria:
            st.success("✅ Criteria Set")
        else:
            st.warning("⏳ Set Criteria")
    
    # Start evaluation button
    st.markdown("### Ready to Evaluate?")
    
    if st.button("🚀 Start AI Evaluation", type="primary", disabled=not (client and question_paper_text and uploaded_files)):
        if not client:
            st.error("❌ Azure OpenAI not configured!")
        elif not question_paper_text:
            st.error("❌ Please upload a question paper!")
        elif not uploaded_files:
            st.error("❌ Please upload assignment files!")
        else:
            with st.spinner("🤖 AI is evaluating assignments..."):
                results = process_multiple_files(uploaded_files, question_paper_text, client, evaluation_criteria)
                st.session_state.evaluation_results = results
                st.success("🎉 Evaluation completed!")
                st.balloons()
    
    # Display results
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.header("📊 Evaluation Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Assignments", len(st.session_state.evaluation_results))
        
        with col2:
            st.metric("✅ Evaluated", len(st.session_state.evaluation_results))
        
        with col3:
            st.metric("📈 Success Rate", "100%")
        
        with col4:
            st.metric("⏱️ Avg. Time", "< 1 min")
        
        st.markdown("---")
        
        # Detailed results
        st.subheader("📋 Individual Reports")
        
        for i, result in enumerate(st.session_state.evaluation_results):
            file_type = result.get('file_type', 'unknown')
            file_icon = "📓" if file_type == 'ipynb' else "📄" if file_type in ['pdf', 'docx'] else "💻" if file_type in ['py', 'txt'] else "📁"
            
            with st.expander(f"{file_icon} **{result['filename']}** ({file_type.upper()}) - {result['timestamp']}", expanded=i==0):
                try:
                    # Try to parse JSON response
                    if result['evaluation'].strip().startswith('{'):
                        eval_data = json.loads(result['evaluation'])
                        
                        # File Analysis Section (if available)
                        if 'file_analysis' in eval_data:
                            file_analysis = eval_data['file_analysis']
                            st.markdown("### 📋 **File Analysis**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**File Type:** {file_analysis.get('file_type', 'N/A').upper()}")
                                st.markdown(f"**Content Summary:** {file_analysis.get('content_summary', 'N/A')}")
                            with col2:
                                st.markdown(f"**Technical Quality:** {file_analysis.get('technical_quality', 'N/A')}")
                                st.markdown(f"**Presentation Quality:** {file_analysis.get('presentation_quality', 'N/A')}")
                            st.markdown("---")
                        
                        # Score display
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("### 🎯 Overall Score")
                            score = eval_data.get('overall_score', 'N/A')
                            grade = eval_data.get('grade', 'N/A')
                            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{score}</h1>", unsafe_allow_html=True)
                            st.markdown(f"<h2 style='text-align: center; color: #ff7f0e;'>Grade: {grade}</h2>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 📊 Detailed Breakdown")
                            scores = eval_data.get('detailed_scores', {})
                            for criterion, score in scores.items():
                                st.markdown(f"**{criterion.replace('_', ' ').title()}:** `{score}`")
                        
                        st.markdown("---")
                        
                        # Feedback sections
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### ✅ **Strengths**")
                            strengths = eval_data.get('strengths', [])
                            if strengths:
                                for strength in strengths:
                                    st.markdown(f"• {strength}")
                            else:
                                st.markdown("*No specific strengths identified*")
                        
                        with col2:
                            st.markdown("### 🔄 **Areas for Improvement**")
                            improvements = eval_data.get('areas_for_improvement', [])
                            if improvements:
                                for improvement in improvements:
                                    st.markdown(f"• {improvement}")
                            else:
                                st.markdown("*No specific improvements suggested*")
                        
                        st.markdown("---")
                        
                        st.markdown("### 📝 **Detailed Feedback**")
                        feedback = eval_data.get('detailed_feedback', 'No detailed feedback available')
                        st.markdown(f"*{feedback}*")
                        
                        st.markdown("### 💡 **Recommendations**")
                        recommendations = eval_data.get('recommendations', [])
                        if recommendations:
                            for recommendation in recommendations:
                                st.markdown(f"• {recommendation}")
                        else:
                            st.markdown("*No specific recommendations provided*")
                    
                    else:
                        # Display raw response if not JSON
                        st.markdown("### 📄 AI Evaluation Response")
                        st.markdown(result['evaluation'])
                
                except json.JSONDecodeError:
                    st.markdown("### 📄 AI Evaluation Response")
                    st.markdown(result['evaluation'])
        
        # Export results
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate PDF data first
            try:
                # Get question paper text for the report
                question_paper_text = ""
                # Try to get question paper from session state or current context
                for key, value in st.session_state.items():
                    if 'question_paper' in key.lower() and isinstance(value, str):
                        question_paper_text = value
                        break
                
                pdf_data = generate_pdf_report(st.session_state.evaluation_results, question_paper_text)
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"assignment_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    help="Click to download the comprehensive PDF evaluation report"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.button("📄 PDF Generation Failed", disabled=True, type="primary")
        
        with col2:
            if st.button("🔄 Clear Results", type="secondary"):
                st.session_state.evaluation_results = []
                st.rerun()
    
    else:
        # Show placeholder when no results
        st.markdown("---")
        st.info("📋 **No evaluations yet.** Upload your question paper and assignments, then click 'Start AI Evaluation' to begin.")

if __name__ == "__main__":
    main()