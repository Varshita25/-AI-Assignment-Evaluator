# 🎓 AI Assignment Evaluator

An intelligent assignment evaluation system that helps teachers and faculty automatically evaluate student submissions using Google's Gemini AI.

## Features

- **Multiple File Format Support**: Supports .txt, .pdf, .docx, and .ipynb files
- **Flexible Upload Options**: 
  - Single file upload
  - Multiple files upload
  - ZIP folder upload for batch processing
- **AI-Powered Evaluation**: Uses Google Gemini AI for intelligent assessment
- **Comprehensive Reports**: Generates detailed evaluation reports with scores, feedback, and recommendations
- **Customizable Criteria**: Define your own evaluation criteria
- **Export Results**: Download evaluation results as JSON

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in the application

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage Guide

### Step 1: Configure API
- Enter your Gemini API key in the sidebar
- Customize evaluation criteria if needed

### Step 2: Upload Question Paper
- Upload the question paper in TXT, PDF, or DOCX format
- Preview the content to ensure it's correctly extracted

### Step 3: Upload Student Submissions
Choose from three upload modes:
- **Single File**: Upload one assignment at a time
- **Multiple Files**: Select multiple assignment files
- **Folder (ZIP)**: Upload a ZIP file containing multiple assignments

### Step 4: Start Evaluation
- Click "Start Evaluation" to begin the AI assessment
- Wait for processing to complete

### Step 5: Review Results
- View detailed evaluation reports for each submission
- Export results as JSON for record-keeping

## Supported File Formats

- **Text Files (.txt)**: Plain text assignments
- **PDF Files (.pdf)**: PDF documents
- **Word Documents (.docx)**: Microsoft Word files
- **Jupyter Notebooks (.ipynb)**: Python notebooks with code and markdown

## Evaluation Criteria

The default evaluation criteria includes:
- **Content Accuracy (25%)**: Correctness of answers and concepts
- **Completeness (25%)**: Coverage of all required topics
- **Presentation (25%)**: Clarity, organization, and formatting
- **Innovation (25%)**: Creative thinking and original insights

You can customize these criteria in the sidebar.

## Output Format

Each evaluation includes:
- Overall score (0-100)
- Detailed scores by criteria
- Letter grade (A/B/C/D/F)
- Strengths identified
- Areas for improvement
- Detailed feedback
- Recommendations for students

## Tips for Best Results

1. **Clear Question Papers**: Ensure question papers are well-formatted and clearly readable
2. **Consistent Formatting**: Encourage students to use consistent formatting in submissions
3. **Detailed Criteria**: Provide specific evaluation criteria for more accurate assessments
4. **Review AI Output**: Always review AI evaluations before finalizing grades

## Troubleshooting

### Common Issues:

1. **API Key Error**: Ensure your Gemini API key is valid and has sufficient quota
2. **File Reading Error**: Check that uploaded files are not corrupted and in supported formats
3. **Memory Issues**: For large batches, consider processing files in smaller groups

### Error Messages:

- "Error configuring Gemini API": Check your API key
- "Error extracting text": File may be corrupted or in unsupported format
- "Error during evaluation": API quota may be exceeded or network issue

## Security Notes

- API keys are stored only in session state and not persisted
- Uploaded files are processed in memory and not saved to disk
- All data is processed locally except for AI evaluation calls

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please check the troubleshooting section or create an issue in the project repository.