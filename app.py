## 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

import streamlit as st
import PyPDF2
import re
import json
from huggingface_hub import InferenceClient

# Hugging Face API client
client = InferenceClient(token="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")

# Streamlit app layout
st.title("Resume Parsing GPT Bot")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract and filter detailed resume information from text
def extract_resume_info(text):
    # Clean text by removing extra newlines and multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    # Define regex patterns for different resume sections
    patterns = {
        "Contact Information": r'Contact\s*Information[:\-]?\s*(.+?)(?=Experience|Education|Skills|Certifications|$)',
        "Experience": r'Experience[:\-]?\s*(.+?)(?=Education|Skills|Certifications|$)',
        "Education": r'Education[:\-]?\s*(.+?)(?=Skills|Certifications|$)',
        "Skills": r'Skills[:\-]?\s*(.+?)(?=Certifications|$)',
        "Certifications": r'Certifications[:\-]?\s*(.+?)(?=Summary|$)',
        "Summary": r'(Summary|Profile|Overview)[:\-]?\s*(.+)$'
    }

    # Extract sections based on defined patterns
    extracted_info = {}
    for section, pattern in patterns.items():
        match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        extracted_info[section] = match.group(1).strip() if match else "No data found."

    # Clean and shorten content where applicable
    for key in extracted_info:
        if len(extracted_info[key]) > 1000:
            extracted_info[key] = extracted_info[key][:1000] + "..."  # Truncate long texts

    return extracted_info

# Upload PDF file
uploaded_pdf = st.file_uploader("Upload a resume PDF file", type="pdf")

if uploaded_pdf is not None:
    # Extract PDF text
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    
    # Extract resume information
    resume_info = extract_resume_info(pdf_text)

    # Display formatted resume information in JSON format
    st.subheader("Extracted Resume Information in JSON Format:")

    # Convert extracted data to JSON
    json_data = json.dumps(resume_info, indent=4)

    # Display JSON data in a code block for better formatting
    st.code(json_data, language='json')

    # Expanders to display each section
    for section, content in resume_info.items():
        with st.expander(f"View {section}"):
            st.write(content)

    # Ask a question about the resume
    question = st.text_input("Ask a question about the resume:")

    if question:
        # Call Hugging Face API for Q&A
        st.write("Fetching response...")
        response = ""
        for message in client.chat_completion(
                messages=[{"role": "user", "content": question}],
                max_tokens=500,
                stream=True,
        ):
            response += message.choices[0].delta.content
            st.write(response)

