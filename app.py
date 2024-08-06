import streamlit as st
import pickle
import re
import fitz  # PyMuPDF


# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(txt):
    clean_txt = re.sub(r'http\S+\s', ' ', txt)
    clean_txt = re.sub(r'RT|cc', ' ', clean_txt)
    clean_txt = re.sub(r'@\S+', ' ', clean_txt)
    clean_txt = re.sub(r'#\S+\s', ' ', clean_txt)
    clean_txt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_txt)
    clean_txt = re.sub(r'[^\x00-\x7f]', ' ', clean_txt)
    clean_txt = re.sub(r'\s+', ' ', clean_txt)
    return clean_txt

def extract_text_from_pdf(pdf_file):
    document = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            resume_text = extract_text_from_pdf(uploaded_file.read())
        else:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

# Python main
if __name__ == "__main__":
    main()
