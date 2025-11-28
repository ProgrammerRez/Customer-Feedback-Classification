from dotenv import load_dotenv
from backend import FeedbackClassifierPipeline
import time
import pandas as pd
import tempfile
import streamlit as st
import os
import json
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv('.env')

st.set_page_config(page_title='Customer Feedback Classifier', initial_sidebar_state='expanded')
st.title('Customer Feedback Classifier')

# --- Sidebar for API key and file upload ---
with st.sidebar:
    api_key = os.getenv('GROQ_API_KEY') or st.text_input('GROQ_API_KEY', type='password')
    
    file = st.file_uploader("Upload your CSV file here", type="csv")

# Proceed only if a file is uploaded and API key exists
if file is not None and api_key:
    # Read CSV content
    file_content = file.getvalue().decode("utf-8")
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv", encoding='utf-8') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    # Load CSV into pandas
    df = pd.read_csv(tmp_file_path)
    st.success("File loaded successfully!")
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Initialize classifier
    base_labels = ["Bug Report", "Billing Issue", "Complaint", "Feature Request", "General Inquiry"]
    classifier = FeedbackClassifierPipeline(
        labels=base_labels,
        llm_model='llama-3.1-8b-instant',
        groq_api_key=api_key,
        threshold=0.6
    )

    classified_data = []
    new_categories = set()

    # Process each feedback row
    with st.spinner("Classifying feedback..."):
        for i, row in df.iterrows():
            feedback_text = str(row['feedback'])
            result = classifier.classify(feedback_text)
            classified_data.append(result)
            
            # Track new categories
            if result["final_category"] not in base_labels:
                new_categories.add(result["final_category"])

    st.success("Classification complete!")

    # Convert to DataFrame
    classified_df = pd.DataFrame(classified_data)

    # --- Download JSON ---
    json_data = classified_df.to_dict(orient='records')
    json_filename = "classified_feedback.json"
    st.download_button(
        label="Download Classified JSON",
        data=json.dumps(json_data, indent=2),
        file_name=json_filename,
        mime="application/json"
    )

    # --- Show Pie Chart for insights ---
    st.subheader("Feedback Category Distribution")
    fig, ax = plt.subplots()
    counts = classified_df['final_category'].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    # --- Display new categories ---
    if new_categories:
        st.subheader("New Categories Created")
        for cat in new_categories:
            st.write(f"- {cat}")
