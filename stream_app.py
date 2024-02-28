import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import io

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Password input by the user
password = st.text_input("Enter the password", type="password")

# "Load App" button
load_app = st.button('Load App')

if load_app:
    if password == 'inspire_24':
        # Update session state to indicate the user is authenticated
        st.session_state['authenticated'] = True
    else:
        st.error("Incorrect password. Please try again.")

if st.session_state['authenticated']:
    # Function to preprocess the uploaded file
    def preprocess_file(uploaded_file):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        else:
            return None

    # Function to perform fuzzy matching
    def fuzzy_match(df1, df2, columns1, columns2, num_records):
        if num_records != 'All':
            if num_records < len(df1):
                df1 = df1.sample(n=num_records)  # Randomly sample records from df1
        matches = []

        for _, row1 in df1.iterrows():
            best_score = -1
            best_match_index = None
            for i, row2 in df2.iterrows():
                score = 0
                for col1, col2 in zip(columns1, columns2):
                    score += fuzz.ratio(str(row1[col1]), str(row2[col2]))
                score /= len(columns1)
                if score > best_score:
                    best_score = score
                    best_match_index = i
            match_data = [row1[col] for col in columns1] + [df2.iloc[best_match_index][col] for col in columns2] + [best_score]
            matches.append(match_data)

        column_labels = ['Doc1_' + col for col in columns1] + ['Doc2_' + col for col in columns2] + ['Average Score']
        return pd.DataFrame(matches, columns=column_labels)

    # Streamlit UI
    st.title('Record Linking Tool - Prototype')

    with st.expander("How to Use This Tool"):
        st.write("""
            1. Upload two CSV files using the file uploaders below.
            2. Preview the uploaded files to confirm their contents.
            3. Select between 2 to 6 columns from each document for comparison.
            4. Enter the number of records you want to compare or 'All' to compare all records. If a number is entered, that many records will be randomly selected from the first document.
            5. Click the 'Compare Documents' button to perform the comparison.
            6. Review the matching records displayed along with their average similarity score.
            7. Use the 'Download matching records as CSV' button to download the results.
        """)

    st.header('Upload Files')
    file1 = st.file_uploader('Choose the first file', type=['csv'])
    file2 = st.file_uploader('Choose the second file', type=['csv'])

    if file1 and file2:
        df1 = preprocess_file(file1)
        df2 = preprocess_file(file2)

        st.header('Preview of Document 1')
        st.write(df1.head())

        st.header('Preview of Document 2')
        st.write(df2.head())

        st.header('Select Columns for Comparison')
        if not df1.empty and not df2.empty:
            columns1 = st.multiselect('Select columns from Document 1:', options=df1.columns, help="Select between 2 to 6 columns")
            columns2 = st.multiselect('Select columns from Document 2:', options=df2.columns, help="Select between 2 to 6 columns")

            num_records = st.text_input('How many records do you want to compare? Enter a number or "All" for all records.', 'All')

            # Convert input to either an integer or "All"
            if num_records != 'All':
                try:
                    num_records = int(num_records)
                except ValueError:
                    st.error('Please enter a valid number or "All".')
                    st.stop()

            if st.button('Compare Documents'):
                if 2 <= len(columns1) <= 6 and 2 <= len(columns2) <= 6:
                    result_df = fuzzy_match(df1, df2, columns1, columns2, num_records)

                    st.header('Matching Records')
                    st.write(result_df)

                    # Convert DataFrame to CSV for download
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download matching records as CSV",
                        data=csv,
                        file_name='matching_records.csv',
                        mime='text/csv',
                    )
                else:
                    st.error('Please select between 2 to 6 columns from each document.')
    else:
        st.write('Please upload both files to compare.')
