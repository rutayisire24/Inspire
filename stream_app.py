import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import io

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Top Expander for guiding the user on how to use the app
with st.expander("How to Use This App"):
    st.write("""
        - **Step 1:** Enter the password to load the app.
        - **Step 2:** Upload the first and second documents using the file uploaders.
        - **Step 3:** Select the columns from both documents that you want to compare.
        - **Step 4:** Click on 'Adjust Weights' to set the importance of each selected column (1 to 5).
        - **Step 5:** Enter the number of records you want to compare or 'All' to compare all records.
        - **Step 6:** Click on 'Compare Documents' to start the fuzzy matching process.
        - **Note:** Make sure the documents are in CSV format.
    """)

def fuzzy_match(df1, df2, columns1, columns2, weights1, weights2, num_records):
    progress_bar = st.progress(0)
    total = len(df1) if num_records == 'All' else min(num_records, len(df1))
    matches = []

    for i, row1 in enumerate(df1.iterrows()):
        if num_records != 'All' and i >= num_records:
            break
        best_score = -1
        best_match_index = None
        for row2 in df2.itertuples(index=False):
            weighted_score = 0
            total_weight = sum(weights1)  # Assuming weights are equal for simplicity
            for col1, col2, weight in zip(columns1, columns2, weights1):
                score = fuzz.ratio(str(row1[1][col1]), str(getattr(row2, col2)))
                weighted_score += (score * weight)
            weighted_average_score = weighted_score / total_weight
            if weighted_average_score > best_score:
                best_score = weighted_average_score
                best_match_index = row2
        match_data = [row1[1][col] for col in columns1] + [getattr(best_match_index, col) for col in columns2] + [best_score]
        matches.append(match_data)
        progress_bar.progress(min((i + 1) / total, 1.0))

    progress_bar.empty()
    column_labels = ['Doc1_' + col for col in columns1] + ['Doc2_' + col for col in columns2] + ['Weighted Average Score']
    matches_df = pd.DataFrame(matches, columns=column_labels)
    # Sort the DataFrame by 'Weighted Average Score' in descending order
    return matches_df.sort_values(by='Weighted Average Score', ascending=False)

# Password input by the user with a unique key
password = st.text_input("Enter the password", type="password", key="password_input")

# "Load App" button
load_app = st.button('Load App')

if load_app:
    if password == 'inspire_24':
        st.session_state['authenticated'] = True
    else:
        st.error("Incorrect password. Please try again.")

if st.session_state['authenticated']:
    with st.spinner('Please wait while the files are being uploaded...'):
        st.title('Record Linking Tool - Prototype')

        file1 = st.file_uploader('Choose the first file', type=['csv'], key='file_uploader_1')
        file2 = st.file_uploader('Choose the second file', type=['csv'], key='file_uploader_2')

        # Capture the names of the uploaded files for use in headings and elsewhere
        file1_name = file1.name if file1 else 'Document 1'
        file2_name = file2.name if file2 else 'Document 2'

        if file1 and file2:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

            st.success('Files uploaded successfully!')

    if file1 is not None and file2 is not None:
        st.header(f'Preview of {file1_name}')
        st.write(df1.head())

        st.header(f'Preview of {file2_name}')
        st.write(df2.head())

        columns1 = st.multiselect(f'Select columns from {file1_name}:', options=df1.columns, key='columns_select_1')
        columns2 = st.multiselect(f'Select columns from {file2_name}:', options=df2.columns, key='columns_select_2')

        # Adjust weights section moved here for clarity and separate functionality
        st.subheader('Adjust Weights for Matching Criteria')
        if columns1 and columns2:
            st.session_state['weights1'] = [st.slider(f"Weight for {col} in {file1_name}:", 1, 5, 5, key=f'weight_1_{col}') for col in columns1]
            st.session_state['weights2'] = [st.slider(f"Weight for {col} in {file2_name}:", 1, 5, 5, key=f'weight_2_{col}') for col in columns2]

        num_records_input = st.text_input('How many records do you want to compare? Enter a number or "All" for all records.', 'All', key='num_records_input')

        if num_records_input != 'All':
            try:
                num_records = int(num_records_input)
            except ValueError:
                st.error('Please enter a valid number or "All".')
                st.stop()
        else:
            num_records = 'All'

        if st.button('Compare Documents', key='compare_documents_button'):
            if columns1 and columns2 and (2 <= len(columns1) <= 6 and 2 <= len(columns2) <= 6):
                with st.spinner('Performing fuzzy matching...'):
                    result_df = fuzzy_match(df1, df2, columns1, columns2, st.session_state['weights1'], st.session_state['weights2'], num_records)
                    st.success('Fuzzy matching completed!')

                st.header(f'Matching Records between {file1_name} and {file2_name}')
                st.write(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download matching records as CSV",
                    data=csv,
                    file_name='matching_records.csv',
                    mime='text/csv',
                    key='download_csv_button'
                )
            else:
                st.error('Please select between 2 to 6 columns from each document and assign weights.')

# Bottom Expander for explaining the approach used by this app
with st.expander("Approach Used by This App"):
    st.write("""
        - The app performs fuzzy matching between selected columns from two documents, using the file names to enhance user experience and clarity.
        - Fuzzy matching involves finding rows in one document that approximately match rows in another document.
        - The matching score is calculated based on the similarity between the text in the selected columns, using the Levenshtein distance to estimate similarity.
        - Weighted scores based on user-assigned importance are used to prioritize the match results.
        - Results are arranged in descending order of the weighted average score to prioritize higher matches.
    """)
