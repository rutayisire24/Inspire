import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np

# Preprocessing function to clean data
def preprocess_data(df, columns):
    for col in columns:
        # Handle missing values directly, avoiding conversion to 'nan' string
        df[col] = df[col].apply(lambda x: preprocess_string(x) if pd.notnull(x) else x)
    return df

def preprocess_string(x):
    """Preprocess a single string: lowercase, trim, and remove special characters."""
    x = str(x)
    # Convert to lowercase
    x = x.lower()
    # Trim whitespace
    x = x.strip()
    # Replace non-alphanumeric characters with spaces
    x = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in x)
    return x

# Fuzzy matching function (with preprocessing included)
def fuzzy_match(df1, df2, columns1, columns2, weights1, weights2, num_records, file1_name, file2_name):
    # Preprocess the dataframes
    df1 = preprocess_data(df1, columns1)
    df2 = preprocess_data(df2, columns2)
    
    progress_bar = st.progress(0)
    total = len(df1) if num_records == 'All' else min(num_records, len(df1))
    matches = []
    score_bins = {'100': 0, '90-99': 0, '80-89': 0, '70-79': 0, '60-69': 0, '<60': 0}

    for i, row1 in enumerate(df1.iterrows()):
        if num_records != 'All' and i >= num_records:
            break
        best_score = -1
        best_match_index = None
        for row2 in df2.itertuples(index=False):
            weighted_score = 0
            total_weight = sum(weights1)
            for col1, col2, weight in zip(columns1, columns2, weights1):
                score = fuzz.ratio(str(row1[1][col1]), str(getattr(row2, col2)))
                weighted_score += (score * weight)
            weighted_average_score = weighted_score / total_weight
            if weighted_average_score > best_score:
                best_score = weighted_average_score
                best_match_index = row2
        match_data = [row1[1][col] for col in columns1] + [getattr(best_match_index, col) for col in columns2] + [best_score]
        matches.append(match_data)
        
        if best_score > 95:
            score_bins['100'] += 1
        elif 90 <= best_score <= 99:
            score_bins['90-99'] += 1
        elif 80 <= best_score <= 89:
            score_bins['80-89'] += 1
        elif 70 <= best_score <= 79:
            score_bins['70-79'] += 1
        elif 60 <= best_score <= 69:
            score_bins['60-69'] += 1
        else:
            score_bins['<60'] += 1

        progress_bar.progress(min((i + 1) / total, 1.0))

    progress_bar.empty()
    column_labels = [f'{file1_name}_{col}' for col in columns1] + [f'{file2_name}_{col}' for col in columns2] + ['Weighted Average Score']
    matches_df = pd.DataFrame(matches, columns=column_labels)
    matches_df_sorted = matches_df.sort_values(by='Weighted Average Score', ascending=False)

    # Convert score_bins to DataFrame for plotting
    score_bins_df = pd.DataFrame(list(score_bins.items()), columns=['Score Range', 'Count'])
    score_bins_df.set_index('Score Range', inplace=True)
    
    # Display the bar chart for score summary
    st.bar_chart(score_bins_df)
    st.info("Scores of 100  represent a perfect match")
    

    return matches_df_sorted

# Streamlit UI code
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

with st.expander("How to Use This App"):
    st.write("""
        - **Step 1:** Enter the password to load the app.
        - **Step 2:** Upload the Documents using the file uploaders.
        - **Step 3:** Select the columns from both documents that you want to compare.
        - **Step 4:** Click on 'Adjust Weights' to set the importance of each selected column (1 to 5).
        - **Step 5:** Enter the number of records you want to compare or 'All' to compare all records.
        - **Step 6:** Click on 'Compare Documents' to start the  matching process.
        - **Note:** Make sure the documents are in CSV format.
    """)

password = st.text_input("Enter the password", type="password", key="password_input")
load_app = st.button('Load App')

if load_app:
    if password == 'inspire_24':
        st.session_state['authenticated'] = True
    else:
        st.error("Incorrect password. Please try again.")

if st.session_state['authenticated']:
    st.title('Record Linking Tool (Prototype) ')

    file1 = st.file_uploader('Choose the First File', type=['csv'], key='file_uploader_1')
    file2 = st.file_uploader('Choose the Second File', type=['csv'], key='file_uploader_2')

    file1_name = file1.name[:-4] if file1 else 'Document 1'
    file2_name = file2.name[:-4] if file2 else 'Document 2'

    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        st.success('Files uploaded successfully!')

    
    if file1 is not None and file2 is not None:
        st.header(f'Preview of {file1_name}')
        st.write(df1.head())

        st.header(f'Preview of {file2_name}')
        st.write(df2.head())

        columns1 = st.multiselect('Select columns from the first document:', options=df1.columns, key='columns_select_1')
        columns2 = st.multiselect('Select columns from the second document:', options=df2.columns, key='columns_select_2')

        if columns1 and columns2:
            weights1 = [st.slider(f"Weight for {col}:", 1, 3, 3, key=f'weight_1_{col}') for col in columns1]
            weights2 = [st.slider(f"Weight for {col}:", 1, 3, 3, key=f'weight_2_{col}') for col in columns2]

                    # Display the total number of records for each file
            st.write(f"Total records in {file1.name}: {df1.shape[0]}")
            st.write(f"Total records in {file2.name}: {df2.shape[0]}")


            num_records_input = st.text_input('How many records do you want to compare? Enter a number or "All" for all records.', 'All', key='num_records_input')
            num_records = 'All' if num_records_input == 'All' else int(num_records_input)

            if st.button('Compare Documents', key='compare_documents_button'):
                with st.spinner('Performing Matching...'):
                    result_df = fuzzy_match(df1, df2, columns1, columns2, weights1, weights2, num_records, file1.name, file2.name)
                    st.success('Matching completed!')
                    st.write(result_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download matching records as CSV",
                        data=csv,
                        file_name='matching_records.csv',
                        mime='text/csv',
                        key='download_csv_button'
                    )

            else:
                st.error('Please select columns from each document and assign weights.')

# Bottom Expander for explaining the approach used by this app
with st.expander("Approach Used by This App"):
    st.write("""
        - The matching score is calculated based on the similarity between the text in the selected columns, using the Levenshtein distance to estimate similarity.
        - Weighted scores based on user-assigned importance are used to prioritize the match results.
        - Results are arranged in descending order of the weighted average score to prioritize higher matches.
    """)
