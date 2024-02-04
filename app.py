import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import matplotlib
from langchain.llms import OpenAI
import json
import warnings
import time
import seaborn as sns
import streamlit as st
import re
import io
import contextlib
import sys
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.simplefilter(action='ignore')


# llm = OpenAI(open_api_key=st.secrets['oai_api'])

@contextlib.contextmanager
def capture_output():
    new_out = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out



def build_agent(df):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613'),
        df, verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    return agent


def build_query(agent, query):
    prompt = (
            """
            For the following query,
            With the table provided here provide a python code to get the output asked in the query.
            Consider the dataframe is stored as 'df', you dont have to use read_csv
            import the necessary library, print the output in the end.
            If 'group' is in query, bin the column under consideration appropriately before proceeding with aggregation.
            
            Do not explain the code.
            
            Lets think step by step.

            Below is the query.
            Query: 
                """ + query
    )

    response = agent.run(prompt)

    return response.__str__()


def decode_and_run(response):
    decoded = bytes(str(response), "utf-8").decode("unicode_escape")

    pattern = r'```python\s*([\s\S]*?)\s*```'

    matches = re.findall(pattern, decoded)
    st.write(matches)

    if matches:
        if 'plt' in decoded:
            fig = exec(matches[0])
            st.pyplot(fig)

        else:

            with capture_output() as captured:
                exec(matches[0])
                output = captured.getvalue() + '\n'
            st.write(output)

    else:
        st.write(decoded)



def get_output_from_agent(df, query):

    agent = build_agent(df)
    function_query = build_query(agent, query)

    decode_and_run(function_query)




st.title("Your Personal Data Analyst 👨‍💻")

st.write("Please upload your CSV file below.")

uploaded_file = st.file_uploader("Upload a CSV")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    # Display the first few rows of the dataset
    st.write("### Data Preview")
    st.write(df.head())

    query = st.text_area("Insert your query")

    local_vars = {}

    if st.button("Submit Query"):

        get_output_from_agent(df, query)







