import streamlit as st
import pandas as pd
from flow import *

def chunk_indices(n, chunk_size=10000):
    """
    Given the size of a text (n characters), split into chunks of length `chunk_size`.
    If the final chunk is < 50% of chunk_size, merge it with the previous chunk.
    Returns a list of (start_index, end_index) pairs.
    """
    if n <= 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end))
        start = end

    # Merge the last chunk if it's too small
    if len(chunks) > 1:
        last_start, last_end = chunks[-1]
        last_chunk_size = last_end - last_start
        if last_chunk_size < (0.5 * chunk_size):
            second_last_start, second_last_end = chunks[-2]
            merged_chunk = (second_last_start, last_end)
            chunks[-2] = merged_chunk
            chunks.pop()

    return chunks

def parse_embedding(embedding_str):
    """Parse embedding string back into numpy array"""
    # Extract values from TextEmbedding string using regex
    values_str = re.search(r'values=\[(.*?)\]', embedding_str)
    if values_str:
        # Convert string of numbers into float array
        values = [float(x) for x in values_str.group(1).split(',')]
        return np.array(values)
    return None

# Load the data
shared = {
    "meta_df": pd.read_csv("meta.csv"),
    "embeddings_df": pd.read_csv("embeddings.csv"),
}

shared["embeddings_df"]["embedding"] = shared["embeddings_df"]["embedding"].apply(parse_embedding)

# Read text contents and create chunk_texts
text_contents = {}
chunk_texts = {}

# Assuming texts are in a 'texts' directory with filenames like '1.txt', '2.txt'
for text_id in shared["meta_df"]["text_id"]:
    try:
        with open(f"./data/{text_id}.txt", "r", encoding="utf-8") as f:
            text = f.read()
            text_contents[text_id] = text
            
            # Create chunks for this text
            chunks = chunk_indices(len(text))
            for chunk_id, (start, end) in enumerate(chunks, 1):
                chunk_texts[(text_id, chunk_id)] = text[start:end]
    except FileNotFoundError:
        print(f"Warning: Text file for id {text_id} not found")

shared["chunk_texts"] = chunk_texts

st.session_state["meta_df"] = shared["meta_df"]

def block_page():
    st.markdown("""
        <style>
        #root > div:first-child {
            position: relative;
        }
        .blockPage {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(255,255,255,0.8);
            z-index: 999999;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .message {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
        </style>
        <div class="blockPage">
            <div class="spinner"></div>
            <div class="message">Working on your question. Should take &lt;1 min...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add JavaScript to ensure the overlay is on top of everything
    st.markdown("""
        <script>
        (function() {
            const blockPage = document.querySelector('.blockPage');
            document.body.appendChild(blockPage);
        })();
        </script>
    """, unsafe_allow_html=True)
    
import streamlit as st

def show_question_page():
    """
    Renders the question input page with an auto-populating dropdown.
    """
    st.success(
        "This project is fully open sourced on GitHub: "
        "[![GitHub (green)](https://badgen.net/badge/icon/YC-Partner-Agent?icon=github&label=&color=green)]"
        "(https://github.com/The-Pocket/YC-Partner-Agent)"
    )

    st.info(
        "We use Pocket Flow, a 100-line LLM framework: "
        "[![GitHub (purple)](https://badgen.net/badge/icon/PocketFlow?icon=github&label=&color=blue)]"
        "(https://github.com/The-Pocket/PocketFlow)"
    )
    
    st.markdown("<h1 style='text-align:center;'>AI YC Partner Agent</h1>", unsafe_allow_html=True)



        
    # Higher-quality example questions
    example_questions = {
        "Equity vs SAFE": {
            "question": "What are the main differences between equity and SAFE fundraising for early-stage startups?",
        },
        "Market Validation": {
            "question": "How can we validate market demand before launching a new B2B SaaS product?",
        },
        "User Acquisition": {
            "question": "What are the best strategies for user acquisition in the consumer social space?",
        },
        "Co-founder Relations": {
            "question": "What are the best practices for building a strong co-founder relationship?",
        }
    }

    # Initialize session state for user_question and selected_example
    if "user_question" not in st.session_state:
        st.session_state["user_question"] = ""
    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = "None"

    # 1. Text area for user's question
    user_question = st.text_area(
        "Ask your question (min. 10 chars):",
        value=st.session_state["user_question"],
        max_chars=5000,
        height=150
    )

    # 2. Dropdown to pick an example question
    # Build a list for the selectbox (first item is 'None')
    dropdown_options = ["None"] + list(example_questions.keys())

    # Determine default index (if user already picked something)
    if st.session_state["selected_example"] != "None":
        default_index = dropdown_options.index(st.session_state["selected_example"])
    else:
        default_index = 0

    selected_example = st.selectbox(
        "Or select an example question:",
        dropdown_options,
        index=default_index
    )

    # 3. If user changes the selection, update text area & re-run
    if selected_example != st.session_state["selected_example"]:
        st.session_state["selected_example"] = selected_example
        if selected_example != "None":
            st.session_state["user_question"] = example_questions[selected_example]["question"]
        st.rerun()

    # 5. Submit button
    if st.button("Submit", use_container_width=True, type="primary"):
        # Enforce minimum question length
        if len(user_question.strip()) < 10:
            st.warning("Please enter at least 10 characters.")
        else:
            block_page()
            
            # Pass the user's question to the shared dict
            shared["question"] = user_question.strip()
            partner_flow.run(shared)

            # Store results into session state so we can access them later
            st.session_state["user_question"] = user_question.strip()
            st.session_state["final_answer"] = shared["final_answer"]
            st.session_state["show_answer"] = True

            # Rerun to get to the answer page
            st.rerun()

def show_answer_page():
    """
    Renders the answer page using a chat-like interface.
    """
    st.markdown("<h1 style='text-align:center;'>Answer</h1>", unsafe_allow_html=True)

    # Retrieve from session state
    user_question = st.session_state.get("user_question", "")
    meta_df = st.session_state.get("meta_df", pd.DataFrame())
    final_answer = st.session_state.get("final_answer", {})

    # 1. Display the user's question in a "chat bubble"
    with st.chat_message("user"):
        st.write(user_question)

    # 2. Display the AI assistant's response:
    with st.chat_message("assistant"):
        # a) Show each citation in its own expander, expanded by default
        if "citations" in final_answer:
            for citation in final_answer["citations"]:
                row = meta_df[meta_df["text_id"] == citation["text_id"]].iloc[0]
                title = row["title"]
                link = row["link"]
                
                with st.expander(title, expanded=True):
                    st.markdown(f"[**Link**]({link})")  # Link as content
                    st.write(citation["citation"])

        # b) Now display the summary **below** the citations
        if "summary" in final_answer:
            st.write(final_answer["summary"])

    # 3. Button to go back and ask another question
    if st.button("Ask Another Question", use_container_width=True, type="primary"):
        st.session_state["show_answer"] = False
        st.rerun()

def main():
    """
    Main entry point for our Streamlit app.
    """
    st.set_page_config(
        page_title="AI YC Partner Agent",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("# Just checking if you are talking to users")
    st.sidebar.image("./images/meme.png", use_column_width=True)

    # Initialize session state
    if "show_answer" not in st.session_state:
        st.session_state["show_answer"] = False

    # Toggle between the question page and the answer page
    if st.session_state["show_answer"]:
        show_answer_page()
    else:
        show_question_page()

if __name__ == "__main__":
    main()
