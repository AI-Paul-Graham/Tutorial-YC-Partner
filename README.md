<h1 align="center">Ask AI Paul Graham</h1>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Didn't get into YC? Don't worry—now you can ask AI Paul Graham for personalized startup advice, instantly available whenever you need it.

<!-- <div align="center">
  <img src="./assets/banner.png" width="700"/>
</div> -->

- **Design Doc:** [docs/design.md](docs/design.md)

- **Flow Source Code:** [flow.py](flow.py)

## I built this in just an hour, and you can, too.

- Built With [Pocket Flow](https://github.com/The-Pocket/PocketFlow), a 100-line LLM framework that lets LLM Agents (e.g., Cursor AI) build Apps for you
  
- Step-by-step YouTube development tutorial coming soon! [Subscribe for notifications](https://www.youtube.com/@ZacharyLLM?sub_confirmation=1).

## How to Run

1. Implement functions in the `utils/` directory based on your chosen APIs.

2. Install dependencies and run the program:

    ```bash
    pip install -r requirements.txt
    python main.py
    ```

3. To run the application server:

    ```bash
    streamlit run app.py
    ```

4. If you change the data in the `data/` directory or the `meta.csv` file, ensure you run the offline processing script to generate the index and metadata:

    ```bash
    python offline_processing.py
    ```
