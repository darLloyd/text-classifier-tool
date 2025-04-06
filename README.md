# Text Classification Tool

## Description

A web application that allows users to upload a CSV/Excel file, define custom labels along with their descriptions, then a classified version of the file will be downloaded locally after submission using an LLM backend (OpenAI GPT-4o Mini).

## Features

* Upload CSV or Excel files.
* Define custom classification labels (up to 50 chars: letters, numbers, spaces, underscore) and descriptions (up to 300 chars).
* Limit of 10 categories per classification task.
* Save and load category definitions locally in the browser (`localStorage`).
* Specify the column containing the text to classify (up to 50 chars: letters, numbers, spaces, underscore).
* Backend processing using OpenAI (`gpt-4o-mini-2024-07-18`).
* Provides justification from the LLM for each classification.
* Handles missing/empty text fields.
* Returns a downloadable CSV file with original text, assigned category, and justification.
* Real-time input validation for category definitions and column name.
* Backend validation for column name existence in the uploaded file.
* Option to clear the selected file.

## Setup (Local Development)

1.  **Clone the repository:**
    ```bash
    # Replace with your actual repository URL after creating it on GitHub
    git clone [https://github.com/darLloyd/your-repo-name.git](https://github.com/darLloyd/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Ensure Python 3.7+ is installed
    python -m venv venv
    # Activate the environment
    source venv/bin/activate  # macOS/Linux
    # OR
    venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes Flask, pandas, openai, python-dotenv, Flask-CORS, openpyxl)*
4.  **Create `.env` file:** Create a file named `.env` in the project root (`your-repo-name/`) and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=sk-YourActualOpenAIKeyHere...
    ```
5.  **Run the backend server:**
    ```bash
    # Set environment variable for Flask (do this once per terminal session or add to system variables)
    export FLASK_APP=app.py # macOS/Linux
    # OR set FLASK_APP=app.py # Windows CMD
    # OR $env:FLASK_APP = "app.py" # Windows PowerShell

    # Run the development server
    flask run
    ```
    The backend should now be running, typically on `http://127.0.0.1:5000`. Check the terminal output.

6.  **Run the frontend:** Open the `index.html` file (located in the project root) directly in your web browser (e.g., using File > Open or by double-clicking).

## Usage

1.  Open `index.html` in your browser.
2.  Click "Choose File" and select a CSV or Excel file. If you select the wrong file, click the "×" button to clear it.
3.  Enter the exact name of the column in your file that contains the text you want to classify (validation rules apply).
4.  Define at least one category by entering a Label and Description (validation rules apply). Use the "+ Add Category" button for more (up to 10).
    * *(Optional)* Use "Save Categories" to store definitions in your browser for later use.
    * *(Optional)* Use "Load Categories" to load previously saved definitions.
5.  Click "Submit for Classification".
6.  Wait for processing. Monitor the backend terminal for progress logs and potential errors from the LLM or file processing.
7.  If successful, a classified CSV file (`classified_output_....csv`) will be downloaded automatically. Check the "Status" area in the web app for messages or errors reported back from the backend.

## Future Enhancements

* Deploy frontend to static hosting (Netlify/Vercel).
* Deploy backend to PaaS (Render/PythonAnywhere).
* Implement backend rate limiting and row limits for hosted version.
* Add option to specify Excel sheet name if multiple sheets exist.
* More robust error handling and user feedback (e.g., specific parsing errors).
* Add unit/integration tests.
* (Add other ideas here)

