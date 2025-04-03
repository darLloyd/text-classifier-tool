# app.py
import os
import io
import json
import re
import pandas as pd # Needs openpyxl installed for Excel support
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import openai # Assuming OpenAI setup

# Load environment variables
load_dotenv()
print("Attempted to load environment variables from .env")

app = Flask(__name__)

# CORS Configuration
CORS(app)
print("Flask-CORS initialized.")

# Configure OpenAI Client (Same as backend_code_v3_openai)
openai_client = None
if openai:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("\n*** WARNING: OPENAI_API_KEY not set. ***\n")
    else:
        try:
            openai_client = openai.OpenAI()
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"\n*** ERROR: Failed to initialize OpenAI client: {e} ***\n")
            openai_client = None
else:
    print("\n*** WARNING: OpenAI library not imported. ***\n")

# --- LLM Classification Function (classify_text_with_llm) ---
# ... (This function remains exactly the same as in backend_code_v3_openai) ...
def classify_text_with_llm(text_to_classify, categories):
    if not openai_client: return {"label": "error", "justification": "OpenAI client not available."}
    if not text_to_classify: return {"label": "no category", "justification": "Input text was empty."}
    category_definitions = "\n".join([f"- **{cat['label']}**: {cat['description']}" for cat in categories])
    system_prompt = f"""
You are a text classification assistant. Your task is to classify the user's text based *only* on the categories provided below.
**Available Categories:**
{category_definitions}
**Instructions:** 1. Read text and descriptions. 2. Choose best matching label or "no category". 3. Provide brief justification.
**Output Format:** Respond *only* with:
Label: <Chosen Label or "no category">
Justification: <Your brief explanation>
"""
    user_prompt = f"Please classify the following text:\n\n\"{text_to_classify}\""
    print(f"\n--- Sending Prompt to OpenAI for text starting: '{text_to_classify[:60]}...' ---")
    try:
        model_name = "gpt-4o-mini-2024-07-18"
        response = openai_client.chat.completions.create( model=model_name, messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ], temperature=0.5, max_tokens=100 )
        print("--- Received OpenAI Response ---")
        response_text = response.choices[0].message.content.strip()
        label, justification = "error", "Could not parse LLM response."
        label_match = re.search(r"^Label:\s*(.*)", response_text, re.MULTILINE | re.IGNORECASE)
        justification_match = re.search(r"^Justification:\s*(.*)", response_text, re.MULTILINE | re.IGNORECASE)
        if label_match:
            label = label_match.group(1).strip()
            valid_labels = [cat['label'] for cat in categories] + ['no category']
            if label.lower() not in [vl.lower() for vl in valid_labels]: print(f"Warning: OpenAI returned invalid label '{label}'.") # label = "no category" # Optional: Force valid label
        if justification_match: justification = justification_match.group(1).strip()
        if label == "error" and response_text: justification = f"Failed parse. Raw: {response_text[:100]}..."
        print(f"Parsed Label: '{label}', Justification: '{justification[:60]}...'")
        return {"label": label, "justification": justification}
    except openai.APIError as e: print(f"!!! OpenAI API Error: {e}"); justification = f"OpenAI API Error: {e}"
    except openai.APIConnectionError as e: print(f"!!! OpenAI Connection Error: {e}"); justification = f"OpenAI Connection Error: {e}"
    except openai.RateLimitError as e: print(f"!!! OpenAI Rate Limit Error: {e}"); justification = f"OpenAI Rate Limit Error: {e}"
    except Exception as e: print(f"!!! ERROR during OpenAI call/parsing: {e}"); justification = f"LLM API call failed: {e}"
    return {"label": "error", "justification": justification}


# --- API Endpoint for Classification ---
@app.route('/classify', methods=['POST'])
def classify_csv():
    """
    Handles POST request to classify text in uploaded CSV or Excel file.
    """
    print("\n=== Received request on /classify ===")
    # 1. --- Input Validation --- (Same as before)
    if 'csv_file' not in request.files: return jsonify({"error": "No file part in the request"}), 400
    file = request.files['csv_file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not request.form.get('text_column'): return jsonify({"error": "Text column name not provided"}), 400
    if not request.form.get('categories'): return jsonify({"error": "Categories definition not provided"}), 400

    text_column_name = request.form['text_column']
    categories_json_string = request.form['categories']
    try: # Category validation (same)
        categories_list = json.loads(categories_json_string)
        if not isinstance(categories_list, list) or not categories_list: raise ValueError("Categories must be a non-empty list.")
        for item in categories_list:
            if not isinstance(item, dict) or 'label' not in item or 'description' not in item: raise ValueError("Invalid category item structure.")
            if not item['label'].strip() or not item['description'].strip(): raise ValueError("Category label/description empty.")
        print(f"Received {len(categories_list)} valid categories.")
    except (json.JSONDecodeError, ValueError) as e: return jsonify({"error": f"Invalid categories format: {e}"}), 400

    # 2. --- ** UPDATED File Parsing (CSV or Excel) ** ---
    filename = file.filename
    df = None # Initialize DataFrame as None
    print(f"Attempting to parse file: {filename}")

    try:
        # Check file extension to determine parsing method
        if filename.lower().endswith('.csv'):
            print("Detected CSV file. Using pd.read_csv...")
            # Use io.BytesIO for read_csv as well, more robust
            file_stream = io.BytesIO(file.stream.read())
            # Try decoding with utf-8-sig first, fallback to others if needed
            try:
                 df = pd.read_csv(file_stream, encoding='utf-8-sig')
            except UnicodeDecodeError:
                 print("UTF-8-sig decode failed, trying standard UTF-8...")
                 file_stream.seek(0) # Reset stream position
                 df = pd.read_csv(file_stream, encoding='utf-8')
            except Exception as csv_e: # Catch other potential CSV errors
                 raise ValueError(f"CSV parsing error: {csv_e}") from csv_e

        elif filename.lower().endswith(('.xls', '.xlsx')):
            print(f"Detected Excel file ({filename.split('.')[-1]}). Using pd.read_excel...")
            # Read directly from the file stream (BytesIO is needed for read_excel)
            file_stream = io.BytesIO(file.stream.read())
            # By default, reads the first sheet. Add sheet_name=None to read all sheets
            # or sheet_name='SheetName' / sheet_name=0 for specific sheet.
            df = pd.read_excel(file_stream, engine='openpyxl')
            print(f"Successfully parsed Excel file. Sheet(s) read.") # Note: Reads only first sheet by default

        else:
            # Unsupported file type
            print(f"Error: Unsupported file type: {filename}")
            allowed_types = ".csv, .xls, .xlsx"
            return jsonify({"error": f"Unsupported file type. Please upload {allowed_types}."}), 400

        # --- Common checks after parsing ---
        print(f"Parsed DataFrame shape: {df.shape}")
        if df.empty and len(df.columns) == 0:
             raise pd.errors.EmptyDataError("File contains no data or headers.")
        if text_column_name not in df.columns:
            available_columns = list(df.columns)
            print(f"Error: Column '{text_column_name}' not found. Available: {available_columns}")
            return jsonify({"error": f"Column '{text_column_name}' not found.", "available_columns": available_columns}), 400

    except UnicodeDecodeError as e: # Catch potential decoding errors not caught by fallback
         print(f"Error decoding file: {e}")
         return jsonify({"error": f"File encoding error: {e}. Please ensure file is UTF-8 (for CSV) or a standard Excel format."}), 400
    except pd.errors.EmptyDataError as e:
         print(f"Error: File is empty or contains no data: {e}")
         return jsonify({"error": f"File contains no data: {e}"}), 400
    except ImportError: # Specifically catch if openpyxl is missing
         print("Error: Missing 'openpyxl' library required for Excel files.")
         return jsonify({"error": "Processing Excel files requires the 'openpyxl' library. Please install it (`pip install openpyxl`) and restart the server."}), 500
    except Exception as e: # Catch other potential parsing errors (e.g., corrupted Excel)
        print(f"Error parsing file: {type(e).__name__} - {e}")
        return jsonify({"error": f"Error reading file: {e}"}), 400

    # 3. --- Processing Loop & LLM Calls --- (Same as before)
    results_labels = []
    results_justifications = []
    row_count = 0
    if df.empty: print("DataFrame has headers but no data rows. Skipping processing loop.")
    else:
        print(f"Starting OpenAI classification for {len(df)} rows...")
        # ... (Rest of the loop is identical to backend_code_v3_openai) ...
        try:
            for index, row in df.iterrows():
                 row_count += 1
                 text_to_classify_raw = row[text_column_name]
                 text_to_classify = str(text_to_classify_raw) if pd.notna(text_to_classify_raw) else ""
                 if not text_to_classify.strip(): label, justification = "no category", "Text field was empty."
                 else: llm_result = classify_text_with_llm(text_to_classify, categories_list); label = llm_result.get("label", "error"); justification = llm_result.get("justification", "Error retrieving justification.")
                 results_labels.append(label); results_justifications.append(justification)
                 if row_count % 10 == 0 or row_count == len(df): print(f"  Processed {row_count}/{len(df)} rows...")
        except Exception as e: print(f"!!! Error during processing loop near row {row_count}: {e}"); return jsonify({"error": f"Processing error near row {row_count}: {e}"}), 500
        print("Finished OpenAI classification processing.")

    # Add results back to the DataFrame
    df['Assigned Category'] = results_labels
    df['Justification'] = results_justifications

    # 4. --- Generate Output CSV --- (Same as before)
    try:
        output_csv_stream = io.StringIO()
        if 'Assigned Category' not in df.columns: df['Assigned Category'] = []
        if 'Justification' not in df.columns: df['Justification'] = []
        df.to_csv(output_csv_stream, index=False, encoding='utf-8')
        csv_data = output_csv_stream.getvalue()
        output_csv_stream.close()
        print("Generated output CSV data.")
    except Exception as e: print(f"Error generating output CSV: {e}"); return jsonify({"error": f"Failed to generate output CSV: {e}"}), 500

    # 5. --- Return CSV Response --- (Same as before)
    print("Sending CSV response.")
    return Response( csv_data, mimetype="text/csv", headers={ "Content-Disposition": "attachment;filename=classified_output.csv", "Content-Type": "text/csv; charset=utf-8" } )

# Basic route (Same)
@app.route('/')
def index(): return "Backend server is running with OpenAI integration (CSV/Excel support)!"

# Main execution block (Same)
if __name__ == '__main__':
    if not openai_client: print("\n--- NOTE: Server starting, but OpenAI client not configured. '/classify' will fail. ---\n")
    app.run(debug=True, port=5000)
