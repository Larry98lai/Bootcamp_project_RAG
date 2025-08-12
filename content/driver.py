import os
import io
import streamlit as st
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from docx import Document
from crewai_tools import RagTool
import threading

def download_drive_files(folder_id, output_dir="downloaded_files"):
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Load credentials directly from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )

    # Build the Drive API service
    service = build('drive', 'v3', credentials=credentials)

    # Query all files in the folder
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])

    if not files:
        st.write("No files found in Google Drive folder.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    downloaded_files = []

    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_path = os.path.join(output_dir, file_name)

        # Only download supported file types
        if file['mimeType'] not in [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]:
            continue

        # Skip if already downloaded
        if os.path.exists(file_path):
            st.write(f"[SKIP] Already exists: {file_name}")
            downloaded_files.append(file_path)
            continue

        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        st.write(f"Downloaded: {file_name}")
        downloaded_files.append(file_path)

    return downloaded_files

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                print(f"[WARN] Skipping encrypted PDF: {file_path}")
                return ""
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except PdfReadError as e:
            print(f"[ERROR] Could not read PDF: {file_path} — {e}")
            return ""
        
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""



def simple_decode_path(raw_path):
    if raw_path.startswith("file:///"):
        raw_path = raw_path.replace("file:///", "")
    # Only decode spaces (%20) for now
    interim_decoded_path = raw_path.replace("%20", " ")

    # Normalize slashes (for Windows compatibility)
    decoded_path = interim_decoded_path.replace("\\", "/")

    # Resolve to absolute system path
    #decoded_path = os.path.normpath(normalized_path)
    return decoded_path


_rag_tool_cache = {}
_rag_tool_cache_lock = threading.Lock()


def build_rag_tool_from_files(file_paths):
    # Create a cache key based on file paths and their last modified times
    cache_key = tuple((os.path.abspath(p), os.path.getmtime(p)) for p in file_paths if os.path.exists(p))
    with _rag_tool_cache_lock:
        if cache_key in _rag_tool_cache:
            print("[CACHE] Returning cached RAG tool.")
            return _rag_tool_cache[cache_key]
    rag_tool = RagTool()
    # Prepare usable file paths and filter supported types first
    prepared_files = []
    for raw_path in file_paths:
        absolute_path = os.path.abspath(raw_path)
        usable_path = simple_decode_path(absolute_path)
        ext = os.path.splitext(usable_path)[1].lower()
        source_name = os.path.basename(usable_path)
        if ext not in [".pdf", ".txt"]:
            print(f"Skipping unsupported file type: {usable_path}")
            continue
        if not isinstance(usable_path, str):
            if isinstance(usable_path, bytes):
                usable_path = usable_path.decode('utf-8')
            else:
                usable_path = str(usable_path)
        prepared_files.append((usable_path, source_name))

    def extract_for_rag(args):
        usable_path, source_name = args
        print(f"[DEBUG] Adding to RAG: {usable_path} | source={source_name}")
        file_text = extract_text_from_file(usable_path)
        if not file_text.strip():
            print(f"[WARN] Skipping empty file: {usable_path}")
            return None
        return (file_text, source_name)

    # Use ThreadPoolExecutor for concurrent reading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_for_rag, prepared_files))

    for result in results:
        if result:
            file_text, source_name = result
            rag_tool.add(
                file_text,
                data_type="text",
                #source=source_name
            )
            print(f"[SUCCESS] Added to RAG: {source_name}")
    with _rag_tool_cache_lock:
        _rag_tool_cache[cache_key] = rag_tool
    return rag_tool