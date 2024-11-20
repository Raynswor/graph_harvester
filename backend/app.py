# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


import os
import re

import jsonpickle
import requests
import uvicorn
from detect_graph import harvest_graph
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kieta_modules.pipeline import PipelineManager

app = FastAPI(
    title="GraphHarvester API",
    description="An API to upload a PDF file, process it, and return graph data.",
    version="1.0.0",
)

allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PipelineManager
pipeline_manager: PipelineManager = PipelineManager()
pipeline_file = os.getenv("PIPELINE_FILE", "pipeline.json")
pipeline_manager.read_from_file(pipeline_file)


def pickle(obj) -> str:
    """
    Serializes a Python object into a JSON-compatible string.

    Args:
        obj: The Python object to serialize.

    Returns:
        str: A JSON-compatible string representation of the object.
    """
    return jsonpickle.encode(obj, make_refs=False, unpicklable=False)


##
# Standard route - file upload
##
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a PDF file, processes it to extract graph data, and returns the result.

    Args:
        file (UploadFile): A PDF file to be uploaded and processed.

    Returns:
        JSONResponse: Contains the processed graph data as JSON.

    Raises:
        HTTPException: Raised if file type is not PDF or processing fails.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF files are allowed."
        )

    # Read file content
    file_content = await file.read()
    file_id = "".join(file.filename.split(".")[:-1])

    # Create document payload for processing
    doc = {
        "file": file_content,
        "id": file_id,
        "suffix": "pdf",
    }

    # Process the file
    try:
        result = pipeline_manager.get_pipeline("GraphExtraction").process_full(doc)
        graph_data = harvest_graph(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # Return the graph data as JSON
    return JSONResponse(content=graph_data)


##
# Alternative route - arXiv link processing
##
def is_arxiv_link(url: str) -> bool:
    """
    Validates if a given URL is a valid arXiv link.

    Args:
        url (str): URL to be validated.

    Returns:
        bool: True if the URL is a valid arXiv link, False otherwise.
    """
    return bool(re.match(r"^https?://arxiv\.org/abs/[0-9]+\.[0-9]+(.pdf)?$", url))


def download_arxiv_pdf(arxiv_url: str) -> bytes:
    """
    Downloads a PDF from an arXiv link.

    Args:
        arxiv_url (str): The arXiv URL of the PDF to download.

    Returns:
        bytes: The content of the downloaded PDF.

    Raises:
        HTTPException: If downloading the PDF fails.
    """
    pdf_url = arxiv_url.replace("/abs/", "/pdf/") + (".pdf" if not arxiv_url.endswith(".pdf") else "")
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(
            status_code=500, detail="Failed to download PDF from arXiv link."
        )


# @app.post("/process-arxiv")
# async def process_arxiv_link(arxiv_url: str = Body(..., embed=True)):
#     """
#     Processes a PDF file from a given arXiv link, extracts graph data, and returns the result.

#     Args:
#         arxiv_url (str): The arXiv URL of the paper to download and process.

#     Returns:
#         JSONResponse: Contains the processed graph data as JSON.

#     Raises:
#         HTTPException: Raised if the URL is not a valid arXiv link or if processing fails.
#     """
#     if not is_arxiv_link(arxiv_url):
#         raise HTTPException(
#             status_code=400, detail="Invalid URL. Please provide a valid arXiv link."
#         )

#     try:
#         file_content = download_arxiv_pdf(arxiv_url)
#         file_id = arxiv_url.split("/")[-1]

#         doc = {
#             "file": file_content,
#             "id": file_id,
#             "suffix": "pdf",
#         }

#         result = pipeline_manager.get_pipeline("GraphExtraction").process_full(doc)
#         graph_data = harvest_graph(result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

#     return JSONResponse(content=graph_data)


# Start the server (when run as a standalone script)
if __name__ == "__main__":
    """
    Runs the application with Uvicorn as the ASGI server.

    This starts a FastAPI app on the specified host and port,
    making the `/upload` endpoint available for PDF processing.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
