import logging
import tempfile
import os
from pathlib import Path
from agno.media import Image as AgnoImage
from typing import List
import streamlit as st

# Only record ERROR
logger = logging.getLogger("emotional_recovery")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def process_images(files) -> List[AgnoImage]:
    """Convert the Streamlit uploaded file into an AgnoImage list"""
    processed = []
    for file in files:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}")
            tmp.write(file.getvalue())
            tmp.close()
            processed.append(AgnoImage(filepath=Path(tmp.name)))
        except Exception as e:
            logger.error(f"Error processing image {file.name}: {e}")
    return processed

