from pathlib import Path
import hashlib
from src.config import PAPERS_DIR
import re
import pymupdf4llm
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# A real section heading
'''
It only locates headings.
Its job is to find the boundaries so you know where section III ends 
and IV begins. All the text stays.
'''
SECTION_RE = re.compile(
    r"^#{1,6}\s+\*{0,2}([IVXLC]+\.\s+.+?)\*{0,2}\s*$",
    re.MULTILINE,
)

# Everything from here to the end of the file is not worth indexing
'''
If we reach the ACKNOWLEDGMENT(S) or REFERENCES section,
treat that as the end of useful paper content.
'''
TAIL_RE = re.compile(
    r"^#{1,6}\s+\*{0,2}(ACKNOWLEDGMENTS?|REFERENCES)\b",
    re.MULTILINE | re.IGNORECASE,
)


def load_markdown(pdf_path):
    '''
    This function takes a PDF path and converts that PDF
    into Markdown text using pymupdf4llm.
    '''
    return pymupdf4llm.to_markdown(str(pdf_path))


def clean(md):
    '''
    This function cleans the extracted Markdown by removing
    unnecessary formatting and reducing extra blank lines.
    '''
    md = md.replace("**", "")                                              # Remove bold markers
    md = re.sub(r"(?<!\w)_|_(?!\w)", "", md)                   # Remove italic markers, keep underscores inside words
    md = re.sub(r"\n{3,}", "\n\n", md)                         # Collapse runs of blank lines to a single break
    return md


def split_sections(md):
    '''
    Takes cleaned Markdown text as input, removes the
    acknowledgments/references section, and splits the paper
    into separate sections using section headings.
    '''

    tail = TAIL_RE.search(md)                                               # Finds where ACKNOWLEDGMENTS or REFERENCES starts
    if tail:
        md = md[:tail.start()]                                              # Keeps only the useful content before that section

    matches = list(SECTION_RE.finditer(md))                                 # Finds all section headings like I. INTRODUCTION, II. METHODS

    if len(matches) < 3:
        return [("BODY", md.strip())]                                       # Not IEEE style, index the whole paper as one block
    sections = []

    for i, match in enumerate(matches):                                     # Loops through every detected section heading
        title = match.group(1).strip()                                      # Extracts and cleans the section title
        start = match.end()                                                 # Body starts immediately after the current heading
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)   # Ends at next heading, or document end
        body = md[start:end].strip()                                        # Extracts and cleans the section content

        if body:
            sections.append((title, body))                                  # Adds non-empty section title and body to the list

    return sections                                                         # Returns all extracted sections


def split_paragraphs(body):
    '''
    Takes a section body as input, splits it into separate
    paragraphs, removes extra spaces, and ignores very short paragraphs.
    '''

    parts = [p.strip() for p in body.split("\n\n")]                         # Splits section text at blank lines and removes extra spaces
    return [p for p in parts if len(p) > 40]                                # Keeps only paragraphs longer than 40 characters

def split_long(para):
    '''
    Splits an oversized paragraph on sentence boundaries so no
    single piece exceeds CHUNK_SIZE.
    '''

    if len(para) <= CHUNK_SIZE:
        return [para]                                            # If paragraph already fits, return it as one piece

    sentences = re.split(r"(?<=[.!?])\s+", para)          # Split paragraph into sentences after . ! or ?
    pieces = []                                                  # Stores the smaller paragraph pieces
    current = ""                                                 # Stores the piece currently being built

    for s in sentences:                                          # Loop through each sentence

        if not current:
            current = s                                          # Start a new piece with the first sentence

        elif len(current) + len(s) + 1 <= CHUNK_SIZE:
            current = current + " " + s                          # Add sentence if it still fits inside CHUNK_SIZE

        else:
            pieces.append(current)                               # Save the completed piece
            current = s                                          # Start a new piece with the current sentence

    if current:
        pieces.append(current)                                   # Save the final piece after the loop ends

    return pieces                                                # Return all smaller pieces


def chunk_section(body):
    '''
    Takes one section body as input and splits it into
    smaller chunks while keeping paragraph boundaries
    and adding overlap between consecutive chunks.
    '''

    chunks = []                                                                     # Stores all completed chunks
    current = ""                                                                    # Stores the chunk currently being built

    for para in [p for long in split_paragraphs(body) for p in split_long(long)]:   # Paragraphs, with oversized ones split on sentences

        if not current:
            current = para                                                          # If current chunk is empty, start it with this paragraph

        elif len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = current + "\n\n" + para                                       # Add paragraph if it still fits inside CHUNK_SIZE

        else:
            chunks.append(current)                                                  # Save the completed chunk

            tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""                # Take the last part of previous chunk as overlap
            tail = tail.split(" ", 1)[-1] if " " in tail else tail                  # Snap overlap to a whole word, no mid-word cuts

            current = (tail + "\n\n" + para).strip() if tail else para              # Start new chunk with overlap + new paragraph

    if current:
        chunks.append(current)                                                      # Save the final chunk after the loop ends

    return chunks                                                                   # Return the list of all created chunks

def file_hash(path):
    '''
    Returns a short hash of the file contents, used to detect
    whether a paper has already been indexed.
    '''

    h = hashlib.sha256()                                      # Create a SHA-256 hash object

    with open(path, "rb") as f:                               # Open the file in binary mode

        for block in iter(lambda: f.read(65536), b""):        # Read the file in 65,536-byte blocks until the file ends
            h.update(block)                                   # Add each block to the hash calculation

    return h.hexdigest()[:16]                                 # Convert hash to text and return only the first 16 characters



def iter_chunks(pdf_path):
    '''
    Parses one PDF and yields (text, payload) pairs, where payload
    carries the metadata needed for citations and filtering.
    '''

    pdf_path = Path(pdf_path)                                 # Convert the given PDF path into a Path object
    doc_id = file_hash(pdf_path)                              # Create a unique ID for this PDF using its file contents
    paper = pdf_path.stem                                     # Get the PDF filename without the .pdf extension

    md = clean(load_markdown(pdf_path))                       # Convert PDF to Markdown and clean the extracted text
    index = 0                                                 # Keeps track of the chunk number inside this paper

    for title, body in split_sections(md):                    # Loop through each detected section of the paper

        for text in chunk_section(body):                      # Split each section body into smaller searchable chunks

            payload = {                                       # Metadata stored together with each chunk
                "paper": paper,                               # Paper name
                "section": title,                             # Section where this chunk came from
                "chunk_index": index,                         # Unique chunk position inside this paper
                "doc_id": doc_id,                             # Unique identifier for the PDF
                "text": text,                                 # Actual chunk text
            }

            yield text, payload                               # Return one chunk and its metadata at a time
            index += 1                                        # Move to the next chunk number


def find_pdfs():
    '''
    Returns every PDF in the papers directory, sorted for stable ordering.
    '''

    return sorted(PAPERS_DIR.glob("*.pdf"))   # Find all .pdf files inside PAPERS_DIR and return them in sorted order



