from pathlib import Path
from pypdf import PdfReader

def read_document(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if file_path.suffix.lower() in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8").strip()

    raise ValueError(f"Unsupported document type: {file_path.suffix}")
