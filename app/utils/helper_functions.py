import hashlib

def _get_doc_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()
