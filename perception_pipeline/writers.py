import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol


class MetadataWriter(Protocol):
    """Protocol for metadata writers."""
    
    def write(self, record: dict[str, Any]) -> None:
        """Write a single record."""
        ...
    
    def close(self) -> None:
        """Close the writer and flush any buffered data."""
        ...


class JsonlWriter:
    def __init__(self, output_path: str | Path, flush_every: int = 30):
        """
        Initialize the JSONL writer.
        
        Args:
            output_path: Path to output file (will be created/appended)
            flush_every: Flush to disk every N records
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._file = open(self.output_path, "a", buffering=1)
        self._count = 0
        self._flush_every = flush_every
    
    @property
    def record_count(self) -> int:
        """Number of records written."""
        return self._count
    
    def write(self, record: dict[str, Any]) -> None:
        """
        Write a single record as JSON line.
        
        Args:
            record: Dictionary to serialize and write
        """
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._count += 1
        
        if self._count % self._flush_every == 0:
            self._file.flush()
    
    def flush(self) -> None:
        """Force flush buffered data to disk."""
        if not self._file.closed:
            self._file.flush()
    
    def close(self) -> None:
        """Close the file handle."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
    
    def __enter__(self) -> "JsonlWriter":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NullWriter:
    """
    A no-op writer for testing or when output is disabled.
    """
    
    def __init__(self):
        self._count = 0
    
    @property
    def record_count(self) -> int:
        return self._count
    
    def write(self, record: dict[str, Any]) -> None:
        self._count += 1
    
    def flush(self) -> None:
        pass
    
    def close(self) -> None:
        pass
    
    def __enter__(self) -> "NullWriter":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
