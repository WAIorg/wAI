"""
Service for streaming logs from background processing to frontend via SSE.
"""
import queue
import threading
from typing import Dict, Optional
from datetime import datetime


class LogStreamManager:
    """Manages log streams for different processing sessions."""
    
    def __init__(self):
        self._streams: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
    
    def create_stream(self, session_id: str) -> queue.Queue:
        """Create a new log stream queue for a session."""
        with self._lock:
            if session_id not in self._streams:
                self._streams[session_id] = queue.Queue()
            return self._streams[session_id]
    
    def get_stream(self, session_id: str) -> Optional[queue.Queue]:
        """Get the log stream queue for a session."""
        with self._lock:
            return self._streams.get(session_id)
    
    def log(self, session_id: str, message: str, log_type: str = "log"):
        """
        Add a log message to the stream.
        
        Args:
            session_id: Unique identifier for the processing session
            message: Log message
            log_type: Type of log (log, error, success, weight)
        """
        stream = self.get_stream(session_id)
        if stream:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": log_type,
                "message": message
            }
            print(f"[LOG MANAGER] Adding log to session {session_id}: {log_entry}")
            stream.put(log_entry)
        else:
            print(f"[LOG MANAGER] WARNING: No stream found for session {session_id}")
    
    def close_stream(self, session_id: str):
        """Close and remove a log stream."""
        with self._lock:
            if session_id in self._streams:
                stream = self._streams[session_id]
                stream.put(None)  # Signal end of stream
                del self._streams[session_id]


# Global instance
_log_manager = LogStreamManager()


def get_log_manager() -> LogStreamManager:
    """Get the global log stream manager."""
    return _log_manager
