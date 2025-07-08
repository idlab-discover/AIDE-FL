from enum import Enum
from typing import Optional, Callable

class MessageType(Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"

class Messenger:
    _instance: Optional['Messenger'] = None
    _message_sender: Optional[Callable] = None
    _origin: str = "preprocessor"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Messenger, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def init(cls, message_sender: Callable):
        """Initialise the messenger with kafka service"""
        instance = cls()
        instance._message_sender = message_sender
        return instance
    
    @classmethod
    def get_messenger(cls) -> 'Messenger':
        if cls._instance is None or cls._instance._message_sender is None:
            raise RuntimeError("Messenger is not initialised")
        return cls._instance
    
    def _send_message(self, message_type: MessageType, message: str):
        """Internal method to send message via the callback"""
        if self._message_sender is None:
            raise RuntimeError("Messenger is not initialised")
        
        general_message = {
            "type": message_type.value,
            "origin": self._origin,
            "message": message
        }
        
        self._message_sender(general_message)
    
    def error(self, message: str):
        """Send error message"""
        self._send_message(MessageType.ERROR, message)
        
    def warn(self, message: str):
        """Send warn message"""
        self._send_message(MessageType.WARN, message)
        
    def info(self, message: str):
        """Send info message"""
        self._send_message(MessageType.INFO, message)
        
    def debug(self, message: str):
        """Send debug message"""
        self._send_message(MessageType.DEBUG, message)
    
    
# Global convenience functions
def send_error(message: str):
    """Send error message globally"""
    Messenger.get_messenger().error(message)
    
def send_warn(message: str):
    """Send warn message globally"""
    Messenger.get_messenger().warn(message)
    
def send_info(message: str):
    """Send info message globally"""
    Messenger.get_messenger().info(message)
    
def send_debug(message: str):
    """Send debug message globally"""
    Messenger.get_messenger().debug(message)