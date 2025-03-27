
from abc import ABC, abstractmethod

class ExpertInterface(ABC):
    """Base interface for all experts in the system."""
    
    @abstractmethod
    def process(self, input_data, context=None):
        """Process input data and produce output."""
        pass
    
    @property
    @abstractmethod
    def specialty(self):
        """Return this expert's specialty domain."""
        pass
    
    @abstractmethod
    def confidence(self, input_data, context=None):
        """Return confidence level for handling this input (0-1)."""
        pass