
class NeuralSignal:
    """Represents a signal passed between components."""
    
    def __init__(self, source, signal_type, data, strength=1.0):
        self.source = source        # Source component
        self.signal_type = signal_type  # Type of signal
        self.data = data            # Signal payload
        self.strength = strength    # Signal strength (0-1)
        self.timestamp = time.time()  # When signal was created