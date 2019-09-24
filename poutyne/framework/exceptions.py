class ModelConfigurationError(Exception):
    """
    The exception raised when a model is misconfigured (e.g. missing properties, invalid properties).
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
