class Model:
    """
    A parent class for models.
    
    This class provides a template for model implementations with a name,
    and methods fit and predict that should be overridden in subclasses.
    """
    
    def __init__(self, name: str = "Model"):
        """
        Initialize the model with a name.
        
        Args:
            name (str): Name of the model. Defaults to "Model".
        """
        if name:
            self.name = name
        else:
            self.name = "Model"

    def fit(self, data):
        """
        Fit the model to the training data.
        
        Args:
            data: Training data used to fit the model.
        """
        raise NotImplementedError("The fit() method must be implemented by subclasses.")

    def predict(self, data):
        """
        Predict output using the model based on new data.
        
        Args:
            data: New data for which predictions are to be made.
        
        Returns:
            Predictions based on the model.
        """
        raise NotImplementedError("The predict() method must be implemented by subclasses.")

    def __str__(self) -> str:
        """
        Return a string representation of the model.
        """
        return f"{self.__class__.__name__} (name={self.name})"