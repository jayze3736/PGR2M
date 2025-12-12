from abc import ABC, abstractmethod

class BaseLossWrapper(ABC):
    def __init__(self, loss_module, use_in_loss, weight=1.0):
        super().__init__()
        self.loss_module = loss_module
        self.use_in_loss = use_in_loss
        self.weight = weight
        self.avg_loss = 0.
        self.avg_vel_loss = 0.
    
    @abstractmethod
    def update(self, pred, gt, mask=None):
        pass

    @abstractmethod
    def state(self) -> dict:
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def return_weights(self):
        pass
    
    def is_use_in_loss(self):
        return self.use_in_loss