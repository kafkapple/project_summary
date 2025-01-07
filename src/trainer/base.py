from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, train_data, val_data, **kwargs):
        """모델 학습을 수행합니다."""
        pass
    
    @abstractmethod
    def evaluate(self, model, eval_data):
        """모델을 평가합니다."""
        pass 