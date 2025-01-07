from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    def __init__(self, model_name, model_config=None):
        self.model_config = model_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델과 토크나이저는 하위 클래스에서 초기화
        self.model = None
        self.tokenizer = None
        
        # 모델 로드
        self.load_model(model_name)
    
    @abstractmethod
    def load_model(self, model_name):
        """모델과 토크나이저를 로드합니다."""
        pass
    
    def to_device(self):
        """모델을 GPU/CPU로 이동합니다."""
        self.model.to(self.device)
    
    def prepare_inputs(self, text, max_length=None):
        """입력 텍스트를 모델 입력으로 변환합니다."""
        pass
    
    def generate(self, input_ids, **kwargs):
        """텍스트를 생성합니다."""
        pass
    
    def parameters(self):
        """모델의 파라미터를 반환합니다."""
        return self.model.parameters()
    
    def named_parameters(self):
        """모델의 이름이 있는 파라미터를 반환합니다."""
        return self.model.named_parameters()
    
    def train(self):
        """모델을 학습 모드로 설정합니다."""
        self.model.train()
    
    def eval(self):
        """모델을 평가 모드로 설정합니다."""
        self.model.eval()
    
    def gradient_checkpointing_enable(self):
        """그래디언트 체크포인팅을 활성화합니다."""
        self.model.gradient_checkpointing_enable() 