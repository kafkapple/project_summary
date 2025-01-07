from transformers import BartTokenizer, BartForConditionalGeneration
from .base import BaseModel

class BartModel(BaseModel):
    def load_model(self, model_name):
        """BART 모델과 토크나이저를 로드합니다."""
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.to_device()
    
    def prepare_inputs(self, text, max_length=None):
        max_length = max_length or self.model_config.get("max_length", 1024)
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(
            input_ids,
            max_length=self.model_config.get("max_length", 1024),
            min_length=self.model_config.get("min_length", 50),
            num_beams=self.model_config.get("num_beams", 4),
            length_penalty=self.model_config.get("length_penalty", 2.0),
            no_repeat_ngram_size=self.model_config.get("no_repeat_ngram_size", 3)
        )
    
    def forward(self, **inputs):
        """모델의 forward pass를 수행합니다."""
        return self.model(**inputs) 