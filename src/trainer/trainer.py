import torch
from tqdm import tqdm
from .base import BaseTrainer
from src.data import create_dataloader
from src.metrics import RougeCalculator
from src.models import setup_for_training, create_optimizer, create_scheduler

class DialogueTrainer(BaseTrainer):
    def __init__(self, training_config=None, fine_tuning_config=None):
        self.training_config = training_config or {}
        self.fine_tuning_config = fine_tuning_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rouge_calculator = RougeCalculator()
    
    def train(self, model, train_data, val_data, **kwargs):
        # 모델 설정
        setup_for_training(model, self.fine_tuning_config)
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = create_optimizer(model, self.training_config)
        scheduler = create_scheduler(optimizer, len(train_data), self.training_config)
        
        # 학습 루프
        for epoch in range(self.training_config.get("epochs", 3)):
            model.train()
            total_loss = 0
            
            train_dataloader = create_dataloader(
                train_data, 
                model.tokenizer, 
                self.training_config.get("batch_size", 8),
                model.model_config.get("max_length", 1024)
            )
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                loss = self._training_step(model, batch, optimizer, scheduler)
                total_loss += loss
            
            # 검증
            val_loss = self._validate(model, val_data)
            
            print(f"Epoch {epoch+1}/{self.training_config.get('epochs', 3)}")
            print(f"Average training loss: {total_loss/len(train_dataloader):.4f}")
            print(f"Validation loss: {val_loss:.4f}")
    
    def _training_step(self, model, batch, optimizer, scheduler):
        """단일 학습 스텝 수행"""
        # 데이터를 GPU로 이동
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model.forward(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.training_config.get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.training_config.max_grad_norm
            )
        
        # Optimization step
        optimizer.step()
        scheduler.step()
        
        return loss.item()
    
    def _validate(self, model, val_data):
        """검증 데이터에 대한 평가 수행"""
        model.eval()
        val_dataloader = create_dataloader(
            val_data, 
            model.tokenizer, 
            self.training_config.get("batch_size", 8),
            model.model_config.get("max_length", 1024),
            shuffle=False
        )
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model.forward(**batch)
                total_loss += outputs.loss.item()
                
        return total_loss / len(val_dataloader)
    
    def evaluate(self, model, eval_data):
        """모델 평가를 수행합니다."""
        model.eval()
        references = []
        hypotheses = []
        
        for sample in tqdm(eval_data, desc="Evaluating"):
            dialogue = sample["dialogue"]
            references.append(sample["summary"])
            
            inputs = model.prepare_inputs(dialogue)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(inputs["input_ids"])
            hypothesis = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            hypotheses.append(hypothesis)
        
        scores = self.rouge_calculator.calculate_scores(references, hypotheses)
        return self.rouge_calculator.calculate_avg_scores(scores) 