import torch
from transformers import get_linear_schedule_with_warmup

def create_optimizer(model, config):
    """옵티마이저 생성"""
    from torch.optim import AdamW
    
    # 학습 가능한 파라미터만 필터링
    params = [p for p in model.parameters() if p.requires_grad]
    
    if not params:
        raise ValueError("No parameters to optimize! Check model setup.")
    
    return AdamW(
        params,
        lr=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01)
    )

def create_scheduler(optimizer, num_training_steps, config):
    """학습률 스케줄러 생성"""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 500),
        num_training_steps=num_training_steps
    )

def setup_for_training(model, config):
    """학습을 위한 모델 설정"""
    print("\nSetting up model for training...")
    
    # 모델 구조 출력
    print("\nModel structure:")
    for name, _ in model.named_parameters():
        print(f"Layer: {name}")
    
    # 먼저 모든 파라미터를 프리즈
    for param in model.parameters():
        param.requires_grad = False
    
    # 파인튜닝 설정에 따라 특정 레이어 언프리즈
    if config.get("unfreeze_layers"):
        for layer_name in config.unfreeze_layers:
            found = False
            for name, param in model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    print(f"Unfreezing layer: {name}")
                    found = True
            if not found:
                print(f"Warning: Layer {layer_name} not found in model")
    else:
        # 파인튜닝 레이어가 지정되지 않은 경우 모든 파라미터를 학습 가능하도록 설정
        for param in model.parameters():
            param.requires_grad = True
    
    # 임베딩 레이어 프리즈 (설정된 경우)
    if config.get("freeze_embeddings", False):
        for name, param in model.named_parameters():
            if 'embed' in name:
                param.requires_grad = False
                print(f"Freezing embedding layer: {name}")
    
    # 그래디언트 체크포인팅 설정
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # 학습 가능한 파라미터 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel setup complete:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters! Check your fine-tuning configuration.") 