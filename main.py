import os
import hydra
from omegaconf import DictConfig
from src.models.bart import BartModel
from src.trainer.trainer import DialogueTrainer
from src.data import download_dialogsum, load_jsonl

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 데이터 다운로드 및 로드
    dataset_config = cfg.dataset.dialogsumm
    download_dialogsum(dataset_config.data_dir)
    
    train_data = load_jsonl(os.path.join(dataset_config.data_dir, dataset_config.files.train))
    val_data = load_jsonl(os.path.join(dataset_config.data_dir, dataset_config.files.validation))
    
    # 모델 초기화
    model_config = cfg.model.bart
    model = BartModel(
        model_name=model_config.name,
        model_config=model_config.model_config
    )
    
    # 트레이너 초기화
    trainer = DialogueTrainer(
        training_config=model_config.training,
        fine_tuning_config=model_config.fine_tuning
    )
    
    # 학습 수행 (선택적)
    if cfg.get("do_train", True):
        trainer.train(model, train_data, val_data)
    
    # 평가 수행
    print("\nEvaluating model...")
    val_scores = trainer.evaluate(model, val_data[:10])  # 10개 샘플로 테스트
    print("Validation ROUGE scores:", val_scores)

if __name__ == "__main__":
    main()
