from rouge_score import rouge_scorer

class RougeCalculator:
    def __init__(self):
        self.evaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    def calculate_scores(self, references, hypotheses):
        """ROUGE 점수를 계산합니다."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.evaluator.score(ref, hyp)
            scores.append(score)
        return scores
    
    def calculate_avg_scores(self, scores):
        """ROUGE 점수 평균을 계산합니다."""
        avg_scores = {
            key: sum([score[key].fmeasure for score in scores]) / len(scores) 
            for key in scores[0]
        }
        return avg_scores 