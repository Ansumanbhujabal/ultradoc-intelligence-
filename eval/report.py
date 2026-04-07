"""Evaluation metrics computation."""

from dataclasses import dataclass, field


@dataclass
class EvalReport:
    total_qa: int = 0
    correct_answers: int = 0
    source_hits: int = 0
    confidence_sum: float = 0.0
    correct_refusals: int = 0
    false_refusals: int = 0
    false_acceptances: int = 0
    total_extraction: int = 0
    field_matches: int = 0
    total_fields_checked: int = 0
    latencies: list = field(default_factory=list)

    @property
    def answer_accuracy(self) -> float:
        return self.correct_answers / self.total_qa if self.total_qa else 0.0

    @property
    def source_hit_rate(self) -> float:
        return self.source_hits / self.total_qa if self.total_qa else 0.0

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.total_qa if self.total_qa else 0.0

    @property
    def field_accuracy(self) -> float:
        return self.field_matches / self.total_fields_checked if self.total_fields_checked else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "qa": {"total": self.total_qa, "answer_accuracy": round(self.answer_accuracy, 3), "source_hit_rate": round(self.source_hit_rate, 3), "avg_confidence": round(self.avg_confidence, 3), "correct_refusals": self.correct_refusals, "false_refusals": self.false_refusals, "false_acceptances": self.false_acceptances},
            "extraction": {"total": self.total_extraction, "field_accuracy": round(self.field_accuracy, 3), "fields_checked": self.total_fields_checked, "fields_matched": self.field_matches},
            "performance": {"avg_latency_ms": round(self.avg_latency_ms, 1), "total_requests": len(self.latencies)},
        }

    def print_report(self):
        d = self.to_dict()
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"\nQ&A ({d['qa']['total']} questions):")
        print(f"  Answer Accuracy:   {d['qa']['answer_accuracy']:.1%}")
        print(f"  Source Hit Rate:   {d['qa']['source_hit_rate']:.1%}")
        print(f"  Avg Confidence:    {d['qa']['avg_confidence']:.2f}")
        print(f"  Correct Refusals:  {d['qa']['correct_refusals']}")
        print(f"  False Refusals:    {d['qa']['false_refusals']}")
        print(f"\nExtraction ({d['extraction']['total']} documents):")
        print(f"  Field Accuracy:    {d['extraction']['field_accuracy']:.1%}")
        print(f"  Fields Checked:    {d['extraction']['fields_checked']}")
        print(f"\nPerformance:")
        print(f"  Avg Latency:       {d['performance']['avg_latency_ms']:.0f}ms")
        print(f"  Total Requests:    {d['performance']['total_requests']}")
        print("=" * 50)
