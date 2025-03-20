"""
This module provides functionality for evaluating and grading LLM answers based
on pre-defined criteria.
"""

from epam.auto_llm_eval.evaluator import EvaluationResult
from epam.auto_llm_eval.evaluator import evaluate_scenario
from epam.auto_llm_eval.evaluator import evaluate_scenario_by_id
from epam.auto_llm_eval.evaluator import evaluate_metric
from epam.auto_llm_eval.evaluator import grade_metric
from epam.auto_llm_eval.evaluation_report import (
    EvaluationReport,
    EvaluationStep
)
from epam.auto_llm_eval.ollama_adapter import get_ollama_model
from epam.auto_llm_eval.llamacpp_server import LlamaServerWrapper

__all__ = [
    "EvaluationResult",
    "evaluate_scenario",
    "evaluate_scenario_by_id",
    "evaluate_metric",
    "grade_metric",
    "EvaluationReport",
    "EvaluationStep",
    "get_ollama_model",
    "LlamaServerWrapper"
]
