"""Contract tests for the v1 LLM chain predictor."""

from __future__ import annotations

import unittest

import pandas as pd

from machine_learning.LLM.baselines import best_single_optimizer
from machine_learning.LLM.data import build_candidate_chains
from machine_learning.LLM.evaluate import evaluate_selection, score_circuit_candidates
from machine_learning.LLM.features import extract_query_features
from machine_learning.LLM.objective import ObjectiveConfig
from machine_learning.LLM.predictor import predict_chain_for_circuit
from machine_learning.LLM.retrieval import TrainingCircuitRetriever
from machine_learning.LLM.schemas import validate_chain_selection


class LLMCoreTests(unittest.TestCase):
    def test_candidate_chain_generation_has_20_stable_entries(self):
        candidates = build_candidate_chains()
        self.assertEqual(len(candidates), 20)
        self.assertEqual(candidates[0]["chain_id"], "chain_01")
        self.assertEqual(candidates[-1]["chain_id"], "chain_20")

    def test_extract_query_features_uses_initial_state_columns(self):
        row = {
            "initial_num_qubits": 5,
            "initial_depth": 12,
            "initial_two_qubit_gates": 3,
            "initial_total_gates": 20,
            "initial_two_qubit_depth": 4,
            "initial_gate_density": 4.0,
            "initial_two_qubit_ratio": 0.15,
            "initial_num_interaction_edges": 6,
            "initial_avg_interaction_degree": 1.5,
            "initial_max_interaction_degree": 3,
            "initial_active_interaction_qubits": 4,
            "initial_active_interaction_qubit_fraction": 0.8,
            "initial_rz_gate_fraction": 0.2,
            "initial_sx_gate_fraction": 0.2,
            "initial_x_gate_fraction": 0.2,
            "initial_cx_gate_fraction": 0.4,
            "initial_two_qubit_gate_fraction": 0.4,
            "initial_depth_per_qubit": 2.4,
            "initial_gates_per_qubit": 4.0,
            "initial_two_qubit_gates_per_qubit": 0.6,
            "depth_per_qubit": 999,
        }
        extracted = extract_query_features(row)
        self.assertIn("initial_depth_per_qubit", extracted)
        self.assertNotIn("depth_per_qubit", extracted)

    def test_schema_validation_rejects_mismatched_chain(self):
        lookup = {candidate["chain_id"]: candidate for candidate in build_candidate_chains()}
        with self.assertRaises(ValueError):
            validate_chain_selection(
                {
                    "selected_chain_id": "chain_01",
                    "selected_chain": ["tket"],
                    "predicted_tradeoff_summary": "x",
                    "reasoning_summary": "y",
                    "confidence": 0.5,
                    "evidence_used": [],
                },
                lookup,
            )

    def test_retrieval_returns_training_neighbors_only(self):
        frame = pd.DataFrame(
            [
                {
                    "original_circuit": "a",
                    "chain_name": "qiskit_standard",
                    "total_two_qubit_gate_reduction_pct": 10,
                    "total_depth_reduction_pct": 5,
                    "total_runtime_seconds": 1,
                    "total_runtime_score": 0.1,
                    "current_step_two_qubit_gate_reduction_pct": 10,
                    "current_step_depth_reduction_pct": 5,
                    "runtime_score": 0.1,
                    "initial_num_qubits": 4,
                    "initial_depth": 10,
                    "initial_two_qubit_gates": 3,
                    "initial_total_gates": 20,
                    "initial_two_qubit_depth": 4,
                    "initial_gate_density": 5,
                    "initial_two_qubit_ratio": 0.15,
                    "initial_num_interaction_edges": 2,
                    "initial_avg_interaction_degree": 1.0,
                    "initial_max_interaction_degree": 1.0,
                    "initial_active_interaction_qubits": 2,
                    "initial_active_interaction_qubit_fraction": 0.5,
                    "initial_rz_gate_fraction": 0.2,
                    "initial_sx_gate_fraction": 0.2,
                    "initial_x_gate_fraction": 0.2,
                    "initial_cx_gate_fraction": 0.4,
                    "initial_two_qubit_gate_fraction": 0.4,
                    "initial_depth_per_qubit": 2.5,
                    "initial_gates_per_qubit": 5.0,
                    "initial_two_qubit_gates_per_qubit": 0.75,
                    "opt_1": "qiskit_standard",
                    "opt_2": None,
                    "opt_3": None,
                },
                {
                    "original_circuit": "b",
                    "chain_name": "qiskit_standard",
                    "total_two_qubit_gate_reduction_pct": 8,
                    "total_depth_reduction_pct": 4,
                    "total_runtime_seconds": 1.2,
                    "total_runtime_score": 0.12,
                    "current_step_two_qubit_gate_reduction_pct": 8,
                    "current_step_depth_reduction_pct": 4,
                    "runtime_score": 0.12,
                    "initial_num_qubits": 5,
                    "initial_depth": 11,
                    "initial_two_qubit_gates": 4,
                    "initial_total_gates": 22,
                    "initial_two_qubit_depth": 5,
                    "initial_gate_density": 4.4,
                    "initial_two_qubit_ratio": 0.18,
                    "initial_num_interaction_edges": 3,
                    "initial_avg_interaction_degree": 1.2,
                    "initial_max_interaction_degree": 2.0,
                    "initial_active_interaction_qubits": 3,
                    "initial_active_interaction_qubit_fraction": 0.6,
                    "initial_rz_gate_fraction": 0.15,
                    "initial_sx_gate_fraction": 0.25,
                    "initial_x_gate_fraction": 0.15,
                    "initial_cx_gate_fraction": 0.45,
                    "initial_two_qubit_gate_fraction": 0.45,
                    "initial_depth_per_qubit": 2.2,
                    "initial_gates_per_qubit": 4.4,
                    "initial_two_qubit_gates_per_qubit": 0.8,
                    "opt_1": "qiskit_standard",
                    "opt_2": None,
                    "opt_3": None,
                },
            ]
        )

        def score_lookup(circuit_id):
            subset = frame[frame["original_circuit"] == circuit_id].copy()
            subset["chain_id"] = "chain_01"
            subset["chain"] = [["qiskit_standard"]]
            subset["objective_score"] = [0.1]
            return subset[["chain_id", "chain_name", "chain", "objective_score", "total_two_qubit_gate_reduction_pct", "total_depth_reduction_pct", "total_runtime_seconds", "total_runtime_score"]]

        retriever = TrainingCircuitRetriever(
            train_frame=frame,
            feature_subset="basic_size",
            score_lookup_fn=score_lookup,
            k=1,
        )
        neighbors = retriever.retrieve(frame.iloc[0].to_dict())
        self.assertEqual(len(neighbors), 1)
        self.assertIn(neighbors[0].circuit_id, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
