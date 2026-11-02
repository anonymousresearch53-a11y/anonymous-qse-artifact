"""Circuit-id metadata helpers for llm_v3."""

from __future__ import annotations


def infer_circuit_family(circuit_id: str) -> str:
    """Infer a coarse circuit family from the circuit id."""
    value = str(circuit_id).strip().lower()

    if value.startswith("mqt_vqe_two_local"):
        return "mqt_vqe_two_local"
    if value.startswith("mqt_vqe_real_amp"):
        return "mqt_vqe_real_amp"
    if value.startswith("mqt_vqe"):
        return "mqt_vqe"
    if value.startswith("mqt_qft"):
        return "mqt_qft"
    if value.startswith("mqt_ghz"):
        return "mqt_ghz"
    if value.startswith("mqt_"):
        return "mqt_other"
    if value.startswith("guoq_qft"):
        return "guoq_qft"
    if value.startswith("guoq_portfolio"):
        return "guoq_portfolio"
    if value.startswith("guoq_qiskit"):
        return "guoq_qiskit"
    if value.startswith("guoq_"):
        return "guoq_other"
    if value.startswith("efficient_su2"):
        return "efficient_su2"
    if value.startswith("real_amplitudes"):
        return "real_amplitudes"
    if value.startswith("barenco_tof"):
        return "barenco_tof"
    return value.split("_", 1)[0] if "_" in value else value


def infer_circuit_source(circuit_id: str) -> str:
    """Infer a broader source bucket from the circuit id."""
    family = infer_circuit_family(circuit_id)
    if family.startswith("mqt_"):
        return "mqt"
    if family.startswith("guoq_"):
        return "guoq"
    return family
