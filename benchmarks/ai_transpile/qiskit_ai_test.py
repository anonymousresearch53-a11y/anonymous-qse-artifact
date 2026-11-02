from transpilers import transpile_with_qiskit_ai
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.rz(0.5, 2)
qc.cx(0, 2)
qc.measure_all()

results = transpile_with_qiskit_ai(qc)

print(type(results))
print(len(results))

for r in results:
    print("label:", r.label)
    print("optimizer:", r.optimizer)
    print("metadata:", r.metadata)
    print("depth:", r.circuit.depth())
    print("2q gates:", r.circuit.num_nonlocal_gates())
    print("total gates:", r.circuit.size())
    print()