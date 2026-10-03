import pytest

from main import quantum_admissibility_core
from qcaf import QuantumAdmissibilityRequest, QuantumConstraints, QuantumObservation


def obs(label, probability, ber, noise, depth, shots):
    return QuantumObservation(
        label=label,
        framework="qiskit",
        backend="aer_simulator",
        target_probability=probability,
        ber=ber,
        noise_probability=noise,
        circuit_depth=depth,
        shots=shots,
    )


@pytest.mark.asyncio
async def test_qcaf_admissible_witness():
    request = QuantumAdmissibilityRequest(
        observations=[
            obs("bad", 0.70, 0.10, 0.20, 30, 4096),
            obs("good", 0.95, 0.02, 0.05, 12, 8192),
        ],
        constraints=QuantumConstraints(
            min_target_probability=0.90,
            max_ber=0.05,
            max_noise_probability=0.10,
            max_circuit_depth=20,
            min_shots=8192,
        ),
    )

    result = await quantum_admissibility_core(request)

    assert result.admissibility == "ADMISSIBLE"
    assert result.selected_observation is not None
    assert result.selected_observation.label == "good"
    assert all(result.checks.values())


@pytest.mark.asyncio
async def test_qcaf_inadmissible_slice():
    request = QuantumAdmissibilityRequest(
        observations=[
            obs("run-a", 0.70, 0.10, 0.20, 30, 4096),
            obs("run-b", 0.80, 0.08, 0.15, 25, 4096),
        ]
    )

    result = await quantum_admissibility_core(request)

    assert result.admissibility == "INADMISSIBLE"
    assert result.selected_observation is None


@pytest.mark.asyncio
async def test_qcaf_optimization_selects_best_probability():
    request = QuantumAdmissibilityRequest(
        observations=[
            obs("good", 0.92, 0.03, 0.04, 10, 8192),
            obs("best", 0.97, 0.03, 0.04, 10, 8192),
        ],
        objective="maximize_target_probability",
    )

    result = await quantum_admissibility_core(request)

    assert result.admissibility == "ADMISSIBLE"
    assert result.selected_observation is not None
    assert result.selected_observation.label == "best"
