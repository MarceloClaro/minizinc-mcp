import pytest
from pydantic import ValidationError

from qcaf import (
    QuantumAdmissibilityRequest,
    QuantumConstraints,
    QuantumObservation,
    build_quantum_admissibility_problem,
    classify_solver_status,
    evaluate_observation_checks,
)


def observation(**overrides):
    data = {
        "label": "run-001",
        "framework": "pennylane",
        "backend": "default.mixed",
        "target_probability": 0.94,
        "ber": 0.03,
        "noise_probability": 0.04,
        "circuit_depth": 12,
        "shots": 8192,
        "fidelity": 0.97,
        "seed": 42,
        "parameters": {"theta": [0.3, 0.7]},
    }
    data.update(overrides)
    return QuantumObservation(**data)


def test_probability_validation():
    with pytest.raises(ValidationError):
        observation(target_probability=1.2)


def test_build_problem_uses_fixed_point_integer_data():
    request = QuantumAdmissibilityRequest(observations=[observation()])
    problem = build_quantum_admissibility_problem(request)

    assert problem.data["target_probability_ppm"] == [940000]
    assert problem.data["ber_ppm"] == [30000]
    assert problem.data["min_target_probability_ppm"] == 900000
    assert "var 1..n: selected;" in problem.model
    assert "solve satisfy;" in problem.model


def test_objective_compilation():
    request = QuantumAdmissibilityRequest(
        observations=[observation()],
        objective="maximize_target_probability",
    )
    problem = build_quantum_admissibility_problem(request)
    assert "solve maximize target_probability_ppm[selected];" in problem.model


def test_maximize_fidelity_requires_fidelity_for_every_observation():
    request = QuantumAdmissibilityRequest(
        observations=[observation(fidelity=None)],
        objective="maximize_fidelity",
    )
    with pytest.raises(ValueError, match="requires fidelity"):
        build_quantum_admissibility_problem(request)


@pytest.mark.parametrize(
    ("solver_status", "expected"),
    [
        ("SATISFIED", "ADMISSIBLE"),
        ("OPTIMAL_SOLUTION", "ADMISSIBLE"),
        ("ALL_SOLUTIONS", "ADMISSIBLE"),
        ("UNSATISFIABLE", "INADMISSIBLE"),
        ("UNKNOWN", "UNRESOLVED"),
        ("ERROR", "ERROR"),
    ],
)
def test_status_classification(solver_status, expected):
    assert classify_solver_status(solver_status) == expected


def test_postsolve_checks():
    constraints = QuantumConstraints(
        min_target_probability=0.90,
        max_ber=0.05,
        max_noise_probability=0.10,
        max_circuit_depth=20,
        min_shots=8192,
        min_fidelity=0.95,
    )
    checks = evaluate_observation_checks(observation(), constraints)
    assert all(checks.values())


def test_postsolve_checks_detect_failure():
    constraints = QuantumConstraints(max_ber=0.01)
    checks = evaluate_observation_checks(observation(), constraints)
    assert checks["ber"] is False
