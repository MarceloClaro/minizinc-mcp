from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCALE = 1_000_000
QCAF_VERSION = "QCAF/1.0"


class QuantumObservation(BaseModel):
    """One reproducible quantum experiment observation."""

    label: str = Field(min_length=1)
    framework: str = Field(default="other", min_length=1)
    backend: Optional[str] = None
    target_probability: float = Field(ge=0.0, le=1.0)
    ber: float = Field(ge=0.0, le=1.0)
    noise_probability: float = Field(ge=0.0, le=1.0)
    circuit_depth: int = Field(ge=0)
    shots: int = Field(ge=1)
    fidelity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seed: Optional[int] = None
    circuit_hash: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuantumConstraints(BaseModel):
    """Admissibility thresholds applied to every candidate observation."""

    min_target_probability: float = Field(default=0.90, ge=0.0, le=1.0)
    max_ber: float = Field(default=0.05, ge=0.0, le=1.0)
    max_noise_probability: float = Field(default=0.10, ge=0.0, le=1.0)
    max_circuit_depth: int = Field(default=20, ge=0)
    min_shots: int = Field(default=8192, ge=1)
    min_fidelity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


QuantumObjective = Literal[
    "satisfy",
    "maximize_target_probability",
    "minimize_ber",
    "maximize_fidelity",
]


class QuantumAdmissibilityRequest(BaseModel):
    """Finite-slice QCAF request built from PennyLane/Qiskit/Cirq observations."""

    observations: List[QuantumObservation]
    constraints: QuantumConstraints = Field(default_factory=QuantumConstraints)
    objective: QuantumObjective = "satisfy"
    solver: str = "gecode"
    timeout: Optional[int] = Field(default=None, ge=1)


class GeneratedConstraintProblem(BaseModel):
    """Generated MiniZinc problem compatible with the generic solver layer."""

    model: str
    data: Dict[str, Any]
    solver: str = "gecode"
    all_solutions: bool = False
    timeout: Optional[int] = None


class QCAFResult(BaseModel):
    framework_version: str = QCAF_VERSION
    admissibility: Literal["ADMISSIBLE", "INADMISSIBLE", "UNRESOLVED", "ERROR"]
    solver_status: str
    objective: QuantumObjective
    solve_time: float = 0.0
    selected_observation: Optional[QuantumObservation] = None
    selected_index: Optional[int] = None
    checks: Dict[str, bool] = Field(default_factory=dict)
    error: Optional[str] = None
    interpretation_boundary: str = (
        "The result applies only to the finite set of supplied observations and "
        "declared thresholds. INADMISSIBLE does not prove global impossibility "
        "outside that bounded experimental slice."
    )


def _scaled(value: float) -> int:
    return int(round(value * SCALE))


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def validate_request(request: QuantumAdmissibilityRequest) -> None:
    if not request.observations:
        raise ValueError("At least one quantum observation is required.")
    if request.objective == "maximize_fidelity":
        missing = [obs.label for obs in request.observations if obs.fidelity is None]
        if missing:
            raise ValueError(
                "maximize_fidelity requires fidelity for every observation; "
                f"missing: {', '.join(missing)}"
            )


def build_quantum_admissibility_problem(
    request: QuantumAdmissibilityRequest,
) -> GeneratedConstraintProblem:
    """Compile a bounded QCAF request into deterministic integer MiniZinc data."""

    validate_request(request)
    constraints = request.constraints
    observations = request.observations

    fidelity_values = [
        _scaled(obs.fidelity if obs.fidelity is not None else 0.0)
        for obs in observations
    ]

    data: Dict[str, Any] = {
        "n": len(observations),
        "target_probability_ppm": [_scaled(obs.target_probability) for obs in observations],
        "ber_ppm": [_scaled(obs.ber) for obs in observations],
        "noise_probability_ppm": [_scaled(obs.noise_probability) for obs in observations],
        "circuit_depth": [obs.circuit_depth for obs in observations],
        "shots": [obs.shots for obs in observations],
        "fidelity_ppm": fidelity_values,
        "min_target_probability_ppm": _scaled(constraints.min_target_probability),
        "max_ber_ppm": _scaled(constraints.max_ber),
        "max_noise_probability_ppm": _scaled(constraints.max_noise_probability),
        "max_circuit_depth": constraints.max_circuit_depth,
        "min_shots": constraints.min_shots,
        "enforce_fidelity": constraints.min_fidelity is not None,
        "min_fidelity_ppm": _scaled(constraints.min_fidelity or 0.0),
    }

    solve_statement = {
        "satisfy": "solve satisfy;",
        "maximize_target_probability": "solve maximize target_probability_ppm[selected];",
        "minimize_ber": "solve minimize ber_ppm[selected];",
        "maximize_fidelity": "solve maximize fidelity_ppm[selected];",
    }[request.objective]

    model = f"""
int: n;
array[1..n] of int: target_probability_ppm;
array[1..n] of int: ber_ppm;
array[1..n] of int: noise_probability_ppm;
array[1..n] of int: circuit_depth;
array[1..n] of int: shots;
array[1..n] of int: fidelity_ppm;

int: min_target_probability_ppm;
int: max_ber_ppm;
int: max_noise_probability_ppm;
int: max_circuit_depth;
int: min_shots;
bool: enforce_fidelity;
int: min_fidelity_ppm;

var 1..n: selected;

constraint target_probability_ppm[selected] >= min_target_probability_ppm;
constraint ber_ppm[selected] <= max_ber_ppm;
constraint noise_probability_ppm[selected] <= max_noise_probability_ppm;
constraint circuit_depth[selected] <= max_circuit_depth;
constraint shots[selected] >= min_shots;
constraint (not enforce_fidelity) \/ (fidelity_ppm[selected] >= min_fidelity_ppm);

{solve_statement}
""".strip()

    return GeneratedConstraintProblem(
        model=model,
        data=data,
        solver=request.solver,
        all_solutions=False,
        timeout=request.timeout,
    )


def classify_solver_status(status: str) -> Literal[
    "ADMISSIBLE", "INADMISSIBLE", "UNRESOLVED", "ERROR"
]:
    normalized = status.upper()
    if "UNSATISFIABLE" in normalized or normalized == "UNSAT":
        return "INADMISSIBLE"
    if "OPTIMAL" in normalized or "SATISFIED" in normalized or "ALL_SOLUTIONS" in normalized:
        return "ADMISSIBLE"
    if "ERROR" in normalized:
        return "ERROR"
    return "UNRESOLVED"


def evaluate_observation_checks(
    observation: QuantumObservation,
    constraints: QuantumConstraints,
) -> Dict[str, bool]:
    checks = {
        "target_probability": (
            observation.target_probability >= constraints.min_target_probability
        ),
        "ber": observation.ber <= constraints.max_ber,
        "noise_probability": (
            observation.noise_probability <= constraints.max_noise_probability
        ),
        "circuit_depth": observation.circuit_depth <= constraints.max_circuit_depth,
        "shots": observation.shots >= constraints.min_shots,
    }
    if constraints.min_fidelity is not None:
        checks["fidelity"] = (
            observation.fidelity is not None
            and observation.fidelity >= constraints.min_fidelity
        )
    return checks


def request_as_dict(request: QuantumAdmissibilityRequest) -> Dict[str, Any]:
    """Compatibility helper for Pydantic v1/v2 clients."""

    return _model_dump(request)
