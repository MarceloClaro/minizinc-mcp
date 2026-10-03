from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import minizinc
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from notebook_factory import (
    ColabNotebookArtifact,
    ColabNotebookRequest,
    build_qcaf_notebook,
)
from qcaf import (
    QCAFResult,
    QuantumAdmissibilityRequest,
    build_quantum_admissibility_problem,
    classify_solver_status,
    evaluate_observation_checks,
)


class ConstraintModel(BaseModel):
    """Model for constraint problem definition."""

    model: str
    data: Optional[Dict[str, Any]] = None
    solver: str = "gecode"
    all_solutions: bool = False
    timeout: Optional[int] = None


class Solution(BaseModel):
    """Model for a single solution."""

    variables: Dict[str, Any]
    objective: Optional[float] = None
    is_optimal: bool = False


class SolveResult(BaseModel):
    """Model for solving results."""

    solutions: List[Solution]
    status: str
    solve_time: float
    num_solutions: int
    error: Optional[str] = None


class SolverInfo(BaseModel):
    """Model for solver information."""

    id: str
    name: str
    version: str
    tags: List[str]


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


async def solve_constraint_core(problem: ConstraintModel) -> "SolveResult":
    """Reusable async solver used by both the MCP tool and tests."""

    try:
        solver = minizinc.Solver.lookup(problem.solver)
        model = minizinc.Model()
        model.add_string(problem.model)
        instance = minizinc.Instance(solver, model)

        if problem.data:
            for key, value in problem.data.items():
                instance[key] = value

        if problem.timeout:
            result = await instance.solve_async(
                all_solutions=problem.all_solutions,
                time_limit=datetime.timedelta(seconds=problem.timeout),
            )
        else:
            result = await instance.solve_async(all_solutions=problem.all_solutions)

        solutions: List[Solution] = []

        if result.status in (minizinc.Status.SATISFIED, minizinc.Status.ALL_SOLUTIONS):
            if problem.all_solutions and result:
                for sol in result:
                    sol_dict: Dict[str, Any] = {}
                    for key in sol.__dict__:
                        if not key.startswith("_"):
                            sol_dict[key] = sol.__dict__[key]
                    solutions.append(
                        Solution(
                            variables=sol_dict,
                            objective=sol.objective if hasattr(sol, "objective") else None,
                            is_optimal=False,
                        )
                    )
            elif result.solution:
                sol_dict: Dict[str, Any] = {}
                for key in result.solution.__dict__:
                    if not key.startswith("_"):
                        sol_dict[key] = result.solution.__dict__[key]
                solutions.append(
                    Solution(
                        variables=sol_dict,
                        objective=result.objective if hasattr(result, "objective") else None,
                        is_optimal=result.status == minizinc.Status.OPTIMAL_SOLUTION,
                    )
                )
        elif result.status == minizinc.Status.OPTIMAL_SOLUTION:
            sol_dict: Dict[str, Any] = {}
            for key in result.solution.__dict__:
                if not key.startswith("_"):
                    sol_dict[key] = result.solution.__dict__[key]
            solutions.append(
                Solution(
                    variables=sol_dict,
                    objective=result.objective if hasattr(result, "objective") else None,
                    is_optimal=True,
                )
            )

        solve_time_value = 0.0
        if hasattr(result, "statistics") and "solveTime" in result.statistics:
            solve_time = result.statistics["solveTime"]
            if hasattr(solve_time, "total_seconds"):
                solve_time_value = solve_time.total_seconds()
            else:
                solve_time_value = float(solve_time)

        return SolveResult(
            solutions=solutions,
            status=str(result.status),
            solve_time=solve_time_value,
            num_solutions=len(solutions),
            error=None,
        )

    except Exception as exc:
        return SolveResult(
            solutions=[],
            status="ERROR",
            solve_time=0,
            num_solutions=0,
            error=str(exc),
        )


async def quantum_admissibility_core(
    request: QuantumAdmissibilityRequest,
) -> QCAFResult:
    """
    Evaluate a finite experimental slice produced by PennyLane/Qiskit/Cirq.

    The MiniZinc solver selects one supplied observation satisfying all declared
    thresholds. This intentionally does not claim anything about parameter
    combinations that were not supplied in observations.
    """

    try:
        generated = build_quantum_admissibility_problem(request)
    except Exception as exc:
        return QCAFResult(
            admissibility="ERROR",
            solver_status="PREPARATION_ERROR",
            objective=request.objective,
            error=str(exc),
        )

    generic_problem = ConstraintModel(**_model_dump(generated))
    solve_result = await solve_constraint_core(generic_problem)
    admissibility = classify_solver_status(solve_result.status)

    if admissibility != "ADMISSIBLE":
        return QCAFResult(
            admissibility=admissibility,
            solver_status=solve_result.status,
            objective=request.objective,
            solve_time=solve_result.solve_time,
            error=solve_result.error,
        )

    if not solve_result.solutions:
        return QCAFResult(
            admissibility="UNRESOLVED",
            solver_status=solve_result.status,
            objective=request.objective,
            solve_time=solve_result.solve_time,
            error="Solver reported satisfiable/optimal but returned no witness solution.",
        )

    selected_raw = solve_result.solutions[0].variables.get("selected")
    try:
        selected_one_based = int(selected_raw)
    except (TypeError, ValueError):
        return QCAFResult(
            admissibility="UNRESOLVED",
            solver_status=solve_result.status,
            objective=request.objective,
            solve_time=solve_result.solve_time,
            error="Solver witness did not contain a valid selected index.",
        )

    selected_zero_based = selected_one_based - 1
    if not 0 <= selected_zero_based < len(request.observations):
        return QCAFResult(
            admissibility="UNRESOLVED",
            solver_status=solve_result.status,
            objective=request.objective,
            solve_time=solve_result.solve_time,
            error="Solver returned a witness index outside the supplied observation set.",
        )

    selected = request.observations[selected_zero_based]
    checks = evaluate_observation_checks(selected, request.constraints)

    if not all(checks.values()):
        return QCAFResult(
            admissibility="UNRESOLVED",
            solver_status=solve_result.status,
            objective=request.objective,
            solve_time=solve_result.solve_time,
            selected_observation=selected,
            selected_index=selected_zero_based,
            checks=checks,
            error="Post-solve verification disagreed with the solver witness.",
        )

    return QCAFResult(
        admissibility="ADMISSIBLE",
        solver_status=solve_result.status,
        objective=request.objective,
        solve_time=solve_result.solve_time,
        selected_observation=selected,
        selected_index=selected_zero_based,
        checks=checks,
        error=None,
    )


def create_server():
    mcp = FastMCP(
        host="0.0.0.0",
        name="MiniZinc Constraint Solver MCP + QCAF",
        instructions=(
            "Solve general MiniZinc constraint problems and evaluate bounded "
            "quantum admissibility over reproducible PennyLane/Qiskit/Cirq observations."
        ),
    )

    @mcp.tool()
    async def solve_constraint(problem: ConstraintModel) -> SolveResult:
        """Solve a general MiniZinc constraint satisfaction or optimization problem."""

        return await solve_constraint_core(problem)

    @mcp.tool()
    async def quantum_admissibility(
        request: QuantumAdmissibilityRequest,
    ) -> QCAFResult:
        """
        Evaluate QCAF admissibility for a finite set of quantum observations.

        Typical inputs come from PennyLane or Qiskit runs and include target-state
        probability, BER, noise probability, circuit depth, shot count and
        optional fidelity.
        """

        return await quantum_admissibility_core(request)

    @mcp.tool()
    async def generate_qcaf_colab_notebook(
        request: ColabNotebookRequest,
    ) -> ColabNotebookArtifact:
        """Generate a portable Google Colab notebook for a QCAF experiment."""

        return build_qcaf_notebook(request)

    setattr(mcp, "solve_constraint", solve_constraint_core)
    setattr(mcp, "quantum_admissibility", quantum_admissibility_core)
    setattr(mcp, "generate_qcaf_colab_notebook", build_qcaf_notebook)

    return mcp


app = create_server()


def run_stdio():
    """Run as a local stdio MCP server."""
    app.run(transport="stdio")


def run_sse():
    """Run as an SSE MCP server for remote/container deployments."""
    app.run(transport="sse")


if __name__ == "__main__":
    run_sse()
