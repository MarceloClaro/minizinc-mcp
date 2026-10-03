from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from qcaf import QuantumAdmissibilityRequest, request_as_dict


NotebookFramework = Literal["qcaf", "pennylane", "qiskit", "hybrid"]


class ColabNotebookRequest(BaseModel):
    title: str = Field(default="QCAF Quantum Admissibility Experiment", min_length=1)
    qcaf_request: QuantumAdmissibilityRequest
    framework: NotebookFramework = "qcaf"
    repository: str = "MarceloClaro/minizinc-mcp"
    repository_ref: str = "main"
    include_install_cell: bool = True


class ColabNotebookArtifact(BaseModel):
    filename: str
    notebook_json: str
    nbformat: int = 4
    nbformat_minor: int = 5


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "qcaf_experiment"


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _install_cell(request: ColabNotebookRequest) -> str:
    extras = []
    if request.framework in ("pennylane", "hybrid"):
        extras.append("pennylane")
    if request.framework in ("qiskit", "hybrid"):
        extras.extend(["qiskit", "qiskit-aer"])

    git_url = (
        f"git+https://github.com/{request.repository}.git@{request.repository_ref}"
    )
    packages = " ".join([git_url, *extras])
    return (
        "# @title Install QCAF and scientific dependencies\n"
        "!apt-get -qq update >/dev/null\n"
        "!apt-get -qq install -y minizinc >/dev/null\n"
        f"%pip -q install {packages}\n"
        "print('QCAF environment ready.')\n"
    )


def build_qcaf_notebook(request: ColabNotebookRequest) -> ColabNotebookArtifact:
    payload = request_as_dict(request.qcaf_request)
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    payload_literal = repr(payload_json)

    cells = [
        _markdown(
            f"""# {request.title}

This notebook implements QCAF — Quantum Constraint Admissibility Framework.

The admissibility result applies only to the finite set of supplied observations
and declared thresholds. An INADMISSIBLE result is not a proof of global
physical or mathematical impossibility outside that bounded experimental slice.

Framework profile: {request.framework}.
"""
        )
    ]

    if request.include_install_cell:
        cells.append(_code(_install_cell(request)))

    cells.extend(
        [
            _code(
                """# @title Load QCAF
import json
from pprint import pprint

from main import quantum_admissibility_core
from qcaf import QuantumAdmissibilityRequest
"""
            ),
            _code(
                f"""# @title Reproducible QCAF request
request_payload = json.loads({payload_literal})
qcaf_request = QuantumAdmissibilityRequest(**request_payload)
pprint(request_payload)
"""
            ),
        ]
    )

    if request.framework in ("pennylane", "hybrid"):
        cells.append(
            _markdown(
                """## PennyLane integration

Generate the parameter and noise grid with PennyLane, normalize every run to
the QCAF observation schema, and replace or extend the observations field.
Recommended provenance fields include backend, seed, circuit hash, package
versions and variational parameters.
"""
            )
        )

    if request.framework in ("qiskit", "hybrid"):
        cells.append(
            _markdown(
                """## Qiskit integration

Generate counts or state metrics with Qiskit or Aer, convert the target
bitstring frequency to target_probability, record transpiled circuit depth and
append each reproducible run to the observation list before solving.
"""
            )
        )

    cells.extend(
        [
            _code(
                """# @title Solve bounded quantum admissibility
result = await quantum_admissibility_core(qcaf_request)
result_data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
pprint(result_data)
"""
            ),
            _code(
                """# @title Publication-oriented interpretation
print("Admissibility:", result.admissibility)
print("Solver status:", result.solver_status)
print("Objective:", result.objective)
print("Interpretation boundary:")
print(result.interpretation_boundary)

if result.selected_observation is not None:
    print("\\nSelected witness:")
    selected = (
        result.selected_observation.model_dump()
        if hasattr(result.selected_observation, "model_dump")
        else result.selected_observation.dict()
    )
    pprint(selected)
"""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": f"{_slugify(request.title)}.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "qcaf": {
                "framework": request.framework,
                "repository": request.repository,
                "repository_ref": request.repository_ref,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    filename = f"{_slugify(request.title)}.ipynb"
    return ColabNotebookArtifact(
        filename=filename,
        notebook_json=json.dumps(notebook, indent=2, ensure_ascii=False),
    )
