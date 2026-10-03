import json

from notebook_factory import ColabNotebookRequest, build_qcaf_notebook
from qcaf import QuantumAdmissibilityRequest, QuantumObservation


def _request():
    return QuantumAdmissibilityRequest(
        observations=[
            QuantumObservation(
                label="run-001",
                framework="pennylane",
                backend="default.mixed",
                target_probability=0.94,
                ber=0.03,
                noise_probability=0.04,
                circuit_depth=12,
                shots=8192,
                fidelity=0.97,
                seed=42,
            )
        ]
    )


def test_generated_notebook_is_valid_nbformat_shape():
    artifact = build_qcaf_notebook(
        ColabNotebookRequest(
            title="QCAF Test",
            qcaf_request=_request(),
            framework="hybrid",
            repository_ref="feature/qcaf-quantum-admissibility",
        )
    )

    notebook = json.loads(artifact.notebook_json)

    assert artifact.filename == "QCAF_Test.ipynb"
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["colab"]["name"] == artifact.filename
    assert len(notebook["cells"]) >= 6


def test_generated_notebook_preserves_interpretation_boundary():
    artifact = build_qcaf_notebook(
        ColabNotebookRequest(qcaf_request=_request())
    )
    notebook = json.loads(artifact.notebook_json)
    markdown = "".join(notebook["cells"][0]["source"])

    assert "finite set of supplied observations" in markdown
    assert "not a proof of global" in markdown


def test_install_cell_can_target_feature_branch():
    artifact = build_qcaf_notebook(
        ColabNotebookRequest(
            qcaf_request=_request(),
            repository_ref="feature/qcaf-quantum-admissibility",
        )
    )
    notebook = json.loads(artifact.notebook_json)
    install = "".join(notebook["cells"][1]["source"])

    assert "git+https://github.com/MarceloClaro/minizinc-mcp.git@feature/qcaf-quantum-admissibility" in install
    assert "minizinc" in install
