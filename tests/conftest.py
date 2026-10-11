"""Test isolation: every test session runs against a fresh temporary SSA_HOME with a synthetic
OULAD fixture. Production data, models, index, feedback DB and LLM cache are never touched,
and no real LLM is called."""
import os

os.environ["SSA_DISABLE_DOTENV"] = "1"
for _k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "LLM_PROVIDER",
           "CLAUDE_MODEL_FAST", "CLAUDE_MODEL_MAIN", "CLAUDE_MODEL_JUDGE", "LLM_PRICE_TABLE", "LLM_ENABLED", "SSA_DATA_DIR", "SSA_ARTIFACT_DIR"):
    os.environ.pop(_k, None)
os.environ.update({"CHECKPOINTS": "0.25,0.40", "CUTOFF_FRACTION": "0.40", "STUDENT_ASSISTANT_ADMIN_KEY": "admin-key-0123456789",
                   "SSA_DEV_MODE": "0", "STUDENT_ASSISTANT_API_KEY": "test-key-0123456789abcdef", "ENABLE_TEXT_EMBEDDINGS": "0",
                   "LLM_CACHE_ENABLED": "0", "RATE_LIMIT_PER_MINUTE": "1000", "LOG_LEVEL": "WARNING"})

import pytest  # noqa: E402

API_KEY = os.environ["STUDENT_ASSISTANT_API_KEY"]


@pytest.fixture(scope="session")
def ssa_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("ssa_home")
    os.environ["SSA_HOME"] = str(home)
    from ssa.config import reset_settings
    reset_settings()
    return home


@pytest.fixture(scope="session")
def built(ssa_home):
    from ssa.cli import main
    from ssa.config import get_settings
    main(["demo-data", "--n", "120"])
    try:
        main(["build", "--no-dense", "--queries", "120"])
    except SystemExit as exc:  # the gate result is asserted in test_quality_gate
        assert exc.code in (0, None, 1)
    return get_settings()


@pytest.fixture(scope="session")
def service(built):
    from ssa.service import StudentSuccessService
    return StudentSuccessService(built).warm()


@pytest.fixture(scope="session")
def features(service):
    return service.features


@pytest.fixture(scope="session")
def handlers(built, service):
    from ssa.api.handlers import Handlers
    return Handlers(built, service=service)


def ref_of(features, scope=1, presentation="2014J"):
    r = features[(features["in_scope"] == scope) & (features["code_presentation"] == presentation)].iloc[0]
    return {"id_student": int(r["id_student"]), "code_module": r["code_module"], "code_presentation": r["code_presentation"]}
