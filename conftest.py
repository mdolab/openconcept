import os
import tempfile


def pytest_configure(config):
    """Write OpenMDAO's output directories to a temporary directory instead of the current directory."""
    if "OPENMDAO_WORKDIR" not in os.environ:
        config.openmdaoWorkDir = tempfile.TemporaryDirectory(prefix="openmdao_")
        os.environ["OPENMDAO_WORKDIR"] = config.openmdaoWorkDir.name


def pytest_unconfigure(config):
    if hasattr(config, "openmdaoWorkDir"):
        config.openmdaoWorkDir.cleanup()
