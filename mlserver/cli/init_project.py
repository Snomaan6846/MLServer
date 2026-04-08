import shutil
import subprocess
from ..logging import logger


def init_cookiecutter_project(template: str):
    cookiecutter_path = shutil.which("cookiecutter")
    if cookiecutter_path:
        subprocess.run([cookiecutter_path, template], check=True, shell=False)
    else:
        logger.error(
            "The cookiecutter command is not found. \n\n"
            "Please install with 'pip install cookiecutter' and retry"
        )
