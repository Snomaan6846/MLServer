DefaultBaseImage = "seldonio/mlserver:{version}-slim"

# Optional files checked in the build context to dynamically generate COPY
# instructions, ensuring compatibility with both Docker and Podman/Buildah.
CONDA_ENV_FILES = ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml")
CONFIG_FILES = ("settings.json", "model-settings.json", "requirements.txt")

DockerfileName = "Dockerfile"
DockerfileTemplateDevelopment = """
FROM continuumio/miniconda3:24.4.0-0 AS env-builder
SHELL ["/bin/bash", "-c"]

ARG MLSERVER_ENV_NAME="mlserver-custom-env" \\
    MLSERVER_ENV_TARBALL="./envs/base.tar.gz"

RUN conda config --add channels conda-forge && \\
    conda install conda-libmamba-solver==23.7.0 && \\
    conda config --set solver libmamba && \\
    conda install conda-pack
# Dynamically generated: COPY conda env files (e.g. environment.yml) if present
{conda_env_copy_instruction}
RUN mkdir $(dirname $MLSERVER_ENV_TARBALL); \\
    for envFile in environment.yml environment.yaml conda.yml conda.yaml; do \\
        if [ -f "$envFile" ]; then \\
            conda env create \
                --name $MLSERVER_ENV_NAME \\
                --file $envFile; \\
            conda-pack --ignore-missing-files \
                -n $MLSERVER_ENV_NAME \\
                -o $MLSERVER_ENV_TARBALL; \\
        fi \\
    done; \\
    chmod -R 776 $(dirname $MLSERVER_ENV_TARBALL)

FROM {base_image}
SHELL ["/bin/bash", "-c"]
# Dynamically generated: COPY conda tarball from env-builder stage if produced
{env_tarball_copy_instruction}
# Dynamically generated: COPY config files if present
{config_copy_instruction}

USER root
# Install dependencies system-wide, to ensure that they are available for every
# user and give permissions to (future) environment folder.
RUN ./hack/build-env.sh . && \\
    mkdir -p ./envs/base && \\
    chown -R 1000:0 ./envs/base && \\
    chmod -R 776 ./envs/base && \\
    rm -rf /root/.cache/pip
USER 1000

# Copy everything else
COPY . .

# Override MLServer's own `CMD` to activate the embedded environment.
# (optionally activating the hot-loaded one as well).
CMD source ./hack/activate-env.sh ./envs/base.tar.gz && \\
    mlserver start $MLSERVER_MODELS_DIR
"""

DockerfileTemplateProduction = """
FROM continuumio/miniconda3:24.4.0-0 AS env-builder
SHELL ["/bin/bash", "-c"]

ARG MLSERVER_ENV_NAME="mlserver-custom-env" \\
    MLSERVER_ENV_TARBALL="./envs/base.tar.gz"

RUN conda config --add channels conda-forge && \\
    conda install conda-libmamba-solver==23.7.0 && \\
    conda config --set solver libmamba && \\
    conda install conda-pack
# Dynamically generated: COPY conda env files (e.g. environment.yml) if present
{conda_env_copy_instruction}
RUN mkdir $(dirname $MLSERVER_ENV_TARBALL); \\
    for envFile in environment.yml environment.yaml conda.yml conda.yaml; do \\
        if [ -f "$envFile" ]; then \\
            conda env create \
                --name $MLSERVER_ENV_NAME \\
                --file $envFile; \\
            conda-pack --ignore-missing-files \
                -n $MLSERVER_ENV_NAME \\
                -o $MLSERVER_ENV_TARBALL; \\
        fi \\
    done; \\
    chmod -R 776 $(dirname $MLSERVER_ENV_TARBALL)

FROM {base_image}
SHELL ["/bin/bash", "-c"]
# Dynamically generated: COPY conda tarball from env-builder stage if produced
{env_tarball_copy_instruction}
# Dynamically generated: COPY config files if present
{config_copy_instruction}

USER root
# Install dependencies system-wide, to ensure that they are available for every
# user and give permissions to (future) environment folder.
RUN ./hack/build-env.sh . && \\
    mkdir -p ./envs/base && \\
    chown -R 1000:0 ./envs/base && \\
    chmod -R 776 ./envs/base && \\
    rm -rf /root/.cache/pip

# Persist trusted runtime allowlist in-image as read-only artifact.
RUN set -eu; \\
    artifact_path="{trusted_runtime_artifact_path}"; \\
    artifact_dir="$(dirname "$artifact_path")"; \\
    mkdir -p "$artifact_dir"; \\
    printf '%s\\n' '{trusted_runtime_allowlist_json}' > "$artifact_path"; \\
    chmod 0444 "$artifact_path"; \\
    chmod 0555 "$artifact_dir"
USER 1000

# Copy everything else
COPY . .
{custom_runtime_copy_instructions}
{custom_runtime_pythonpath_env}

# Override MLServer's own `CMD` to activate the embedded environment.
# (optionally activating the hot-loaded one as well).
CMD source ./hack/activate-env.sh ./envs/base.tar.gz && \\
    mlserver start $MLSERVER_MODELS_DIR
"""

DockerignoreName = ".dockerignore"
Dockerignore = """
# Binaries for programs and plugins
*.exe
*.exe~
*.dll
*.so
*.dylib
*.pyc
*.pyo
*.pyd
bin

# MLServer folders
.metrics
.envs

# Keep repo content from replacing image-owned bootstrap code / generated envs
/hack/
/envs/

# Mac file system
**/.DS_Store

# Python dev
__pycache__
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.mypy_cache
eggs/
.eggs/
*.egg-info/
./pytest_cache
.tox
build/
dist/

# Notebook Checkpoints
.ipynb_checkpoints

.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*,cover
*.log
.git
"""
