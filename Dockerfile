FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="LatticeProbe"
LABEL description="Full reproducible environment for LatticeProbe (Ologunde 2026)"

WORKDIR /workspace

# System deps (git for pip editable installs, gcc for some wheel builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt pyproject.toml ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

# Copy scripts and tests after deps (better layer caching)
COPY scripts/ scripts/
COPY tests/   tests/
COPY docs/    docs/
COPY CHECKLIST.md REPRODUCE.md README.md ./

# Smoke-test: import core modules (fail fast if install is broken)
RUN python -c "from latticeprobe.ring import ntt, intt; from latticeprobe.sampler import generate_lwe_sample; print('Install OK')"

# Default: run the full test suite
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
