# Stage 1: ANTs install
FROM ubuntu:noble-20260509.1@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS ants

# DL4006: fail pipelines on the first failing command
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## OS dependencies
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        ca-certificates=20260601* \
        wget=1.21.4* \
        unzip=6.0* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

## ANTs 2.4.3
RUN wget --progress=dot:giga https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-centos7-X64-gcc.zip -O /opt/ants.zip \
 && echo "80b449b96fdd215c83cc1c4b261c76c42939b321a69a31b2e2ad0b9b8d2e9837  /opt/ants.zip" | sha256sum -c - \
 && unzip /opt/ants.zip -d /opt/ \
 && rm /opt/ants.zip

# MI->GC in the linear stages makes registration reproducible; killme.sh is unused here.
# sed exits 0 on a stale line number, hence the asserts. Notices: Apache-2.0 4(b), see NOTICE.
RUN [ "$(sed -n '169p;170p;186p;213p' /opt/ants-2.4.3/bin/ANTSpexec.sh | grep -c 'killme\.sh')" = 4 ] \
 && [ "$(sed -n '1158p;1517p;1523p' /opt/ants-2.4.3/bin/antsMultivariateTemplateConstruction2.sh | grep -c ' -m MI\[.*,32,Regular')" = 3 ] \
 && sed -i -e '169s/^/# /' -e '170s/^/# /' -e '186s/^/# /' -e '213s/^/# /' /opt/ants-2.4.3/bin/ANTSpexec.sh \
 && sed -i '1a# Modified from upstream ANTs by the DELTA-SVD project; see NOTICE.' /opt/ants-2.4.3/bin/ANTSpexec.sh \
 && sed -i -e '1158s/ -m MI/ -m GC/' \
        -e '1158s/,32,Regular/,1,Regular/' \
        -e '1517s/ -m MI/ -m GC/' \
        -e '1517s/,32,Regular/,1,Regular/' \
        -e '1523s/ -m MI/ -m GC/' \
        -e '1523s/,32,Regular/,1,Regular/' /opt/ants-2.4.3/bin/antsMultivariateTemplateConstruction2.sh \
 && sed -i '1a# Modified from upstream ANTs by the DELTA-SVD project; see NOTICE.' /opt/ants-2.4.3/bin/antsMultivariateTemplateConstruction2.sh

# Stage 2
FROM ubuntu:noble-20260509.1@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS final

# DL4006: fail pipelines on the first failing command
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="DELTA-SVD" \
    org.opencontainers.image.authors="Anna Dewenter, Benno Gesierich, Marco Duering" \
    org.opencontainers.image.licenses="CC-BY-NC-ND-4.0" \
    org.opencontainers.image.source="https://github.com/isdneuroimaging/DELTA-SVD" \
    org.opencontainers.image.url="https://github.com/isdneuroimaging/DELTA-SVD" \
    org.opencontainers.image.description="Ready-to-use container image for DELTA-SVD" \
    org.opencontainers.image.documentation="https://delta-svd.com"

ARG DEBIAN_FRONTEND=noninteractive

## OPENBLAS_CORETYPE pins the BLAS kernel. The bundled libopenblas is a
## DYNAMIC_ARCH build, so it otherwise picks kernels from the CPU's feature set
## at load time, and that reaches the metrics: np.linalg.pinv in the free-water
## fit differs between kernel families, which moved delta-PSMD by ~9% between
## Nehalem and Haswell on the reference subject. Haswell is the family the
## method was validated on. It needs AVX2/FMA (x86-64-v3); delta-svd.py checks
## for those and reports it, because OpenBLAS itself would die with SIGILL.
ENV OPENBLAS_CORETYPE="Haswell" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    PATH="/opt/scripts:/opt/conda/bin:/opt/ants-2.4.3/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLDIR="/opt/conda" \
    ANTSPATH="/opt/ants-2.4.3/bin" \
    MPLCONFIGDIR="/tmp/matplotlib" \
    QT_QPA_PLATFORM="offscreen"

## OS dependencies, locale (Apptainer needs it generated), nonroot user
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        ca-certificates=20260601* \
        dc=1.07.1* \
        locales=2.39* \
        wget=1.21.4* \
 && useradd --system --no-create-home --shell /usr/sbin/nologin nonroot \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && dpkg-reconfigure --frontend=noninteractive locales \
 && update-locale LANG="en_US.UTF-8" \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

## Conda
RUN wget --progress=dot:giga https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-0-Linux-x86_64.sh -O /opt/miniconda.sh \
 && echo "a921abd74e16f5dee8a4d79b124635fac9b939c465ba2e942ea61b3fcd1451d8  /opt/miniconda.sh" | sha256sum -c - \
 && bash /opt/miniconda.sh -b -p /opt/conda \
 && rm /opt/miniconda.sh \
 && conda clean -afy

## Explicit lock: exact URLs, no solve. Read its header before regenerating.
## The sed disables tbss_1_preproc's slicesdir report, guarded by the line-number check.
COPY container/conda-explicit-linux-64.txt /opt/conda-explicit-linux-64.txt
RUN conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r \
 && conda install -y -p /opt/conda --file /opt/conda-explicit-linux-64.txt \
 && conda clean -afy \
 && [ "$(sed -n '105,109p' /opt/conda/bin/tbss_1_preproc | grep -c 'slicesdir\|grot')" = 4 ] \
 && sed -i -e '105s/^/# /' -e '106s/^/# /' -e '107s/^/# /' -e '108s/^/# /' -e '109s/^/# /' /opt/conda/bin/tbss_1_preproc \
 && find /opt/conda/data/standard -type f \( -name 'FSL*' -o -name 'MNI152*' \) -delete \
 && rm -rf /opt/conda/data/standard/bianca /opt/conda/data/standard/tissuepriors

## The FSL Licence permits redistribution only with the source included (see NOTICE).
## Versions come from the installed env so they cannot drift from what is shipped.
RUN set -e \
 && mkdir -p /opt/fsl-source \
 && for name in flirt fnirt avwutils tbss; do \
        ver=$(conda list -p /opt/conda -f "fsl-${name}" | tail -n1 | awk '{print $2}'); \
        wget --progress=dot:giga \
            --tries=5 --waitretry=10 --retry-connrefused \
            "https://git.fmrib.ox.ac.uk/fsl/${name}/-/archive/${ver}/${name}-${ver}.tar.gz" \
            -O "/opt/fsl-source/${name}-${ver}.tar.gz"; \
        sleep 3; \
    done \
 && apt-get purge -y --auto-remove wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

## ANTs from stage 1
WORKDIR /opt/ants-2.4.3/bin
COPY --from=ants /opt/ants-2.4.3/bin/antsMultivariateTemplateConstruction2.sh \
    /opt/ants-2.4.3/bin/antsAI \
    /opt/ants-2.4.3/bin/antsApplyTransforms \
    /opt/ants-2.4.3/bin/ANTSpexec.sh \
    /opt/ants-2.4.3/bin/antsRegistration \
    /opt/ants-2.4.3/bin/antsRegistrationSyNQuick.sh \
    /opt/ants-2.4.3/bin/AverageAffineTransform \
    /opt/ants-2.4.3/bin/AverageImages \
    /opt/ants-2.4.3/bin/ImageMath \
    /opt/ants-2.4.3/bin/MeasureMinMaxMean \
    /opt/ants-2.4.3/bin/MultiplyImages \
    /opt/ants-2.4.3/bin/N4BiasFieldCorrection \
    /opt/ants-2.4.3/bin/waitForPBSQJobs.pl \
    /opt/ants-2.4.3/bin/waitForSGEQJobs.pl \
    /opt/ants-2.4.3/bin/waitForSlurmJobs.pl \
    /opt/ants-2.4.3/bin/waitForXGridJobs.pl \
    /opt/ants-2.4.3/bin/WarpImageMultiTransform ./

## Kept out of /opt/scripts so they aren't read as covering the DELTA-SVD scripts.
COPY LICENSE NOTICE /opt/
COPY container/third-party-licenses/ /opt/third-party-licenses/

## Pipeline scripts, plus the two FSL-derived reference images they need at runtime,see NOTICE.
WORKDIR /opt/scripts
COPY container/scripts/ .

## Read by delta_svd_version.py for --version. A file, not an ENV: Apptainer
## forwards the host environment in, which could then override it.
COPY VERSION ./VERSION

USER nonroot
ENTRYPOINT [ "python", "/opt/scripts/delta-svd.py" ]

## Provenance from container/build.sh; Defaults allow a bare build. 'unknown',
## not a version-shaped default: a bare build still copies the real VERSION
## above, and a label contradicting it would be worse than an obvious blank.
ARG VERSION=unknown
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.version="$VERSION" \
    org.opencontainers.image.revision="$VCS_REF" \
    org.opencontainers.image.created="$BUILD_DATE"
