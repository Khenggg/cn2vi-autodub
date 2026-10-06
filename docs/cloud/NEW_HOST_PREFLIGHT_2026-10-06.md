# Cloud host preflight — 2026-10-06

## Workflow

Code will be written and synchronized from the local Windows machine. Environment checks and future project verification belong on this cloud host. Project implementation and project tests were paused pending the user's later project discussion.

## Host

- Ubuntu 22.04.5 LTS (Jammy), x86_64, Linux 6.8.0-136-generic.
- This is a container: PID 1 is `s6-svscan`; `/.dockerenv` exists.
- 12 online vCPUs; cgroup CPU quota reads `max`.
- About 16 GB RAM; cgroup memory limit reads `max`.
- Root filesystem is an overlay, 442 GB total with about 368 GB free at inspection.
- NVIDIA RTX 3060 visible to the container, 12,288 MiB; driver 580.173.02. This confirms device visibility only; no CUDA framework or model was installed or tested.

## Confirmed runnable tools

- Git 2.34.1 and curl 7.81.0 were present before setup.
- Node v24.14.1 and npm 11.11.0 are runnable. Node came from the official release archive and its SHA256 matched the official SHASUMS file.
- uv 0.9.6 is runnable. It installed isolated CPython 3.12.12, which runs through `uv run --no-project python --version`.
- wget 1.21.2, unzip 6.0, Python 3.10.12, pip 22.0.2, Docker CLI 29.1.3, and Docker Compose 2.40.3 respond to version commands.
- Docker has no `/var/run/docker.sock`; `docker info` cannot reach a daemon. The host is a container, and no nested daemon was started or configured.

## Incomplete package installation

Ubuntu packages were being installed with apt when the underlying overlay filesystem stalled. The remote apt/dpkg process was still active at capture; it was preserved. dpkg spent long periods in uninterruptible `D` state at `xlog_wait_on_iclog`. The apt log continued advancing slowly, so the package transaction was left intact.

At the captured status, these packages were configured and runnable: wget, unzip, Docker Compose. Other requested packages were present only in unpacked or trigger-pending state: ca-certificates, jq, FFmpeg, Python 3/pip/venv, tmux, and docker.io. The Docker CLI and Compose binary respond to version commands, but the Docker daemon is unavailable. FFmpeg currently fails to start because `liblapack.so.3` is missing while its dependencies remain unconfigured.

Remote sanitized status log: `/root/cn2vi-host-setup.log`.

## Scope

No project files were edited, no project tests were run, and no models or CUDA framework were downloaded. Complete the interrupted apt/dpkg transaction on the cloud host before relying on the Ubuntu packages above.
