#!/usr/bin/env bash
set -euo pipefail

JDK_VERSION="${JDK_VERSION:-21.0.5+11}"
BASE_VERSION="${JDK_VERSION%%+*}"
BUILD_VERSION="${JDK_VERSION##*+}"
ARCHIVE_NAME="OpenJDK21U-jdk_x64_linux_hotspot_${BASE_VERSION}_${BUILD_VERSION}.tar.gz"
DOWNLOAD_TAG="jdk-${JDK_VERSION/+/%2B}"
DOWNLOAD_URL="https://github.com/adoptium/temurin21-binaries/releases/download/${DOWNLOAD_TAG}/${ARCHIVE_NAME}"
INSTALL_ROOT="${JAVA_INSTALL_PREFIX:-$HOME/.local/share/java}"

echo "Installing Temurin JDK ${JDK_VERSION} under ${INSTALL_ROOT}..."
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive_path="${tmpdir}/${ARCHIVE_NAME}"
echo "Downloading ${DOWNLOAD_URL}"
curl -L -o "${archive_path}" "${DOWNLOAD_URL}"

mkdir -p "${INSTALL_ROOT}"
tar -xzf "${archive_path}" -C "${INSTALL_ROOT}"

echo
echo "JDK installed to:"
ls -d "${INSTALL_ROOT}"/jdk-*
echo
echo "Add the following to your shell profile:"
echo "  export JAVA_HOME=\"${INSTALL_ROOT}/jdk-${JDK_VERSION}\""
echo "  export PATH=\"\$JAVA_HOME/bin:\$PATH\""
