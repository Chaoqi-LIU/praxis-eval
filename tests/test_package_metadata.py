# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import metadata, version

import praxis_eval

PACKAGE_NAME = "praxis-eval"
EXPECTED_VERSION = "0.1.3"
EXPECTED_EMAIL = "liuchaoqi730@gmail.com"
EXPECTED_HOMEPAGE = "https://chaoqi-liu.com"
EXPECTED_SOURCE = "https://github.com/Chaoqi-LIU/praxis-eval"
EXPECTED_DOCUMENTATION = "https://chaoqi-liu.com/praxis-eval/"


def test_version_matches_distribution_metadata() -> None:
    assert version(PACKAGE_NAME) == EXPECTED_VERSION
    assert praxis_eval.__version__ == EXPECTED_VERSION


def test_distribution_metadata_identifies_project_owner() -> None:
    package_metadata = metadata(PACKAGE_NAME)

    assert package_metadata["Author"] == "Chaoqi Liu"
    assert package_metadata["Author-email"] == EXPECTED_EMAIL
    assert package_metadata["Maintainer"] == "Chaoqi Liu"
    assert package_metadata["Maintainer-email"] == EXPECTED_EMAIL
    assert package_metadata["Home-page"] == EXPECTED_HOMEPAGE


def test_distribution_metadata_has_standard_project_urls() -> None:
    project_urls = set(metadata(PACKAGE_NAME).get_all("Project-URL") or [])

    assert f"Homepage, {EXPECTED_HOMEPAGE}" in project_urls
    assert f"Source, {EXPECTED_SOURCE}" in project_urls
    assert f"Issues, {EXPECTED_SOURCE}/issues" in project_urls
    assert f"Documentation, {EXPECTED_DOCUMENTATION}" in project_urls


def test_remote_extra_accepts_current_minor_line() -> None:
    requirements = metadata(PACKAGE_NAME).get_all("Requires-Dist") or []

    assert any(
        requirement.startswith("praxis-remote")
        and ">=0.1.0" in requirement
        and "<0.2.0" in requirement
        and 'extra == "remote"' in requirement
        for requirement in requirements
    )
