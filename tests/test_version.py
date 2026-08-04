"""Version reporting: the lookup itself, and the two entry points that expose it.

The version is the one piece of provenance a user has to quote when comparing or
pooling results, so 'unknown' leaking into a release image would be a silent
failure - hence the checks that the real checkout resolves to the real VERSION.
"""

import re
from pathlib import Path

import pytest

import delta_svd_version

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "VERSION"


# ---------------------------------------------------------------------------
# read_version()

def test_version_resolves_in_a_source_checkout():
    # the scripts sit two levels below the repo root, so the fallback candidate
    # is the one that has to fire here; 'unknown' would mean it broke
    expected = VERSION_FILE.read_text().strip()
    assert delta_svd_version.__version__ == expected
    assert delta_svd_version.__version__ != delta_svd_version.VERSION_UNKNOWN


def test_version_looks_like_a_release_number():
    assert delta_svd_version.__version__.count(".") == 2


def test_sibling_version_file_wins(tmp_path):
    # the image layout: the Dockerfile copies VERSION next to the scripts
    (tmp_path / "VERSION").write_text("1.2.3\n")
    assert delta_svd_version.read_version(str(tmp_path)) == "1.2.3"


def test_falls_back_two_levels_up(tmp_path):
    # the checkout layout: <root>/VERSION with the scripts in <root>/a/b
    (tmp_path / "VERSION").write_text("4.5.6\n")
    scripts = tmp_path / "container" / "scripts"
    scripts.mkdir(parents=True)
    assert delta_svd_version.read_version(str(scripts)) == "4.5.6"


def test_unknown_when_no_version_file_is_found(tmp_path):
    scripts = tmp_path / "container" / "scripts"
    scripts.mkdir(parents=True)
    assert delta_svd_version.read_version(str(scripts)) == delta_svd_version.VERSION_UNKNOWN


def test_an_empty_version_file_is_skipped(tmp_path):
    # a truncated sibling must not shadow a good file further up
    (tmp_path / "VERSION").write_text("4.5.6\n")
    scripts = tmp_path / "container" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "VERSION").write_text("   \n")
    assert delta_svd_version.read_version(str(scripts)) == "4.5.6"


def test_read_version_never_raises_on_a_directory(tmp_path):
    # VERSION existing but not being a readable file must not abort a run
    (tmp_path / "VERSION").mkdir()
    assert delta_svd_version.read_version(str(tmp_path)) == delta_svd_version.VERSION_UNKNOWN


# ---------------------------------------------------------------------------
# --version / --help on both entry points

def _version_output(parser, capsys, argv):
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_pipeline_reports_its_version(delta_svd, capsys):
    out = _version_output(delta_svd.iniParser(), capsys, ["--version"])
    assert out.strip() == f"DELTA-SVD {delta_svd_version.__version__}"


def test_aggregator_reports_its_version(aggregate_results, capsys):
    out = _version_output(aggregate_results.iniParser(), capsys, ["--version"])
    assert out.strip() == f"DELTA-SVD {delta_svd_version.__version__}"


@pytest.fixture
def wide_terminal(monkeypatch):
    # argparse wraps the description to the terminal width, which would other-
    # wise be free to split "DELTA-SVD <version>" across two lines
    monkeypatch.setenv("COLUMNS", "100")


def test_pipeline_help_names_the_version(delta_svd, capsys, wide_terminal):
    out = _version_output(delta_svd.iniParser(), capsys, ["--help"])
    assert f"DELTA-SVD {delta_svd_version.__version__}" in out


def test_aggregator_help_names_the_version(aggregate_results, capsys, wide_terminal):
    out = _version_output(aggregate_results.iniParser(), capsys, ["-h"])
    assert f"DELTA-SVD {delta_svd_version.__version__}" in out


# ---------------------------------------------------------------------------
# Everything else that names the release has to agree with VERSION
#
# VERSION is the single source of truth; the files below restate it by hand,
# because their formats cannot interpolate it. These checks turn "someone bumped
# VERSION and forgot one of them" into a failing test on the bump commit, which
# is the moment anyone is actually in a position to fix it. Deliberately not
# covered: container/build.sh's usage-example comment, which is illustrative and
# affects neither a build nor what a user pulls.

def _repo_version():
    return VERSION_FILE.read_text().strip()


def test_citation_cff_version_matches():
    # CITATION.cff drives GitHub's "Cite this repository" output, so a stale
    # version there misattributes a release
    cff = (REPO_ROOT / "CITATION.cff").read_text()
    assert re.findall(r'(?m)^version:\s*(\S+)\s*$', cff) == [_repo_version()]


@pytest.mark.parametrize("doc", sorted((REPO_ROOT / "docs").rglob("*.md")),
                         ids=lambda p: p.name)
def test_documented_image_tags_match(doc):
    # a stale tag in the docs silently hands users the *previous* image - the
    # one drift here with real consequences, given results from different
    # versions must not be pooled. '<version>' placeholders are left alone.
    concrete = [t for t in re.findall(r'ghcr\.io/isdneuroimaging/delta-svd:([^\s`]+)',
                                      doc.read_text()) if not t.startswith('<')]
    assert all(t == _repo_version() for t in concrete), \
        f"{doc.name} pulls {sorted(set(concrete) - {_repo_version()})}, VERSION is {_repo_version()}"


def test_the_install_page_pins_a_concrete_version():
    # guards the check above: were the pull commands to stop carrying a literal
    # tag, it would pass by finding nothing at all
    tags = re.findall(r'ghcr\.io/isdneuroimaging/delta-svd:([^\s`]+)',
                      (REPO_ROOT / "docs" / "install.md").read_text())
    assert [t for t in tags if not t.startswith('<')]
