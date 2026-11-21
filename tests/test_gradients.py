from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import delta_svd_common
import delta_svd_gradients


# ---------------------------------------------------------------------------
# bval/bvec round trip

def test_bval_roundtrip(tmp_path):
    content = "0 0 1000 1000\n"
    fn = tmp_path / "test.bval"
    fn.write_text(content)

    arrFloat, arrStr = delta_svd_gradients.read_bval_or_bvec(str(fn))
    assert arrFloat.tolist() == [0.0, 0.0, 1000.0, 1000.0]

    out = tmp_path / "out.bval"
    delta_svd_gradients.write_bval_or_bvec(arrStr, str(out))
    assert out.read_text() == content


def test_bvec_roundtrip(tmp_path):
    content = "1 0 0.7071\n0 1 0.7071\n0 0 0\n"
    fn = tmp_path / "test.bvec"
    fn.write_text(content)

    arrFloat, arrStr = delta_svd_gradients.read_bval_or_bvec(str(fn))
    assert arrFloat.shape == (3, 3)

    out = tmp_path / "out.bvec"
    delta_svd_gradients.write_bval_or_bvec(arrStr, str(out))
    assert out.read_text() == content


def test_read_bval_or_bvec_rejects_malformed_shape(tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("1 2\n3 4\n")
    with pytest.raises(ValueError):
        delta_svd_gradients.read_bval_or_bvec(str(fn))


# A ragged or non-numeric file is reported with its name and line, not as numpy's
# own ValueError.

def test_read_bval_or_bvec_rejects_lines_of_unequal_length(tmp_path):
    fn = tmp_path / "bad.bvec"
    fn.write_text("0 1 0\n0 0 1\n0 0\n")          # third direction truncated
    with pytest.raises(delta_svd_common.DeltaSvdError) as excinfo:
        delta_svd_gradients.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert str(fn) in msg
    assert "line 1: 3 value(s)" in msg
    assert "line 3: 2 value(s)" in msg
    assert "blank line" not in msg               # none here, so no misleading hint


def test_read_bval_or_bvec_points_at_a_blank_line(tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("0 1000 1000\n\n")             # trailing blank line
    with pytest.raises(delta_svd_common.DeltaSvdError, match="blank line"):
        delta_svd_gradients.read_bval_or_bvec(str(fn))


def test_read_bval_or_bvec_locates_a_non_numeric_value(tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("0 1000 abc 1000\n")
    with pytest.raises(delta_svd_common.DeltaSvdError) as excinfo:
        delta_svd_gradients.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert "'abc'" in msg
    assert "line 1 at position 3" in msg
    assert str(fn) in msg


def test_read_bval_or_bvec_locates_a_non_numeric_value_in_a_bvec(tmp_path):
    # the reported position is the one in the file, not in the transposed array
    fn = tmp_path / "bad.bvec"
    fn.write_text("1 0 0\n0 1 0\n0 0 x\n")
    with pytest.raises(delta_svd_common.DeltaSvdError) as excinfo:
        delta_svd_gradients.read_bval_or_bvec(str(fn))

    assert "line 3 at position 3" in str(excinfo.value)


def test_read_bval_or_bvec_truncates_a_long_list_of_lines(tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("\n".join(["1 2 3"] * 12 + ["1 2"]) + "\n")
    with pytest.raises(delta_svd_common.DeltaSvdError) as excinfo:
        delta_svd_gradients.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert f"line {delta_svd_gradients.MAX_LINES_REPORTED}:" in msg
    assert f"line {delta_svd_gradients.MAX_LINES_REPORTED + 1}:" not in msg
    assert "and 3 more line(s)" in msg


@pytest.mark.parametrize("content", ["0 0 1000 1000\n", "1 0 0.7071\n0 1 0.7071\n0 0 0\n"])
def test_read_bval_or_bvec_still_accepts_well_formed_files(tmp_path, content):
    # the added checks must not narrow what parses: 'nan'/'inf'/exponents included
    fn = tmp_path / "ok.bval"
    fn.write_text(content)
    arrFloat, arrStr = delta_svd_gradients.read_bval_or_bvec(str(fn))
    assert arrFloat.size == arrStr.size


def test_read_bval_or_bvec_accepts_exponent_notation(tmp_path):
    fn = tmp_path / "ok.bval"
    fn.write_text("0 1e3 1.2E+03\n")
    arrFloat, _ = delta_svd_gradients.read_bval_or_bvec(str(fn))
    assert arrFloat.tolist() == [0.0, 1000.0, 1200.0]


# ---------------------------------------------------------------------------
# resolve_b_intervals(): the '--bRange' / '--shells' request as b-value windows

def test_resolve_b_intervals_widens_a_range_by_its_tolerance():
    assert delta_svd_gradients.resolve_b_intervals(bRange=[800, 1200]) == [(795.0, 1205.0)]


def test_resolve_b_intervals_orders_the_range_limits():
    # argparse takes the two limits positionally, so nothing stops a user from
    # passing them the wrong way round
    assert delta_svd_gradients.resolve_b_intervals(bRange=[1200, 800]) == [(795.0, 1205.0)]


def test_resolve_b_intervals_gives_each_shell_its_own_window():
    # one window per shell, so the b-values *between* the shells stay excluded -
    # the whole point of '--shells' over a range spanning them
    assert delta_svd_gradients.resolve_b_intervals(shells=[700, 1000]) == [(675.0, 725.0), (975.0, 1025.0)]


def test_resolve_b_intervals_sorts_and_deduplicates_shells():
    assert delta_svd_gradients.resolve_b_intervals(shells=[1000, 700, 1000]) == [(675.0, 725.0), (975.0, 1025.0)]


def test_resolve_b_intervals_prefers_shells_over_the_range_default():
    # '--bRange' keeps its default even when '--shells' is given, so the shells
    # have to win - argparse guarantees the user set only one of the two
    assert delta_svd_gradients.resolve_b_intervals(bRange=[800, 1200], shells=[1000]) == [(975.0, 1025.0)]


# ---------------------------------------------------------------------------
# describe_directions(): angular sampling, and whether the tensor is estimable.
# The count drives the floor in filter_b_values() and counts unique directions.

def _directions(n):
    """n distinct unit vectors, spread over a hemisphere by the golden-section
    spiral. A hemisphere because directions are antipodally equivalent here, so
    a full sphere would hand back near-duplicate pairs. Well-conditioned for a
    tensor fit at any n >= 6."""
    i = np.arange(n) + 0.5
    z = i / n
    r = np.sqrt(np.maximum(0.0, 1 - z**2))
    phi = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _plane_directions(n):
    """n distinct unit vectors, all in the xy-plane. However many there are,
    their outer products span only three of the six tensor components, so the
    design matrix stays rank-deficient - a damaged gradient table looks like
    this."""
    a = np.linspace(0, np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)


def test_describe_directions_counts_directions_and_reports_full_rank():
    g = _directions(30)
    bvals = np.r_[0.0, np.full(30, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g])

    assert delta_svd_gradients.describe_directions(bvals, bvecs) == (30, 7)


def test_describe_directions_counts_a_repeated_direction_once():
    # 24 volumes, 12 directions: the extra acquisitions average noise, they do
    # not constrain the free-water fraction any further
    g = _directions(12)
    bvals = np.r_[0.0, np.full(24, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g, g])

    assert delta_svd_gradients.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_counts_antipodal_directions_once():
    # g and -g probe the same tensor element, so a full-sphere scheme that
    # includes both must not be credited with twice the angular sampling
    g = _directions(12)
    bvals = np.r_[0.0, np.full(24, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g, -g])

    assert delta_svd_gradients.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_ignores_b0_volumes_and_zero_bvecs():
    g = _directions(12)
    # three b0s, and one diffusion-weighted volume whose bvec is all zeros: it
    # probes no direction at all
    bvals = np.r_[np.zeros(3), np.full(13, 1000.0)]
    bvecs = np.vstack([np.zeros((3, 3)), g, np.zeros(3)])

    assert delta_svd_gradients.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_reports_deficient_rank_for_coplanar_directions():
    # plenty of directions, but they span only part of the tensor
    bvals = np.r_[0.0, np.full(20, 1000.0)]
    bvecs = np.vstack([np.zeros(3), _plane_directions(20)])

    nDirections, rank = delta_svd_gradients.describe_directions(bvals, bvecs)
    assert nDirections == 20
    assert rank < delta_svd_gradients.DESIGN_MATRIX_RANK


def test_describe_directions_reaches_full_rank_at_six_directions():
    # six non-collinear directions plus a b0 are what the seven-column design
    # matrix needs; the floor in filter_b_values() sits well above this, but the
    # rank check must not fire here
    bvals = np.r_[0.0, np.full(6, 1000.0)]
    bvecs = np.vstack([np.zeros(3), _directions(6)])

    assert delta_svd_gradients.describe_directions(bvals, bvecs)[1] == 7


# ---------------------------------------------------------------------------
# format_b_values(): the b-values present, for error messages

def test_format_b_values_groups_within_shell_deviation():
    # scanners report b-values that scatter around the nominal shell; an error
    # message listing each of them verbatim would be unreadable
    assert delta_svd_gradients.format_b_values([0, 0, 998, 1000, 1002, 2000]) == '0 (n=2), 1000 (n=3), 2000 (n=1)'


# ---------------------------------------------------------------------------
# filter_b_values(): shell selection of the b-value range. Volumes, bvals and
# bvecs must stay index-aligned, or the fit silently uses the wrong directions.

DEFAULT_INTERVALS = [(795.0, 1205.0)]


def _write_dwi_set(tmp_path, bvals, bvecs):
    """Write a matched data/bval/bvec triplet from a b-value sequence and an
    (n, 3) array of directions. Volume i is filled with the constant i, so a
    test can tell from the voxel values which volumes survived."""
    bvals = list(bvals)
    bvecs = np.asarray(bvecs, dtype=float)
    (tmp_path / "in").mkdir(exist_ok=True)
    fnBval = tmp_path / "in" / "sub01.bval"
    fnBvec = tmp_path / "in" / "sub01.bvec"
    fnData = tmp_path / "in" / "sub01.nii.gz"
    fnBval.write_text(" ".join(str(b) for b in bvals) + "\n")
    fnBvec.write_text("\n".join(" ".join(f"{v:.6f}" for v in row) for row in bvecs.T) + "\n")
    img = np.arange(len(bvals), dtype=float) * np.ones((2, 2, 2, len(bvals)))
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(fnData))
    outDir = tmp_path / "out"
    outDir.mkdir(exist_ok=True)
    return str(fnData), str(fnBval), str(fnBvec), str(outDir)


def _shelled_set(tmp_path, shells, nPerShell=20, nB0=1):
    """A b0 plus 'nPerShell' whole-sphere directions on each of 'shells'."""
    g = _directions(nPerShell)
    bvals = [0] * nB0 + [b for b in shells for _ in range(nPerShell)]
    bvecs = np.vstack([np.zeros((nB0, 3))] + [g for _ in shells])
    return _write_dwi_set(tmp_path, bvals, bvecs)


def test_filter_b_values_drops_volumes_outside_brange(tmp_path):
    # b=2000 falls outside the default [800, 1200] shell and must go, together
    # with its bvec columns and its image volumes.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000, 2000], nPerShell=20)

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert [Path(f).parent.name for f in (fnDataO, fnBvalO, fnBvecO)] == ["out"] * 3
    # b-value strings are carried over verbatim, not reformatted
    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20
    # bvecs stay index-aligned with what survived: three rows, 21 columns
    assert [len(line.split()) for line in Path(fnBvecO).read_text().splitlines()] == [21] * 3
    # volumes 0..20 survive - and in that order
    kept = nib.load(fnDataO).get_fdata()
    assert kept.shape == (2, 2, 2, 21)
    assert kept[0, 0, 0].tolist() == list(range(21))


@pytest.mark.parametrize("length", [0.0, 0.98, 2.0, float("nan")])
def test_filter_b_values_refuses_non_unit_bvecs(tmp_path, length):
    # dipy's gradient_table() would raise later, with a traceback
    bvals = [0] + [1000] * 20
    bvecs = np.vstack([np.zeros((1, 3)), _directions(20)])
    bvecs[5] *= length
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(tmp_path, bvals, bvecs)
    with pytest.raises(delta_svd_common.DeltaSvdError, match="unit vector") as excinfo:
        delta_svd_gradients.filter_b_values(fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
                                  out_dir=outDir, bIntervals=DEFAULT_INTERVALS)
    assert "Volume(s): 6 " in str(excinfo.value)


def test_filter_b_values_accepts_bvecs_within_dipys_tolerance(tmp_path):
    bvals = [0] + [1000] * 20 + [2000]
    bvecs = np.vstack([np.zeros((1, 3)), _directions(20) * 1.009, [[0, 0, 5]]])
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(tmp_path, bvals, bvecs)
    # the b = 2000 volume is dropped, so its bvec does not matter
    delta_svd_gradients.filter_b_values(fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
                              out_dir=outDir, bIntervals=DEFAULT_INTERVALS)


# selected_volumes_float32() replaces get_fdata()[..., sel] in filter_b_values()
# to keep the whole image out of float64. It must produce the same float32
# values bit for bit, scaled input included: a last-bit change here reaches the
# metrics (see CONTRIBUTING.md).

def _stored_image(path, dtype, rng, shape=(3, 4, 5, 7)):
    data = rng.normal(800, 300, shape)
    img = nib.Nifti1Image(data, np.eye(4))
    img.set_data_dtype(dtype)               # int types: nibabel picks a slope/inter
    nib.save(img, str(path))
    return nib.load(str(path))


@pytest.mark.parametrize("dtype, ext", [
    ("float32", ".nii.gz"), ("float64", ".nii.gz"), ("int16", ".nii.gz"),
    ("uint8", ".nii.gz"), ("int16", ".nii")])
def test_selected_volumes_match_get_fdata_bit_for_bit(tmp_path, dtype, ext):
    rng = np.random.default_rng(0)
    nii = _stored_image(tmp_path / ("dwi" + ext), dtype, rng)
    if dtype in ("int16", "uint8"):
        assert (nii.dataobj.slope, nii.dataobj.inter) != (1, 0)    # scaling exercised
    idx = np.array([0, 2, 3, 6])

    expected = nii.get_fdata()[..., idx].astype(np.float32)
    got = delta_svd_gradients.selected_volumes_float32(nib.load(nii.get_filename()), idx)

    assert got.dtype == np.float32
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_filtered_file_is_unchanged_by_the_per_volume_read(tmp_path, dtype):
    # the file written before this change (float64 selection, cast on save)
    # and the one written now hold the same bytes of data
    rng = np.random.default_rng(1)
    nii = _stored_image(tmp_path / "dwi.nii.gz", dtype, rng)
    idx = np.array([1, 4, 5])
    old, new = str(tmp_path / "old.nii.gz"), str(tmp_path / "new.nii.gz")
    delta_svd_common.save_nifti(old, nii.get_fdata()[..., idx], nii.affine, nii.header, dtype="float32")
    delta_svd_common.save_nifti(new, delta_svd_gradients.selected_volumes_float32(nib.load(nii.get_filename()), idx),
                         nii.affine, nii.header, dtype="float32")

    a, b = nib.load(old), nib.load(new)
    assert a.header.get_data_dtype() == b.header.get_data_dtype() == np.float32
    assert np.array_equal(np.asanyarray(a.dataobj.get_unscaled()).view(np.uint32),
                          np.asanyarray(b.dataobj.get_unscaled()).view(np.uint32))


def test_selected_volumes_never_load_the_whole_image_as_float(tmp_path, monkeypatch):
    nii = _stored_image(tmp_path / "dwi.nii.gz", "int16", np.random.default_rng(2))
    monkeypatch.setattr(nib.Nifti1Image, "get_fdata",
                        lambda self, *a, **k: pytest.fail("whole image loaded as float"))
    delta_svd_gradients.selected_volumes_float32(nib.load(nii.get_filename()), np.array([0, 1]))


def test_filter_b_values_rounds_near_zero_bvalues_to_zero(tmp_path):
    # b=3 is a b0 in all but name; dipy's gradient_table would otherwise treat it
    # as a (useless) diffusion-weighted direction. Nothing is dropped here - the
    # volume is kept, only its b-value is rewritten.
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [3] + [1000] * 20,
        np.vstack([np.zeros(3), _directions(20)]),
    )

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20
    assert nib.load(fnDataO).get_fdata().shape[-1] == 21     # no volume dropped


def test_filter_b_values_is_a_no_op_when_nothing_needs_filtering(tmp_path):
    # The no-op path must hand back the *input* paths untouched: it writes no
    # files, so returning out_dir paths would point the pipeline at nothing.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert (fnDataO, fnBvalO, fnBvecO) == (fnData, fnBval, fnBvec)
    assert list(Path(outDir).iterdir()) == []


def test_filter_b_values_checks_the_unfiltered_data_too(tmp_path):
    # the no-op path returns early, so the guards have to sit ahead of it - a
    # dataset that needs no filtering can still be unfittable
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=8)

    with pytest.raises(ValueError, match="unique diffusion direction"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_keeps_b0s_regardless_of_brange(tmp_path):
    # b0 volumes are selected by the '<= 5' rule, not by the requested range - a
    # narrow range must never strip the b0s the fit needs for S0.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [700, 1000], nPerShell=20)

    _, fnBvalO, _, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20


def test_filter_b_values_returns_the_direction_count(tmp_path):
    # the count qualifies every metric the run produces, so it is handed back
    # for the QC report rather than only printed
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=32)

    assert delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )[3] == 32


# ---------------------------------------------------------------------------
# filter_b_values(): the b-value tolerance
#
# Scanners report b-values that deviate from the nominal shell, so a limit met
# exactly would discard volumes of the very shell that was asked for.

@pytest.mark.parametrize("bval,kept", [(1205, True), (1206, False),
                                       (795, True), (794, False)])
def test_filter_b_values_meets_the_range_limits_with_a_tolerance(tmp_path, bval, kept):
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20 + [bval],
        np.vstack([np.zeros(3), _directions(21)]),
    )

    _, fnBvalO, _, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert (str(bval) in Path(fnBvalO).read_text().split()) is kept


@pytest.mark.parametrize("bval,kept", [(1025, True), (1026, False),
                                       (975, True), (974, False)])
def test_filter_b_values_matches_a_shell_with_its_tolerance(tmp_path, bval, kept):
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20 + [bval],
        np.vstack([np.zeros(3), _directions(21)]),
    )

    _, fnBvalO, _, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=delta_svd_gradients.resolve_b_intervals(shells=[1000]),
    )

    assert (str(bval) in Path(fnBvalO).read_text().split()) is kept


def test_filter_b_values_excludes_shells_between_requested_ones(tmp_path):
    # what '--shells' is for: a range spanning 700 and 1500 would drag the 1000
    # shell into the fit with it
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [700, 1000, 1500], nPerShell=20)

    _, fnBvalO, _, _ = delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=delta_svd_gradients.resolve_b_intervals(shells=[700, 1500]),
    )

    kept = set(Path(fnBvalO).read_text().split())
    assert kept == {"0", "700", "1500"}


# ---------------------------------------------------------------------------
# filter_b_values(): the guards. Unchecked, each case would reach the fits, which
# return plausible-looking numbers instead of raising.

def test_filter_b_values_rejects_a_shell_that_matches_nothing(tmp_path):
    # the user named the shell explicitly, so its absence is a misconfiguration,
    # not something to quietly carry on without
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    with pytest.raises(ValueError, match=r"No volume has a b-value in \[1475, 1525\]"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=delta_svd_gradients.resolve_b_intervals(shells=[1000, 1500]),
        )


def test_filter_b_values_reports_the_b_values_present_when_nothing_matches(tmp_path):
    # the likeliest cause is a selection that does not fit the data, so the
    # error has to show what the data actually holds
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [2000], nPerShell=20)

    with pytest.raises(ValueError, match=r"0 \(n=1\), 2000 \(n=20\)"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_data_without_a_b0(tmp_path):
    # S0 is averaged from the b0 volumes; without one the fits have no scale
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path, [1000] * 20, _directions(20))

    with pytest.raises(ValueError, match="close to zero"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


@pytest.mark.parametrize("shells", [[1000, 2000], [1000]], ids=["filtered", "unfiltered"])
@pytest.mark.parametrize("shape", [(2, 2, 2, 40), (2, 2, 2, 42), (2, 2, 2)])
def test_filter_b_values_rejects_a_volume_count_mismatch(tmp_path, shells, shape):
    # volumes are selected by position, so on the filtered path a mismatch
    # used to pair b-values with the wrong volumes without any error
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, shells, nPerShell=20)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), fnData)

    with pytest.raises(delta_svd_common.DeltaSvdError, match="does not hold one volume per b-value"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_too_few_directions(tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=11)

    with pytest.raises(ValueError, match="Only 11 unique diffusion direction"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_accepts_exactly_the_minimum_directions(tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=12)

    assert delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )[3] == 12


def test_filter_b_values_does_not_count_repeats_towards_the_minimum(tmp_path):
    # 24 volumes, but only 6 directions: enough to solve for the tensor, and
    # nowhere near enough to pin the free-water fraction down
    g = _directions(6)
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 24,
        np.vstack([np.zeros(3), g, g, g, g]),
    )

    with pytest.raises(ValueError, match="Only 6 unique diffusion direction"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_counts_directions_after_the_selection(tmp_path):
    # 20 directions per shell, but a range that keeps only eight of them: the
    # floor applies to what reaches the fit, not to what was acquired
    g = _directions(20)
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 8 + [2000] * 12,
        np.vstack([np.zeros(3), g[:8], g[8:]]),
    )

    with pytest.raises(ValueError, match="Only 8 unique diffusion direction"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_a_degenerate_gradient_table(tmp_path):
    # enough directions to pass the count, but they lie in one plane and so
    # cannot span the tensor - what a damaged bvec file looks like
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20,
        np.vstack([np.zeros(3), _plane_directions(20)]),
    )

    with pytest.raises(ValueError, match="do not span the diffusion tensor"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_a_bvec_file_of_the_wrong_length(tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)
    Path(fnBvec).write_text("\n".join(" ".join("0" for _ in range(19)) for _ in range(3)) + "\n")

    with pytest.raises(ValueError, match="19 directions but the bval file holds 21"):
        delta_svd_gradients.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_warns_below_the_recommended_direction_count(tmp_path, capsys):
    # between the floor and the recommendation the fit runs, but the free-water
    # fraction is noisy - a warning, not an error
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=16)

    delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    out = capsys.readouterr().out
    assert "WARNING" in out and "recommended minimum of 20" in out


def test_filter_b_values_does_not_warn_at_the_recommended_direction_count(tmp_path, capsys):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    delta_svd_gradients.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert "WARNING" not in capsys.readouterr().out
