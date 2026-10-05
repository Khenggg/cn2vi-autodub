import pytest
from pydantic import ValidationError

from autodub.contracts import Roi, Segment
from autodub.domain import Budget, check_transition, fitting_action, merge_windows
from autodub.storage import safe_path


def test_pipeline_rejects_skipping_review():
    check_transition("QUEUED", "PREPARING")
    check_transition("PREPARING", "CHECKPOINTED")
    with pytest.raises(ValueError):
        check_transition("PREVIEW_READY", "TEXT_REMOVAL")
    with pytest.raises(ValueError):
        check_transition("COMPLETED", "QUEUED")


def test_windows_merge_and_clip_episode_boundaries():
    assert merge_windows([(50, 400), (450, 900), (5000, 5900)], 6000) == [(0, 1500), (4400, 6000)]
    assert merge_windows([], 6000) == []
    with pytest.raises(ValueError):
        merge_windows([(0, 7000)], 6000)


@pytest.mark.parametrize("duration,attempts,expected", [(1080, 0, "STRETCH"), (1081, 0, "SPEED"),
                                                      (1200, 0, "SPEED"), (1201, 0, "REWRITE"),
                                                      (1300, 3, "NEEDS_REVIEW")])
def test_duration_fit_boundaries(duration, attempts, expected):
    assert fitting_action(duration, 1000, attempts) == expected


def test_budget_includes_waiting_and_api_cost():
    assert Budget().project(30 * 60000, 15 * 60000, 600)["projected_vnd"] == 5100
    assert Budget().project(70 * 60000)["policy"] == "NO_AUTO_PROPAINTER"
    assert Budget().project(81 * 60000)["policy"] == "REQUIRE_CONTINUE"


def test_roi_and_segment_validation():
    Roi(x=0.15, y=0.78, w=0.7, h=0.16)
    with pytest.raises(ValidationError):
        Roi(x=0.9, y=0.7, w=0.5, h=0.1)
    with pytest.raises(ValidationError):
        Segment(id="s", start_ms=100, end_ms=99)
    with pytest.raises(ValidationError):
        Segment(id="s", start_ms=100, end_ms=200, words=[{"t": "a", "s": 90, "e": 120}])


def test_paths_reject_traversal_and_absolute_escape(tmp_path):
    assert safe_path(tmp_path, "work/safe.json").is_relative_to(tmp_path)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")
    with pytest.raises(ValueError):
        safe_path(tmp_path, str(tmp_path.parent / "outside"))
