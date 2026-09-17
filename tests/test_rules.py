"""The rules, which are the part of the system that is allowed to decide things.

The plan requires every row of the README's review-trigger table to have a test, so the table
and the code cannot drift apart. `test_every_documented_reason_is_reachable` enforces that
mechanically.
"""

from __future__ import annotations

import pytest

from relay.config import Settings
from relay.rules import (
    CLASSIFICATION,
    REVIEW_REASONS,
    classify,
    compose_confidence,
    review_reasons,
    template_summary,
)
from relay.schema import Category, Priority


@pytest.fixture()
def cfg() -> Settings:
    return Settings(review_conf_threshold=0.75, review_conf_other=0.85, review_conf_high=0.90)


def reasons(cfg, **kw):
    base = dict(
        confidence=0.95, category="routine", priority="low", site_id="site-118",
        video_ts="00:00:01.00", straddle=False, cfg=cfg,
    )
    base.update(kw)
    return review_reasons(**base)


# ---------------------------------------------------------------- classification

@pytest.mark.parametrize(
    ("observed", "category", "priority"),
    [
        ("post_manned", Category.routine, Priority.low),
        ("post_unattended", Category.other, Priority.high),
        ("person_loitering_near_entry", Category.loitering, Priority.medium),
        ("unidentified_person_at_post", Category.intrusion, Priority.high),
    ],
)
def test_the_classification_table(observed, category, priority):
    assert classify(observed) == (category, priority)


def test_an_unknown_state_degrades_rather_than_crashing():
    """Adding a state to the machine should produce a reviewable event, not kill a run."""
    assert classify("something_new_entirely") == (Category.other, Priority.medium)


def test_every_state_the_machine_emits_is_classified():
    from relay.detect.state import OBSERVED_FOR

    assert set(OBSERVED_FOR.values()) == set(CLASSIFICATION)


# ---------------------------------------------------------------- confidence

def test_confidence_is_the_weakest_signal():
    assert compose_confidence(0.93, 0.95, 1.00) == pytest.approx(0.93)
    assert compose_confidence(0.93, 0.55, 1.00) == pytest.approx(0.55)
    assert compose_confidence(0.93, 0.95, 0.08) == pytest.approx(0.08)


def test_confidence_is_not_a_product():
    """Three independent 0.9s multiply to 0.729 and would put a perfectly healthy detection
    under the 0.75 bar. That is the bug this choice avoids."""
    assert compose_confidence(0.9, 0.9, 0.9) == pytest.approx(0.9)
    assert 0.9 * 0.9 * 0.9 < 0.75


def test_darkness_alone_drags_an_event_under_the_bar(cfg):
    """The lights-off demo, as a test: nothing special-cases darkness."""
    dark = compose_confidence(0.93, 0.95, 0.08)
    assert "low_confidence" in reasons(cfg, confidence=dark)


# ---------------------------------------------------------------- thresholds

def test_a_healthy_routine_event_needs_no_review(cfg):
    assert reasons(cfg) == []


def test_low_confidence_trips_the_default_bar(cfg):
    assert reasons(cfg, confidence=0.70) == ["low_confidence"]


def test_the_bar_is_stricter_for_other(cfg):
    """'other' is the catch-all, so it is the category we trust least."""
    got = reasons(cfg, confidence=0.80, category="other", priority="medium")
    assert set(got) == {"low_confidence", "other_low_confidence"}


def test_the_bar_is_strictest_for_high_priority(cfg):
    """A false routine row costs nothing; a false high-priority page wakes a supervisor."""
    got = reasons(cfg, confidence=0.88, category="intrusion", priority="high")
    assert set(got) == {"low_confidence", "high_priority_low_confidence"}


def test_the_same_confidence_passes_as_routine_and_fails_as_high_priority(cfg):
    """The threshold differs because the cost of being wrong differs -- one number, two
    verdicts, which is the whole argument for a per-category bar."""
    assert reasons(cfg, confidence=0.88, category="routine", priority="low") == []
    assert reasons(cfg, confidence=0.88, category="intrusion", priority="high") != []


def test_post_unattended_gets_both_stricter_bars(cfg):
    """It is 'other' AND high priority, so both apply and the strictest wins."""
    got = reasons(cfg, confidence=0.86, category="other", priority="high")
    assert set(got) == {"low_confidence", "high_priority_low_confidence", "other_low_confidence"}


# ---------------------------------------------------------------- incomplete data

def test_a_missing_site_id_is_a_review_trigger_on_its_own(cfg):
    """A different kind of trigger entirely: the system is perfectly confident about a record
    nobody can route. No amount of model certainty fixes it."""
    assert reasons(cfg, confidence=0.99, site_id=None) == ["missing_site_id"]


def test_a_missing_timestamp_is_a_review_trigger(cfg):
    assert reasons(cfg, confidence=0.99, video_ts="") == ["missing_video_ts"]


def test_a_boundary_detection_is_a_review_trigger(cfg):
    assert reasons(cfg, confidence=0.99, straddle=True) == ["zone_straddle"]


def test_triggers_accumulate(cfg):
    got = reasons(cfg, confidence=0.40, category="other", priority="high",
                  site_id=None, straddle=True)
    assert set(got) == {
        "low_confidence", "high_priority_low_confidence", "other_low_confidence",
        "missing_site_id", "zone_straddle",
    }


def test_reasons_from_the_vision_stage_are_carried_through(cfg):
    got = reasons(cfg, extra=["stage_disagreement", "model_reports_dark"])
    assert set(got) == {"stage_disagreement", "model_reports_dark"}


def test_reasons_are_sorted_and_deduplicated(cfg):
    """Stable ordering means a live run and its replay produce comparable reason lists."""
    got = reasons(cfg, confidence=0.4, extra=["low_confidence", "low_confidence"])
    assert got == sorted(set(got))


# ---------------------------------------------------------------- the table itself

def test_an_undocumented_reason_is_rejected(cfg):
    """A reason with no documented meaning is a reason nobody can act on."""
    with pytest.raises(ValueError, match="undocumented"):
        reasons(cfg, extra=["some_new_trigger"])


def test_every_documented_reason_is_reachable(cfg):
    """Guards against the README table listing a trigger the code can never emit.

    The vision reasons are produced by VisionStage rather than here, so they are listed
    explicitly -- each has its own test in test_vision_repair.py.
    """
    from_vision = {
        "stage_disagreement", "vision_unavailable", "vision_bad_schema",
        "vision_timeout", "vision_rate_limited", "vision_failed", "model_reports_dark",
    }
    produced: set[str] = set()
    produced |= set(reasons(cfg, confidence=0.4))
    produced |= set(reasons(cfg, confidence=0.4, category="other", priority="high"))
    produced |= set(reasons(cfg, site_id=None))
    produced |= set(reasons(cfg, video_ts=""))
    produced |= set(reasons(cfg, straddle=True))
    assert produced | from_vision == set(REVIEW_REASONS)


# ---------------------------------------------------------------- template summaries

def test_the_template_summary_says_what_was_seen_and_that_nobody_wrote_it():
    """This is what a dispatcher reads when the model is down, so it has to be useful."""
    s = template_summary("post_unattended", "site-118", 0, 0)
    assert "site-118" in s
    assert "unattended" in s.lower()
    assert "auto-generated" in s.lower()


def test_the_template_summary_reports_the_counts():
    s = template_summary("unidentified_person_at_post", "site-118", 2, 1)
    assert "2 person(s) in the post zone" in s
    assert "1 in the approach zone" in s


def test_the_template_summary_survives_a_missing_site_id():
    assert "unspecified site" in template_summary("post_manned", None, 1, 0)


def test_every_observed_state_has_its_own_phrasing():
    """A template that just prints the state name back is not a summary."""
    for observed in CLASSIFICATION:
        assert observed not in template_summary(observed, "site-118", 1, 0)
