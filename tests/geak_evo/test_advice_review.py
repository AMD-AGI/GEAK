"""NativeAdviceReview: projection errors are recorded; store errors propagate."""

import unittest

from geak_evo.archive.store import StateError
from geak_evo.selection.advice import NativeAdviceReview


class FakeLoop:
    def __init__(self):
        self.ticks, self.events = iter([10.0, 12.5]), []

    def monotonic(self):
        return next(self.ticks)

    def record_advice_preparation(self, event, *, expected_revision):
        self.events.append((event, expected_revision))
        return "recorded"


class FakeDevelopment:
    def __init__(self, error=None):
        self.error = error

    def _project(self, snapshot):
        if self.error:
            raise self.error
        return ["a", "b"]

    def _record_projected(self, snapshot, projected):
        return dict(snapshot, revision=snapshot["revision"] + 1)


class AdviceReviewTest(unittest.TestCase):
    def test_projection_is_recorded_at_the_new_revision(self):
        loop = FakeLoop()
        self.assertEqual(NativeAdviceReview(loop, FakeDevelopment())({"revision": 3}), "recorded")
        self.assertEqual(loop.events, [({"input_revision": 3, "status": "available", "error_type": None,
                                         "reason": None, "projected_episode_count": 2,
                                         "elapsed_seconds": 2.5}, 4)])

    def test_numerical_error_is_recorded_as_unavailable(self):
        loop = FakeLoop()
        NativeAdviceReview(loop, FakeDevelopment(ZeroDivisionError("x")))({"revision": 3})
        event, revision = loop.events[0]
        self.assertEqual((event["status"], event["error_type"], event["reason"], revision),
                         ("unavailable", "ZeroDivisionError", "x", 3))

    def test_store_error_propagates(self):
        loop = FakeLoop()
        with self.assertRaises(StateError):
            NativeAdviceReview(loop, FakeDevelopment(StateError("journal")))({"revision": 3})
        self.assertEqual(loop.events, [])


if __name__ == "__main__":
    unittest.main()
