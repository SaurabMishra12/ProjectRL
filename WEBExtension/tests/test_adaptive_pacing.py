"""Dispatcher invariants using local control-flow fixtures, never model calls."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from filelock import FileLock, Timeout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.adaptive_pacing import AdaptivePacer


class FakeClock:
    def __init__(self): self.now = 100.; self.sleeps = []
    def time(self): return self.now
    def sleep(self, duration): self.sleeps.append(duration); self.now += duration


class AdaptivePacingTests(unittest.TestCase):
    def test_network_runs_outside_scheduler_lock_and_spacing_is_from_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(); state = Path(tmp) / 'state.json'
            pacer = AdaptivePacer(state, clock=clock, initial_interval=2.)
            calls = []
            def create(**params):
                with FileLock(str(state) + '.lock', timeout=0): pass
                calls.append((clock.time(), params)); clock.now += 1.5
                return 'saved fixture response'
            for _ in range(2): pacer.call(create, {'unchanged': 'request'}, lambda e: (60., {}), lambda d: None)
            self.assertEqual(calls, [(100., {'unchanged': 'request'}), (102., {'unchanged': 'request'})])
            self.assertAlmostEqual(sum(clock.sleeps), .5)

    def test_429_cooldown_is_shared_and_timeout_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(); state = Path(tmp) / 'state.json'
            one = AdaptivePacer(state, clock=clock); two = AdaptivePacer(state, clock=clock)
            error = Exception('software control-flow fixture'); error.status_code = 429
            attempts = []; diagnostics = []
            def fail(**params): attempts.append(clock.time()); raise error
            with self.assertRaises(Exception): one.call(fail, {}, lambda e: (30., {}), diagnostics.append)
            two.call(lambda **params: attempts.append(clock.time()), {}, lambda e: (30., {}), diagnostics.append)
            self.assertEqual(attempts, [100., 130.])
            self.assertEqual(diagnostics[0]['http_status'], 429)
            error.status_code = None
            with self.assertRaises(Exception): one.call(fail, {}, lambda e: (30., {}), diagnostics.append)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len(diagnostics), 1)

    def test_no_more_than_two_network_calls_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / 'state.json'
            pacer = AdaptivePacer(state, initial_interval=.001, minimum_interval=.001)
            release = threading.Event(); entered = threading.Event(); active = 0; maximum = 0; total = 0
            lock = threading.Lock(); errors = []
            def create(**params):
                nonlocal active, maximum, total
                with lock:
                    active += 1; total += 1; maximum = max(maximum, active)
                    if active == 2: entered.set()
                release.wait(5)
                with lock: active -= 1
            def run():
                try: pacer.call(create, {}, lambda e: (60., {}), lambda d: None)
                except Exception as e: errors.append(e)
            workers = [threading.Thread(target=run) for _ in range(3)]
            for t in workers: t.start()
            try:
                self.assertTrue(entered.wait(3))
                with lock: self.assertEqual(total, 2)
            finally:
                release.set()
                for t in workers: t.join(5)
            self.assertFalse(errors); self.assertEqual(total, 3); self.assertEqual(maximum, 2)
