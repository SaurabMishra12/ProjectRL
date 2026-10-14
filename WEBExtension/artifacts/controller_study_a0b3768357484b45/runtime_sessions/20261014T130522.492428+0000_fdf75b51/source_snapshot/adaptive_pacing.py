"""Cross-process start pacing and two network slots; no changes to requests."""
import json
import time
from pathlib import Path
from filelock import FileLock, Timeout
from protocol import write_json


class AdaptivePacer:
    def __init__(self, state_path, clock=time, initial_interval=2., minimum_interval=1.):
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.initial = initial_interval
        self.minimum = minimum_interval

    def read(self):
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        if state.get('pacing_version') != 2:
            # Preserve any previously declared provider cooldown at migration.
            state = {'pacing_version': 2, 'interval_seconds': self.initial,
                     'next_allowed': state.get('next_allowed', 0.), 'success_streak': 0}
        return state

    def slot(self):
        while True:
            for index in range(2):
                slot = FileLock(str(self.path) + f'.slot{index}.lock', timeout=0)
                try:
                    slot.acquire()
                    return slot
                except Timeout:
                    pass
            self.clock.sleep(.1)

    def reserve(self):
        while True:
            with FileLock(str(self.path) + '.lock', timeout=7200):
                state = self.read()
                wait = state['next_allowed'] - self.clock.time()
                if wait <= 0:
                    state['next_allowed'] = self.clock.time() + state['interval_seconds']
                    write_json(self.path, state)
                    return
            # Sleep outside the scheduler lock, never while holding the network-wide lock.
            self.clock.sleep(min(30., wait))

    def call(self, create, params, guidance, diagnostic):
        slot = self.slot()
        try:
            self.reserve()
            try:
                result = create(**params)
            except Exception as error:
                if getattr(error, 'status_code', None) == 429:
                    delay, headers = guidance(error)
                    with FileLock(str(self.path) + '.lock', timeout=7200):
                        state = self.read()
                        state['interval_seconds'] = min(60., max(self.initial, state['interval_seconds'] * 2))
                        state['next_allowed'] = max(state['next_allowed'], self.clock.time() + delay)
                        state['success_streak'] = 0
                        write_json(self.path, state)
                    diagnostic({'http_status': 429, 'retry_after_seconds': delay,
                                'selected_headers': headers,
                                'next_interval_seconds': state['interval_seconds']})
                raise
            with FileLock(str(self.path) + '.lock', timeout=7200):
                state = self.read()
                state['success_streak'] += 1
                if state['success_streak'] >= 25:
                    state['interval_seconds'] = max(self.minimum, state['interval_seconds'] * .9)
                    state['success_streak'] = 0
                write_json(self.path, state)
            return result
        finally:
            slot.release()
