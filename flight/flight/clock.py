"""Injectable time: mission code calls ``clock.sleep``/``clock.time``, never ``time.sleep``.

``RealClock`` sleeps; the simulator's clock advances the physics instead, so simulated
missions run faster than real time with no code change.
"""
import time


class RealClock:
    def time(self):
        return time.monotonic()

    def sleep(self, dt):
        if dt > 0:
            time.sleep(dt)


class ManualClock:
    """Advances only when told to; for tests with a mocked Crazyflie."""

    def __init__(self, t0=0.):
        self.t = t0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += max(0., dt)
