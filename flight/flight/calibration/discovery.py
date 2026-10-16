"""Crazyradio link discovery and the same live Lighthouse bitmasks used by cfclient."""
from queue import Empty
from collections import deque
import time


def choose_uri(found):
    radios = sorted({entry[0] for entry in found if entry[0].startswith('radio://')})
    if not radios:
        raise RuntimeError('No Crazyflie found through Crazyradio. Power it on and disconnect cfclient, then retry.')
    if len(radios) != 1:
        raise RuntimeError('Multiple Crazyflies found; select one with --uri: ' + ', '.join(radios))
    return radios[0]


def scan_uri(address=None):
    import cflib.crtp
    cflib.crtp.init_drivers()
    return choose_uri(cflib.crtp.scan_interfaces(address=address))


def stations_from_status(data):
    # bsReceive reports current sweeps; bsAvailable alone can include stale entries.
    receiving = int(data['lighthouse.bsReceive'])
    confirmed = int(data['lighthouse.bsCalCon'])
    mask = receiving & confirmed
    return [i for i in range(16) if mask & (1 << i)]


def recent_stations(observations, now, window=.75):
    """Require three fresh receptions per station; tolerate brief bitmask gaps."""
    counts = {}
    for at, ids in observations:
        if 0 <= now-at <= window:
            for station in ids:
                counts[station] = counts.get(station, 0)+1
    return sorted(station for station, count in counts.items() if count >= 3)


def discover_stations(cf, timeout=30., history=None):
    from cflib.crazyflie.log import LogConfig
    from ..clock import RealClock
    from ..preflight import samples
    clock = RealClock()
    names = ('lighthouse.bsReceive', 'lighthouse.bsAvailable', 'lighthouse.bsCalCon')
    deadline, latest, previous, stable_since = clock.time()+timeout, {}, None, clock.time()
    observations = deque()
    next_status = 0.
    with samples(cf, LogConfig, clock, names, period_ms=100, log_name='Setup stations') as queue:
        while clock.time() < deadline:
            if not cf.is_connected():
                raise ConnectionError('Crazyradio link lost during station discovery')
            try:
                received, data = queue.get(timeout=.2)
            except Empty:
                continue
            if isinstance(data, Exception):
                raise data
            if clock.time()-received > .5:
                continue
            latest = data
            now = clock.time()
            instantaneous = stations_from_status(data)
            observations.append((received, instantaneous))
            while observations and now-observations[0][0] > .75:
                observations.popleft()
            ids = recent_stations(observations, now)
            if history is not None:
                history.append(dict(time=received, values=dict(data), instantaneous_ids=instantaneous, recent_ids=ids))
            if now >= next_status:
                print(f'Lighthouse channels receiving now: {[i+1 for i in instantaneous]}; '
                      f'repeated reception in last 0.75 s: {[i+1 for i in ids]}.')
                next_status = now+2
            if ids != previous:
                previous, stable_since = ids, now
            if len(ids) == 2 and now-stable_since >= 1.:
                return ids, dict(latest)
    raise RuntimeError(f'Need repeated recent sweeps from two Lighthouse stations with received factory calibration; '
                       f'latest status: {latest}. Available/stored station bits alone do not establish current reception.')


def inspect(uri=None):
    from ..hardware import connect
    from ..clock import RealClock
    from ..preflight import wait_for_parameters
    from .lighthouse import wait_callback
    from cflib.crazyflie.mem import LighthouseMemHelper
    uri = uri or scan_uri()
    with connect(uri) as scf:
        wait_for_parameters(scf, RealClock())
        if int(float(scf.cf.param.get_value('deck.bcLighthouse4'))) != 1:
            raise RuntimeError('The connected Crazyflie has no Lighthouse deck')
        ids, status = discover_stations(scf.cf)
        calibs = wait_callback(LighthouseMemHelper(scf.cf).read_all_calibs)
        return dict(version=1, uri=uri, base_stations=ids, channels=[i+1 for i in ids], status=status,
                    factory_uids={str(i): calibs[i].uid for i in ids if i in calibs and calibs[i].valid})
