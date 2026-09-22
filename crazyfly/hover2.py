#!/usr/bin/env python3
"""Flow deck flight without waiting for stable position or position telemetry."""

import sys

import hover


def run(args):
    hover.cflib.crtp.init_drivers()
    with hover.SyncCrazyflie(args.uri, cf=hover.Crazyflie(rw_cache="./.cache")) as scf:
        # Start on the floor. No measured launch position is used in this variant.
        hover.prepare_flight(scf, initial_z=0.0)
        print("Skipping position stability checks; launch Z is initialized to zero.")
        if args.check_only:
            print("Flow deck/configuration check complete; no arming requested.")
            return
        hover.fly(scf.cf, 0.0, args.height, args.duration, args.distance, args.speed)
        print("Sequence complete; stop and disarm commands sent.")


def main(argv=None):
    args = hover.parse_args(argv, description=__doc__)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; flight cleanup attempted if needed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Hover failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
