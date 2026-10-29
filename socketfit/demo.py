import argparse
from pathlib import Path

import numpy as np

from .align import align
from .fit import adapt_socket, limb_clearance, report
from .mesh import write_stl
from .synthetic import make_limb, make_socket, rotation_matrix, transform
from .viz import before_after


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic demo: align a scanned limb to a socket, measure fit, adapt the socket.")
    ap.add_argument("--target", type=float, default=2.0, help="target clearance in mm")
    ap.add_argument("--out", default="out")
    ap.add_argument("--figure", default="docs/before_after.png")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    socket = make_socket()
    truth_R, truth_t = rotation_matrix(25, -15, 140), np.array([60.0, -35.0, 80.0])
    scan = transform(make_limb(), truth_R, truth_t)

    limb, R, t, icp_err = align(scan, socket.vertices)
    tilt = np.degrees(np.arccos(np.clip((np.trace(R @ truth_R) - 1) / 2, -1, 1)))
    print(f"alignment: mean closest-point distance {icp_err:.2f} mm (the socket wall sits {args.target:.0f} mm off the limb by design)")
    print(f"           rotation recovered to within {tilt:.2f} degrees")

    before = limb_clearance(limb, socket)
    adapted = adapt_socket(socket, limb, target=args.target)
    after = limb_clearance(limb, adapted)
    print("before:", report(before, args.target))
    print("after: ", report(after, args.target))

    write_stl(limb, str(out / "limb_aligned.stl"), "limb")
    write_stl(socket, str(out / "socket_original.stl"), "socket")
    write_stl(adapted, str(out / "socket_adapted.stl"), "socket_adapted")
    before_after(limb, before, after, args.figure)
    print(f"wrote STL files to {out}/ and {args.figure}")


if __name__ == "__main__":
    main()
