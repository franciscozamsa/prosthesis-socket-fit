import argparse

from .align import align
from .fit import adapt_socket, limb_clearance, report
from .mesh import read_stl, write_stl


def main(argv=None):
    ap = argparse.ArgumentParser(description="Align a scanned limb to a socket inner surface, report the fit and write an adapted socket.")
    ap.add_argument("limb", help="ASCII STL of the scanned limb surface (mm)")
    ap.add_argument("socket", help="ASCII STL of the socket inner surface (mm)")
    ap.add_argument("--target", type=float, default=2.0, help="target clearance in mm")
    ap.add_argument("--out", default="socket_adapted.stl")
    args = ap.parse_args(argv)

    socket = read_stl(args.socket)
    limb, _, _, err = align(read_stl(args.limb), socket.vertices)
    print(f"alignment residual {err:.2f} mm")
    print("before:", report(limb_clearance(limb, socket), args.target))
    adapted = adapt_socket(socket, limb, target=args.target)
    print("after: ", report(limb_clearance(limb, adapted), args.target))
    write_stl(adapted, args.out, "socket_adapted")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
