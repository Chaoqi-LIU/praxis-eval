from __future__ import annotations

import argparse

from policy_handler import PointPolicyHandler
from praxis_remote import PolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--gain", type=float, default=1.0)
    args = parser.parse_args()

    server = PolicyServer(
        PointPolicyHandler(gain=args.gain),
        host=args.host,
        port=args.port,
    )
    print(f"Serving policy on {args.host}:{args.port}")
    server.serve()


if __name__ == "__main__":
    main()
