"""Command-line entry point for the ALS bootstrapper."""

from als.bootstrap import main as bootstrap_main


def main() -> int:
    return bootstrap_main()


if __name__ == "__main__":
    raise SystemExit(main())
