import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="Run training or inference for a chosen env")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run training")
    group.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument(
        "--env",
        required=True,
        choices=["cartpole", "lunar_lander", "cartpole_td0", "cartpole_mc", "cartpole_tdn"],
        help="Environment name",
    )

    args = parser.parse_args()

    env_base = args.env

    try:
        if args.train:
            mod = importlib.import_module(f"{env_base}.train")
            if hasattr(mod, "main"):
                mod.main()
            elif hasattr(mod, "train"):
                mod.train()
            else:
                print(f"No training entrypoint found in {env_base}.train")
        else:
            mod = importlib.import_module(f"{env_base}.inference")
            if hasattr(mod, "main"):
                mod.main()
            elif hasattr(mod, "inference"):
                mod.inference()
            else:
                print(f"No inference entrypoint found in {env_base}.inference")
    except ModuleNotFoundError as e:
        print(f"Could not import module for env '{args.env}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
