import argparse

from core.utils.logging.visualizer import LogVisualizer


def main(args):
    visualizer = LogVisualizer(args.log_key, args.output)
    visualizer.visualize()


def get_args():
    parser = argparse.ArgumentParser(description="Visualize Logs")
    parser.add_argument("--log_key", required=True, help="The log key")
    parser.add_argument("--output", required=True, help="The output directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
