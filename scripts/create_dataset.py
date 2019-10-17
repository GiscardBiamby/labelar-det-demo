# Standard Library imports:
import argparse
import glob
import json
from pathlib import Path

# 3rd Party imports:
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

# 1st Party imports:
from _import_helper import ROOT_DIR


def main(opt):
    pass


class opts(object):
    """
    Handle parsing of command line args.
    """

    def __init__(self):
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--collect_ids",
            type=str,
            required=True,
            help="List of collect id's to include in the dataset",
        )
        self.parser.add_argument(
            "--ds_name", type=str, required=True, help="Name of the new dataset"
        )
        self.parser.add_argument(
            "--collect_path",
            type=Path,
            default=ROOT_DIR / "data",
            help="Path to a directory that contains the collects as subfolders",
        )
        self.parser.add_argument(
            "--output_dir",
            type=Path,
            default=ROOT_DIR / "training/data",
            help="Path to directory where the final COCO formatted dataset will be saved. The final dataset will be created in a subdirectory of this path, where the subdir name is the name of the created dataset.",
        )

    def parse(self, args=""):
        """
        Sets up the command line params, checks for validity, converts options to types
        that are easier to use later on in the code, fills in default values, etc.
        """
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # Collects dir:
        print(f"root_dir: {ROOT_DIR}")
        assert (
            opt.collect_path.exists()
        ), f"Collect_path '{opt.collect_path}' does not exist"

        # Output path:
        assert opt.output_dir.exists(), f"output_dir '{opt.output_dir}' does not exist"

        # Collect_id's:
        opt.collect_ids = str(opt.collect_ids).split(",")

        print(opt)
        return opt


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
