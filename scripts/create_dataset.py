# Standard Library imports:
import argparse
from contextlib import redirect_stdout
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# 3rd Party imports:
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

# 1st Party imports:
from _import_helper import ROOT_DIR


class LabelARToCOCO(object):
    def __init__(self, opt: argparse.Namespace) -> None:
        self.opt: argparse.Namespace = opt
        self.collect_ids: List[str] = opt.collect_ids
        self.ds_name: str = opt.ds_name
        self.split: str = opt.split
        self.output_dir: Path = opt.output_dir
        self.collect_path: Path = opt.collect_path

    def convert(self):
        self.fix_and_remap_cats()
        misspelled = self.get_mispelling_map()
        new_cats = self.get_new_cats(misspelled)
        self.get_old_to_new_cat_mapping(misspelled, new_cats)

    def fix_and_remap_cats(self):
        self.print_current_cats()

    def print_current_cats(self):
        print("\nCategories from individual collects (before remapping):")
        for i, cid in enumerate(self.collect_ids):
            src_folder = self.collect_path / cid
            json_files = list(src_folder.glob("*.json"))
            with open(os.devnull, "w") as f, redirect_stdout(f):  # suppress output
                coco_misp = COCO(json_files[0])
            print("For cid:", cid, "\n", coco_misp.dataset["categories"], "i: ", i)

    def get_mispelling_map(self):
        # hardcode while i get the script up and running, but later pull this from a
        # json file whose name matches opt.ds_name, located in dir: opt.collects_path
        misspelled = {
            "FPLM": {"mug-blue-s": "mug-blu-s"},
            "SWKW": {},
            "3RPC": {
                "0": "mug-wht-s",
                "1": "mug-blu-t",
                "2": "mug-wht-t",
                "3": "mug-blu-s",
                "4": "mug-red",
            },
            "VHR7": {},
            "1DKT": {"mug-white-t": "mug-wht-t", "mug-white-s": "mug-wht-s"},
            "LYVP": {"mug-white-t": "mug-wht-t", "mug-white-s": "mug-wht-s"},
            "KS9A": {},
        }
        print("\nUsing misspelling remaps: ", misspelled)
        return misspelled

    def get_new_cats(self, misspelled):
        """
        Scans all individual labelar json files and returns new_cats, dict where all
        the categories from the collects have been renamed/remapped and merged into a
        de-duplicated set of category names
        """
        cats_merged = set()
        for i, cid in enumerate(self.collect_ids):
            src_folder = self.collect_path / cid
            json_files = list(src_folder.glob("*.json"))
            # load data as a COCO object
            with open(os.devnull, "w") as f, redirect_stdout(f):  # suppressed output
                coco = COCO(json_files[0])
            cats = coco.dataset["categories"]
            for c in cats:
                corrected_name = corrected_name = (
                    misspelled[cid][c["name"]]
                    if (cid in misspelled and c["name"] in misspelled[cid])
                    else c["name"]
                )
                #         print(cid, c, corrected_name)
                cats_merged.add(corrected_name)

        cats_merged_sorted = list(sorted(cats_merged))
        new_cats = {}
        for i, c in enumerate(cats_merged_sorted):
            new_cats[c] = {"supercategory": "", "id": i, "name": c}
        print("new_cats (merged): ", new_cats)
        return new_cats

    def get_old_to_new_cat_mapping(self, misspelled, new_cats):
        """
        Returns a dict where keys are a tuple: (collect_id, cat_id), and values are:
        dict containing keys: [cid, old_file, old_cat, old_id, old_name, id, name].
        The dict helps map back and forth between "old" cats (cats from individual
        labelAR collect json's, and the merged categories that will be in the final
        merged COCO json that we'll generate). Bro, it's gonna work.
        """
        old_to_new_cats = {}
        for i, cid in enumerate(self.collect_ids):
            src_folder = self.collect_path / cid
            json_files = list(src_folder.glob("*.json"))
            # load data as a COCO object
            with open(os.devnull, "w") as f, redirect_stdout(f):  # suppressed output
                coco = COCO(json_files[0])
            cats = coco.dataset["categories"]
            for c in cats:
                corrected_name = corrected_name = (
                    misspelled[cid][c["name"]]
                    if (cid in misspelled and c["name"] in misspelled[cid])
                    else c["name"]
                )
                # If new cat name already registered, use the id already assigned to that cat name:
                assert (
                    corrected_name in new_cats
                ), "Remapped category name not found in new_cats registry!"
                new_cat_id = new_cats[corrected_name]["id"]
                # Update the old_to_new mapping:
                old_to_new_cats[(cid, c["id"])] = {
                    "cid": cid,
                    "old_file": json_files[0],
                    "old_cat": c,
                    "old_id": c["id"],
                    "old_name": c["name"],
                    "id": new_cat_id,
                    "name": corrected_name,
                }

        print("\nNEW CATS (merged & remapped):")
        for k, v in new_cats.items():
            print(f"{k}: {v}")

        print("\nOLD TO NEW cat mappings:")
        for k, v in old_to_new_cats.items():
            print(k, v)


def main(opt):
    converter = LabelARToCOCO(opt)
    converter.convert()
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
            help="List of collect id's to include in the dataset. You can also specify 'all' for this value and all the collect_id's in the collect_path will be processed.",
        )
        self.parser.add_argument(
            "--ds_name", type=str, required=True, help="Name of the new dataset"
        )
        self.parser.add_argument(
            "--split", type=str, default="train", help="train | val"
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
        assert (
            opt.collect_path.exists()
        ), f"Collect_path '{opt.collect_path}' does not exist"

        # Name

        # Output path:
        assert opt.output_dir.exists(), f"output_dir '{opt.output_dir}' does not exist"
        opt.output_dir = opt.output_dir / f"{opt.ds_name}-{opt.split}"

        # Collect_id's:
        print(type(opt.collect_ids))
        if isinstance(opt.collect_ids, str) and opt.collect_ids.lower() == "all":
            opt.collect_ids = list(opt.collect_path.glob("*"))
            opt.collect_ids = [cid_dir.name for cid_dir in opt.collect_ids]
            print(opt.collect_ids)
        else:
            opt.collect_ids = str(opt.collect_ids).upper().split(",")
        for cid in opt.collect_ids:
            collect_id_path = opt.collect_path / cid
            assert (
                collect_id_path.exists()
            ), f"No folder found for collect_id: '{cid}' in base path: '{opt.collect_path}'"

        self.print_options(opt)

        return opt

    def print_options(self, opt):
        """ 'Nuff said"""
        print("opt: {")
        for k, v in opt.__dict__.items():
            print("  ", k, ": ", v)
        print("}")


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
