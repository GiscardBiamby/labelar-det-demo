# Standard Library imports:
import argparse
from contextlib import redirect_stdout
import glob
import json
import os
from pathlib import Path
import shutil
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
        misspelled = self.get_mispelling_map()
        new_cats = self.get_new_cats(misspelled)
        old_to_new_cats = self.get_old_to_new_cat_mapping(misspelled, new_cats)
        if not opt.dry_run:
            self.generate_merged_dataset(new_cats, old_to_new_cats)

    def fix_and_remap_cats(self):
        self.print_current_cats()

    def print_current_cats(self):
        print("\nCategories from individual collects (before remapping):")
        for i, cid in enumerate(self.collect_ids):
            src_folder = self.collect_path / cid
            print(f"cid: {cid}, Src folder: {src_folder}")
            json_files = list(src_folder.glob("*.json"))
            with open(os.devnull, "w") as f, redirect_stdout(f):  # suppress output
                coco_misp = COCO(json_files[0])
            print("For cid:", cid, "\n", coco_misp.dataset["categories"], "i: ", i)

    def get_mispelling_map(self):
        label_remap_path: Path = opt.collect_path / "labelremap.json"
        if label_remap_path.exists():
            assert (
                label_remap_path.is_file()
            ), f"Label remap path is not a file: {label_remap_path}"
            with open(label_remap_path, "r") as JSON:
                misspelled = json.load(JSON)
        else:
            misspelled = {}
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
                corrected_name = self.remap_cat(cid, c, misspelled)
                #         print(cid, c, corrected_name)
                cats_merged.add(corrected_name)

        cats_merged_sorted = list(sorted(cats_merged))
        new_cats = {
            "background": {"supercategory": "", "id": 0, "name": "background"}
        }
        for i, c in enumerate(cats_merged_sorted):
            new_cats[c] = {"supercategory": "", "id": i+1, "name": c}
        print("new_cats (merged): ", new_cats)
        return new_cats

    def remap_cat(self, cid, cat, misspelled):
        corrected_name = (
            misspelled[cid][cat["name"]]
            if (cid in misspelled and cat["name"] in misspelled[cid])
            else cat["name"]
        )
        if "all" in misspelled and cat["name"] in misspelled["all"]:
            corrected_name = misspelled["all"][cat["name"]]
        return corrected_name

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
                corrected_name = self.remap_cat(cid, c, misspelled)
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

        return old_to_new_cats

    def generate_merged_dataset(self, new_cats, old_to_new_cats):
        ann_counter = 0
        old_to_new_imgs = {}
        master_imgs, master_anns = [], []

        newImgFolder = self.output_dir / f"images/{opt.ds_name}_{opt.split}"
        print("newImgFolder: ", newImgFolder)

        # New image directory
        if not newImgFolder.exists():
            os.makedirs(newImgFolder)

        # for each collect
        for i, cid in enumerate(self.collect_ids):
            src_folder = self.collect_path / cid
            json_files = list(src_folder.glob("*.json"))

            # for each json file in the folder
            for json_file in json_files:
                print("JSON_FILE: ", json_file)
                # load data as a COCO object
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    coco = COCO(json_file)

                # APPEND IMAGES TO MASTER LIST
                # -  -  -  -  -  -  -  -  -  -  -  -
                img = coco.loadImgs(ids=coco.getImgIds())[0]
                new_img = img.copy()
                new_img["id"] = len(master_imgs)
                new_img["collection_id"] = cid
                new_img["old_file_name"] = str(img["id"]).zfill(12) + ".png"
                new_img["file_name"] = str(new_img["id"]).zfill(12) + ".png"
                new_img["old_id"] = img["id"]
                old_to_new_imgs[img["id"]] = new_img["id"]
                master_imgs.append(new_img)

                I = Image.open(os.path.join(src_folder, img["file_name"]))
                npim = np.array(I)
                # If png has 4 channels, save only 3
                src = os.path.join(src_folder, img["file_name"])
                dst = os.path.join(newImgFolder, str(new_img["id"]).zfill(12) + ".png")
                if npim.shape[2] == 4:
                    img_3chan = Image.fromarray(npim[..., :3])
                    portrait = False
                    if portrait:
                        img_rot = img_3chan.rotate(270)  # if in portrait mode
                        img_rot.save(dst)
                    else:
                        img_3chan.save(dst)
                    print("Saved image: {}".format(dst))
                else:
                    shutil.copy(src, dst)

                # APPEND ANNOTATIONS TO MASTER LIST
                # -  -  -  -  -  -  -  -  -  -  -  -

                # get the category dictionary for this collect
                anns = coco.loadAnns(coco.getAnnIds())
                for ann in anns:
                    new_ann = ann.copy()
                    # quick formatting mods
                    new_ann["segmentation"] = [ann["segmentation"][0]["points"]]
                    new_ann["bbox"] = ann["box"]
                    # update to keep category-id consistent across multiple collects
                    new_ann["category_id"] = old_to_new_cats[(cid, ann["category_id"])][
                        "id"
                    ]
                    new_ann["id"] = ann_counter
                    new_ann["image_id"] = old_to_new_imgs[ann["image_id"]]
                    ann_counter += 1
                    master_anns.append(new_ann)

        new_instances = {
            "images": master_imgs,
            "annotations": master_anns,
            "categories": list(new_cats.values()),
        }

        # Check 'annotations' directory
        annotations_folder = opt.output_dir / "annotations"
        if not os.path.exists(annotations_folder):
            os.makedirs(annotations_folder)

        newAnnFile = annotations_folder / f"instances_{opt.ds_name}_{opt.split}.json"
        print("new Ann file: ", newAnnFile)

        with open(newAnnFile, "w") as outfile:
            json.dump(new_instances, outfile)

        print("New instance annotations save as {}".format(newAnnFile))


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
            "--ds_name", type=str, required=True, help="Name of the new dataset"
        )
        self.parser.add_argument(
            "--split", type=str, default="train", help="train | val"
        )
        self.parser.add_argument(
            "--collect_ids",
            type=str,
            required=True,
            help="List of collect id's to include in the dataset. You can also specify 'all' for this value and all the collect_id's in the collect_path will be processed.",
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
            help="""Path to directory where the final COCO formatted dataset will be saved. The final dataset will be created in a subdirectory of this path, where the subdir name is the name of the created dataset.""",
        )
        self.parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Output merged/renamed categories, but don't do anything else.",
        )
        self.parser.add_argument(
            "--delete_existing",
            type=bool,
            default=True,
            help="Delete output_path before creating new dataset."
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
        opt.output_dir: Path = opt.output_dir / f"{opt.ds_name}-{opt.split}"
        output_dir: Path = opt.output_dir
        if opt.delete_existing:
            if opt.output_dir.exists():
                print(f"Deleting output_dir: '{opt.output_dir}'")
                shutil.rmtree(opt.output_dir)
        print(f"Creating output_dir: '{opt.output_dir}'")
        output_dir.mkdir()

        # Collect_id's:
        print(type(opt.collect_ids))
        if isinstance(opt.collect_ids, str) and opt.collect_ids.lower() == "all":
            opt.collect_ids = list(opt.collect_path.glob("*"))
            opt.collect_ids = [
                cid_dir.name for cid_dir in opt.collect_ids if cid_dir.is_dir()
            ]
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
