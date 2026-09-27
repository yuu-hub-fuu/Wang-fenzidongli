import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic  # noqa: E402

NAME = "1abc_A"


@pytest.fixture(scope="session")
def atlas(tmp_path_factory):
    """A one-target ATLAS directory plus a matching split CSV."""
    root = tmp_path_factory.mktemp("atlas")
    atlas_dir = str(root / "atlas")
    ref = synthetic.make_atlas_target(atlas_dir, NAME)
    split = synthetic.split_csv(str(root / "split.csv"), [(NAME, synthetic.DEFAULT_SEQ)])
    return {"atlas_dir": atlas_dir, "split": split, "name": NAME, "ref": ref}
