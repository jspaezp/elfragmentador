"""
Solution to execute a notebook with command line arguments

https://stackoverflow.com/questions/37534440/passing-command-line-arguments-to-argv-in-jupyter-ipython-notebook

call as:
    $ python evaluate.py --checkpoint ${PWD}/foo.ckpt --data_csv ${PWD}/my_data.csv

Note that it only takes absolute paths for the inputs... sorry
"""

import sys, os
from pathlib import Path

IPYNB_FILENAME = str(Path(__file__).parent) + "/templates/template_evaluation.ipynb"
CONFIG_FILENAME = str(Path(__file__).parent) + "/templates/.config_ipynb"


def main(argv):
    with open(CONFIG_FILENAME, "w") as f:
        f.write(" ".join(argv))
    os.system(
        "jupyter nbconvert --output-dir . --output evaluation_output --execute {:s} --to html".format(
            IPYNB_FILENAME
        )
    )
    return None


if __name__ == "__main__":
    print(Path(__file__))
    print(sys.argv)
    main(sys.argv)
