#!/usr/bin/env python3
"""Runs spaCy NLP pipeline to annotate privacy policies and save the results.

Subsequent scripts will access NLP annotations again and again. For efficiency,
we just run the pipeline once with this script.

Check `PolicyDocument.initialize` for details.
"""

import argparse
import logging
import os

import spacy
import torch

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline


def main(workdirs, nlp_model_dir="", debug=False, gpu_memory_threshold=0.9):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )

    use_gpu = spacy.prefer_gpu()
    nlp = setup_nlp_pipeline(nlp_model_dir)

    for d in workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.initialize(d, nlp=nlp)
        document.save()

        if debug:
            with open(os.path.join(d, "document.txt"), "w", encoding="utf-8") as fout:
                fout.write(document.print_tree())

        if use_gpu:
            # Updated to avoid deprecated torch.has_mps / getattr(..., 'has_mps') pattern
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            if device.type == "cuda":
                current_device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(current_device)
                gmem_total = props.total_memory
                gmem_reserved = torch.cuda.memory_reserved(current_device)
                if gmem_total > 0 and gmem_reserved / gmem_total > gpu_memory_threshold:
                    logging.info(
                        "Empty GPU CUDA cache (reserved %.2f%% > %.0f%% threshold)",
                        100 * gmem_reserved / gmem_total,
                        100 * gpu_memory_threshold,
                    )
                    torch.cuda.empty_cache()
            elif device.type == "mps":
                # MPS backend has limited memory introspection; optionally clear cache if available
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    logging.info("Empty GPU MPS cache (proactive)")
                    torch.mps.empty_cache()
            # CPU: nothing to clear


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "usage: init_document.py <workdir1> [<workdir2> ...] [--nlp MODEL_DIR] [--debug] [--gpu-memory-threshold FLOAT]"
        )
        sys.exit(1)
    # parse sys.argv manually or via argparse then call:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+")
    parser.add_argument("--nlp", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu-memory-threshold", default=0.9, type=float)
    args = parser.parse_args()
    main(args.workdirs, args.nlp, args.debug, args.gpu_memory_threshold)
