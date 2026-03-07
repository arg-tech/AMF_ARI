"""
Export raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L to OpenVINO IR with INT8 weight quantisation.

Build and run:
    docker build -f scripts/Dockerfile -t amf-ari-export .
    docker run --rm -v "$(pwd)/exported_model:/app/exported_model" amf-ari-export

Optional push to HuggingFace Hub:
    docker run --rm \
      -v "$(pwd)/exported_model:/app/exported_model" \
      -e HUGGING_FACE_HUB_TOKEN=<your-token> \
      amf-ari-export python scripts/export_model.py --push-to-hub arg-tech/amf-ari-roberta-ov-int8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_path(config_path: Path) -> str:
    with config_path.open() as f:
        return json.load(f)["model_path"]


def export(model_path: str, output_dir: Path) -> None:
    from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Exporting model to OpenVINO IR with INT8 weight quantisation...")
    quantization_config = OVWeightQuantizationConfig(bits=8, ratio=1.0)
    ov_model = OVModelForSequenceClassification.from_pretrained(
        model_path,
        export=True,
        compile=False,
        quantization_config=quantization_config,
    )

    logger.info("Saving model to %s", output_dir)
    ov_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Export complete.")


def push(output_dir: Path, repo_id: str) -> None:
    from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer

    logger.info("Pushing model to Hub repo: %s", repo_id)
    ov_model = OVModelForSequenceClassification.from_pretrained(output_dir, export=False, compile=False)
    ov_model.push_to_hub(str(output_dir), repository_id=repo_id, use_auth_token=True)

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.push_to_hub(repo_id)
    logger.info("Push complete. Set 'ov_model_path' in config/config.json to: %s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RoBERTa ARI model to OpenVINO IR")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "config.json",
        help="Path to config.json (default: config/config.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./exported_model"),
        help="Local output directory (default: ./exported_model)",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="HuggingFace Hub repo ID to push the exported model to",
    )
    args = parser.parse_args()

    model_path = load_model_path(args.config)
    logger.info("Source model: %s", model_path)

    export(model_path, args.output)

    if args.push_to_hub:
        push(args.output, args.push_to_hub)
    else:
        logger.info(
            "Skipping Hub upload. Run with --push-to-hub <repo-id> to upload, "
            "then set 'ov_model_path' in config/config.json."
        )


if __name__ == "__main__":
    sys.exit(main())
