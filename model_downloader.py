from argparse import ArgumentParser
from huggingface_hub import hf_hub_download, hf_hub_url
import logging

logging.basicConfig(level=logging.INFO)

BASE_MODEL_URL = "facebook/sapiens-pose-{p1}-torchscript/sapiens_{p1}_goliath_best_goliath_AP_{p2}_torchscript.pt2"

POSE_1B_MODEL_URL = BASE_MODEL_URL.format(p1="1b", p2="639")
POSE_06B_MODEL_URL = BASE_MODEL_URL.format(p1="0.6b", p2="609")
POSE_03B_MODEL_URL = BASE_MODEL_URL.format(p1="0.3b", p2="573")

def download_huggingface_model(model_url, filename):
    hf_hub_download(repo_id=model_url, filename=filename)

def download_pose_model(model_type):
    model_url = {
        "1b": POSE_1B_MODEL_URL,
        "0.6b": POSE_06B_MODEL_URL,
        "0.3b": POSE_03B_MODEL_URL
    }.get(model_type)

    if model_url:
        logging.info(f"Downloading pose model from {model_url}")
        filename = model_url.split("/")[-1]
        repo_id = "{}/{}".format(model_url.split("/")[0], model_url.split("/")[1])
        download_huggingface_model(repo_id, filename)
        logging.info(f"Downloaded pose model to {filename}")
    else:
        logging.error(f"Unknown pose model type: {model_type}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--pose-model-type", default="1b", help="Specify the pose model type (1b, 0.6b, 0.3b)")

    args = parser.parse_args()
    if args.pose_model_type:
        logging.info(f"Pose model type specified: {args.pose_model_type}")
        download_pose_model(args.pose_model_type)
    else:
        logging.error("No pose model type specified.")


if __name__ == "__main__":
    main()
