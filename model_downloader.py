from argparse import ArgumentParser
from huggingface_hub import hf_hub_download, hf_hub_url
import logging
import os
from tqdm import tqdm
import requests
logging.basicConfig(level=logging.INFO)

BASE_MODEL_URL = "facebook/sapiens-pose-{p1}-torchscript/sapiens_{p1}_goliath_best_goliath_AP_{p2}_torchscript.pt2"

POSE_1B_MODEL_URL = BASE_MODEL_URL.format(p1="1b", p2="639")
POSE_06B_MODEL_URL = BASE_MODEL_URL.format(p1="0.6b", p2="609")
POSE_03B_MODEL_URL = BASE_MODEL_URL.format(p1="0.3b", p2="573")

def download_huggingface_model(model_url, filename):
    hf_hub_download(repo_id=model_url, filename=filename)

def download_pose_model(model_type, models_path):
    model_url = {
        "1b": POSE_1B_MODEL_URL,
        "0.6b": POSE_06B_MODEL_URL,
        "0.3b": POSE_03B_MODEL_URL
    }.get(model_type)

    if model_url:
        logging.info(f"Downloading pose model from {model_url}")
        filename = model_url.split("/")[-1]
        repo_id = "{}/{}".format(model_url.split("/")[0], model_url.split("/")[1])
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        download_path = os.path.join(models_path, filename)
        
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        with open(download_path, 'wb') as f:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))

                # tqdm has many interesting parameters. Feel free to experiment!
                tqdm_params = {
                    'total': total,
                    'miniters': 1,
                    'unit': 'B',
                    'unit_scale': True,
                    'unit_divisor': 1024,
                }
                with tqdm(**tqdm_params) as pb:
                    for chunk in r.iter_content(chunk_size=8192):
                        pb.update(len(chunk))
                        f.write(chunk)
        
        logging.info(f"Downloaded pose model to {filename}")
    else:
        logging.error(f"Unknown pose model type: {model_type}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--pose-model-type", default="1b", help="Specify the pose model type (1b, 0.6b, 0.3b)")
    parser.add_argument("--models-path", default="models", help="Specify the path to the models directory")

    args = parser.parse_args()
    if args.pose_model_type:
        logging.info(f"Pose model type specified: {args.pose_model_type}")
        download_pose_model(args.pose_model_type, args.models_path)
    else:
        logging.error("No pose model type specified.")


if __name__ == "__main__":
    main()
