import argparse
import json
import os
import base64

from azureml.core import Run
from create_score_card import get_parser as scorecard_argparser, main as score_card_main

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # Constructor arguments
    parser.add_argument(
        "--rai_insights_dashboard_path", type=str, help="name:version", required=True
    )
    parser.add_argument(
        "--encoded_json", type=str, help="base64 encoded json string", required=True
    )

    return parser.parse_args()

def download_rai_insights_dashboard(insight_path, local_path):
    run = Run.get_context()
    ws = run.experiment.workspace

    if insight_path.startswith("ExperimentRun"):
        ds = ws.datastores["workspaceartifactstore"]
    else:
        ds = ws.datastores["workspaceblobstore"]

    ds.download(local_path, prefix=insight_path)


def write_base64_to_json(b64s, output_path):
    pdf_config = json.loads(base64.b64decode(b64s))
    with open(output_path, "w") as outfile:
        outfile.write(json.dumps(pdf_config))


def main(args):
    dashboard_folder = "./rai_i"
    output_folder = "./scorecard"
    os.makedirs(dashboard_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    dashboard_path = os.path.join(dashboard_folder, args.rai_insights_dashboard_path.lstrip("/"))
    json_path = "./score_card_config.json"

    download_rai_insights_dashboard(args.rai_insights_dashboard_path.lstrip("/"), dashboard_folder)
    write_base64_to_json(args.encoded_json, json_path)

    ap = scorecard_argparser()
    scorecard_args = ap.parse_args([
        '--rai_insights_dashboard', dashboard_path,
        '--pdf_output_path', output_folder,
        '--pdf_generation_config', json_path
        ])

    score_card_main(scorecard_args)

if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("Launching scorecard bootstrapper.")
    print("\n\n")

    # run main function
    main(parse_args())

    # add space in logs
    print("*" * 60)
    print("\n\n")
