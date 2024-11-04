from __future__ import print_function
import argparse
import yaml

from trainers.test_spillage import selfsupervised
from trainers.predict_scoop import predict_scoop
from trainers.predict_spillage import predict_spillage


if __name__ == "__main__":

    # Load the config file
    parser = argparse.ArgumentParser(description="Sensor fusion model")
    parser.add_argument("--config", help="YAML config file", default = "configs/training_default.yaml")
    parser.add_argument("--notes", default="", help="run notes")
    parser.add_argument("--dev", type=bool, default=False, help="run in dev mode")
    parser.add_argument(
        "--continuation",
        type=bool,
        default=False,
        help="continue a previous run. Will continue the log file",
    )
    parser.add_argument("--type", default="spillage")
    args = parser.parse_args()

    
    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.safe_load(f)


    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)

    # Initialize the trainer
    if args.type == "spillage":
        trainer = predict_spillage(configs)
    elif args.type == "scoop":
        trainer = predict_scoop(configs)
    else : 
        trainer = selfsupervised(configs)
        
    # if configs["load"] :
    #     print("start validatioln")
    #     trainer.validate()
    # else : 
    print("start training")
    trainer.train()
