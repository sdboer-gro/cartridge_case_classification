import numpy as np
import torch
from train import get_args, get_model, set_seed, get_data_loaders
import os
import argparse
from trustscore.shap_score import get_shap_values

def get_classifier(args):
    """
    Function used to only get the trained classifier module from the models.
    """
    model = get_model(args)
    model.to(args.device)
    if args.model == "img":
        from models.image import Classifier

        classifier = Classifier(args)
    else:
        assert args.model == "mmbt"
        from models.mmbt import Classifier

        classifier = Classifier(args)

    best_checkpoint = torch.load(os.path.join(args.savedir, "model_best.pt"), map_location=torch.device('cpu'))
    model.load_state_dict(best_checkpoint["state_dict"])

    model_dict = model.state_dict()
    processed_dict = {"clf.weight": model_dict["clf.weight"], "clf.bias": model_dict["clf.bias"]}

    classifier.load_state_dict(processed_dict, strict=False)
    classifier.to(args.device)

    model.eval()
    classifier.eval()
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    classifier = get_classifier(args)

    test_embeddings = np.load(os.path.join(args.savedir, 'test_embeddings.npy'))
    test_targets = np.load(os.path.join(args.savedir, 'test_targets.npy'))
    background = np.load(os.path.join(args.savedir, 'train_embeddings.npy'))

    test_embeddings = torch.from_numpy(np.array(test_embeddings)).to(args.device)
    test_embeddings.requires_grad_()
    background = torch.from_numpy(np.array(background)).to(args.device)

    print("finding shapley values....")
    shap_vector_train, shap_vector_test = get_shap_values(args, train_feature_vectors=background,
                                                          test_feature_vectors=test_embeddings,
                                                          classifier_model=classifier)