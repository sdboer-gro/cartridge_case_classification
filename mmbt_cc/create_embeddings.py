import numpy as np
import torch
from train import get_args, get_model, set_seed, get_data_loaders
import os
import argparse


def get_embeddings(args, model, train_loader, val_loader, test_loader):
    """
    This function is to make and save the embeddings for the train and test sets. The embeddings can be used in the
    trust scores.
    """
    targets = []
    embeddings = []
    predictions = []
    set_seed(args.seed)

    for batch in train_loader:
        txt, segment, mask, img, tgt = batch
        txt, img = txt.to(args.device), img.to(args.device)
        mask, segment = mask.to(args.device), segment.to(args.device)
        if args.model == "mmbt":
            for param in model.enc.img_encoder.parameters():
                param.requires_grad = False
            for param in model.enc.encoder.parameters():
                param.requires_grad = False
            x = model.enc(txt, mask, segment, img)
            out = model(txt, mask, segment, img)
        else:
            assert args.model == "img"
            x = model.img_encoder(img)
            out = model(img)
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        predictions.append(pred)
        targets.append(tgt.detach().cpu().numpy())
        embeddings.append(x.detach().cpu().numpy())

    for batch in val_loader:
        txt, segment, mask, img, tgt = batch
        txt, img = txt.to(args.device), img.to(args.device)
        mask, segment = mask.to(args.device), segment.to(args.device)
        if args.model == "mmbt":
            for param in model.enc.img_encoder.parameters():
                param.requires_grad = False
            for param in model.enc.encoder.parameters():
                param.requires_grad = False
            x = model.enc(txt, mask, segment, img)
            out = model(txt, mask, segment, img)
        else:
            assert args.model == "img"
            x = model.img_encoder(img)
            out = model(img)
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        predictions.append(pred)
        targets.append(tgt.detach().cpu().numpy())
        embeddings.append(x.detach().cpu().numpy())

    np.save(os.path.join(args.savedir, 'train_embeddings'), np.array(np.vstack(embeddings)))
    np.save(os.path.join(args.savedir, 'train_targets'), np.array(np.hstack(targets)))
    np.save(os.path.join(args.savedir, 'train_predictions'), np.array(np.hstack(predictions)))

    ttargets = []
    tembeddings = []
    for batch in test_loader:
        txt, segment, mask, img, tgt = batch
        txt, img = txt.to(args.device), img.to(args.device)
        mask, segment = mask.to(args.device), segment.to(args.device)
        if args.model == "mmbt":
            for param in model.enc.img_encoder.parameters():
                param.requires_grad = False
            for param in model.enc.encoder.parameters():
                param.requires_grad = False
            x = model.enc(txt, mask, segment, img)
        else:
            assert args.model == "img"
            x = model.img_encoder(img)
        ttargets.append(tgt.detach().cpu().numpy())
        tembeddings.append(x.detach().cpu().numpy())

    np.save(os.path.join(args.savedir, 'test_embeddings'), np.array(np.vstack(tembeddings)))
    np.save(os.path.join(args.savedir, 'test_targets'), np.array(np.hstack(ttargets)))

    return tembeddings, ttargets, embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

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

    test_embeddings, test_targets, background = get_embeddings(args, model, train_loader, val_loader, test_loader)

