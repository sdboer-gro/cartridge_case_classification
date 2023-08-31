import shap
import numpy as np
import os

def get_shap_values(args, train_feature_vectors, test_feature_vectors, classifier_model):
    """
    Function to get the shapley values from the embeddings and the classifier modules of the models.
    """
    if args.model == "mmbt":
        to_explain = test_feature_vectors.reshape((test_feature_vectors.shape[0], 1, test_feature_vectors.shape[1])).to(args.device)
        train_feature_vectors = train_feature_vectors.reshape((train_feature_vectors.shape[0], 1, train_feature_vectors.shape[1])).to(args.device)
        background = train_feature_vectors[:10]
    else:
        to_explain = test_feature_vectors.to(args.device)
        background = train_feature_vectors.to(args.device)[:10]

    e = shap.GradientExplainer(classifier_model, background,
                               batch_size=1)

    shap_values = e.shap_values(to_explain)
    shap_vector_test = np.array(shap_values)
    print(f"feature shape: {test_feature_vectors.shape}, shap_vector_test shape: {shap_vector_test.shape}")
    np.save(os.path.join(args.savedir, 'shap_vector_test'), shap_vector_test)

    print("for train features....")
    shap_values = e.shap_values(train_feature_vectors[10:1010].to(args.device))
    shap_vector_train = np.array(shap_values)
    print(f"feature shape: {train_feature_vectors.shape}, shap_vector_train shape: {shap_vector_train.shape}")
    np.save(os.path.join(args.savedir, 'shap_vector_train'), shap_vector_train)

    print("saving done")
    return shap_vector_train, shap_vector_test
