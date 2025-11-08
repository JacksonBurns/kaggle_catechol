import pandas as pd
import tqdm

from custommlp_model import CustomMLPModel
from utils import generate_leave_one_out_splits, generate_leave_one_ramp_out_splits, load_data

if __name__ == "__main__":
    X, Y = load_data("single_solvent")
    split_generator = generate_leave_one_out_splits(X, Y)
    all_predictions = []
    for fold_idx, split in tqdm.tqdm(enumerate(split_generator)):
        (train_X, train_Y), (test_X, test_Y) = split
        model = CustomMLPModel()
        model.train_model(train_X, train_Y)
        predictions = model.predict(test_X)
        predictions_np = predictions.detach().cpu().numpy()
        for row_idx, row in enumerate(predictions_np):
            all_predictions.append({
                "task": 0,
                "fold": fold_idx,
                "row": row_idx,
                "target_1": row[0],
                "target_2": row[1],
                "target_3": row[2]
            })
    submission_single_solvent = pd.DataFrame(all_predictions)
    X, Y = load_data("full")
    split_generator = generate_leave_one_ramp_out_splits(X, Y)
    all_predictions = []
    for fold_idx, split in tqdm.tqdm(enumerate(split_generator)):
        (train_X, train_Y), (test_X, test_Y) = split
        model = CustomMLPModel()
        model.train_model(train_X, train_Y)
        predictions = model.predict(test_X)
        predictions_np = predictions.detach().cpu().numpy()
        for row_idx, row in enumerate(predictions_np):
            all_predictions.append({"task": 1, "fold": fold_idx, "row": row_idx, "target_1": row[0], "target_2": row[1], "target_3": row[2]})
    submission_full_data = pd.DataFrame(all_predictions)
    submission = pd.concat([submission_single_solvent, submission_full_data])
    submission = submission.reset_index()
    submission.index.name = "id"
    submission.to_csv("submission.csv", index=True)
