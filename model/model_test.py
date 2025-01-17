# ------------------------------------------------------------------------
# Usage Example (adjust `recording_dir` and `classes` as you need):

from model import predict_trajectory, save_predictions_to_csv


if __name__ == "__main__":
    my_recording_dir = "demo_recording_dir"

    # Classes you want to detect
    my_classes = ["green soda can", "white mug"]

    # results = capture_and_process_once(my_recording_dir, my_classes)
    # print("--"*50)
    # x1,y1,z1 = results['white mug']
    x1, y1, z1 = -469.49,   -721.5051806449866, 152.6567254316445
    x2, y2, z2 = -370.808101, -735.661648, 104.0947533
    container = [[x1, y1, z1, x2, y2, z2]]
    print(container)
    preds = predict_trajectory('pouring_trajectory_model.pth', container)
    csv = save_predictions_to_csv(preds, 'predicted_trajectories.csv')
    print(preds)