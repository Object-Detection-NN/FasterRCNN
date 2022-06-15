from utils.utilities import *
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import detection.transforms as T


if __name__ == "__main__":
    """Training preparations"""
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=["person", "car"],
        max_samples=1200,
        dataset_name="coco-train-1000",
    )
    dataset.compute_metadata()
    dataset.persistent = True
    class_list = ["person", "car"]
    class_view = dataset.filter_labels("ground_truth",
                                       F("label").is_in(class_list))

    train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
    test_transforms = T.Compose([T.ToTensor()])

    # split the dataset in train and test set
    train_view = class_view.take(1000, seed=69)
    test_view = class_view.exclude([s.id for s in train_view])
    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
                                         classes=class_list)
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
                                              classes=class_list)
    model = get_model(len(class_list) + 1)
    load_checkpoint_only_model('./model_10K/epoch-32.pth', model)
    add_detections(model, torch_dataset_test, dataset, field_name="predictions")
    results = fo.evaluate_detections(
        test_view,
        "predictions",
        classes=["person", "car"],
        eval_key="eval",
        compute_mAP=True
    )
    results.mAP()
    results.print_report()
    results_interclass = fo.evaluate_detections(
        test_view,
        "predictions",
        classes=["person", "car"],
        compute_mAP=True,
        classwise=False
    )
    results_interclass.plot_confusion_matrix()
    session = fo.launch_app(dataset)
    session.view = class_view
    session.view = test_view.filter_labels("predictions", F("confidence") > 0.5)
    session.wait()
