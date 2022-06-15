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
    train_view = class_view.take(1000, seed=42)
    test_view = class_view.exclude([s.id for s in train_view])

    # use our dataset and defined transformations
    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
                                         classes=class_list)
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
                                              classes=class_list)
    model = get_model(len(class_list) + 1)
    do_training(model, torch_dataset, torch_dataset_test, num_epochs=100)
