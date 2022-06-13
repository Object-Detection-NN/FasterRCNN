import torch
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
from fiftyone import ViewField as F
import detection.transforms as T
import os
from PIL import ImageFile
torch.manual_seed(1)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
            self,
            fiftyone_dataset,
            transforms=None,
            gt_field="ground_truth",
            classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes


def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint_only_model(pth, model):
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def convert_torch_predictions(preds, det_id, s_id, w, h, classes):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
            preds["boxes"].cpu().detach().numpy(),
            preds["labels"].cpu().detach().numpy(),
            preds["scores"].cpu().detach().numpy()
    ):
        # Parse prediction into FiftyOne Detection object
        x0, y0, x1, y1 = bbox
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1 - x0, y1 - y0])
        det = coco_obj.to_detection((w, h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1

    detections = fo.Detections(detections=dets)

    return detections, det_id


def add_detections(model, torch_dataset, view, field_name="predictions"):
    # Run inference on a dataset and add results to FiftyOne
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    model.eval()
    model.to(device)
    image_paths = torch_dataset.img_paths
    classes = torch_dataset.classes
    det_id = 0

    with fo.ProgressBar() as pb:
        for img, targets in pb(torch_dataset):
            # Get FiftyOne sample indexed by unique image filepath
            img_id = int(targets["image_id"][0])
            img_path = image_paths[img_id]
            sample = view[img_path]
            s_id = sample.id
            w = sample.metadata["width"]
            h = sample.metadata["height"]

            # Inference
            preds = model(img.unsqueeze(0).to(device))[0]

            detections, det_id = convert_torch_predictions(
                preds,
                det_id,
                s_id,
                w,
                h,
                classes,
            )

            sample[field_name] = detections
            sample.save()

if __name__ == "__main__":
    """Training preparations"""
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=["person", "car"],
        max_samples=11000,
        dataset_name="coco-train-11000",
    )
    dataset.compute_metadata()
    dataset.persistent = True
    class_list = ["person", "car"]
    class_view = dataset.filter_labels("ground_truth",
                                       F("label").is_in(class_list))

    train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
    test_transforms = T.Compose([T.ToTensor()])

    # split the dataset in train and test set
    train_view = class_view.take(10000, seed=42)
    test_view = class_view.exclude([s.id for s in train_view])
    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
                                         classes=class_list)
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
                                              classes=class_list)
    model = get_model(len(class_list) + 1)
    load_checkpoint_only_model('./model/epoch-32.pth', model)
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
    session.view = test_view.sort_by("predictions", F("confidence") > 0.5)
    session.wait()