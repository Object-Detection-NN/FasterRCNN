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


def save_checkpoint(state, pth):
    f_path = pth
    torch.save(state, f_path)


def load_checkpoint(pth, model, optimizer, lr_scheduler):
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, lr_scheduler, checkpoint['epoch']


def do_training(model, torch_dataset, torch_dataset_test, num_epochs=4, ckp_pth=''):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        torch_dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.5)

    start_epoch = 0
    if ckp_pth:
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(ckp_pth, model, optimizer, lr_scheduler)
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch, printing every 100 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        # update the learning rate
        lr_scheduler.step()
        # save model every epoch
        checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict()
                   }
        save_checkpoint(checkpoint, os.path.join('./model', 'epoch-{}.pth'.format(epoch)))
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), './model/model_final.pth')


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

    # use our dataset and defined transformations
    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
                                         classes=class_list)
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
                                              classes=class_list)
    model = get_model(len(class_list) + 1)
    do_training(model, torch_dataset, torch_dataset_test, num_epochs=40, ckp_pth='./model/epoch-31.pth')
