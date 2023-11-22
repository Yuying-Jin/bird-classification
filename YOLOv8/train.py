from ultralytics import YOLO
import torch

# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    print("Torch version:", torch.__version__)
    print("Is CUDA enabled?", torch.cuda.is_available())

    # Load a model
    # model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

    # model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='./Bird_Dataset', epochs=100, imgsz=224)
