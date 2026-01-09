from typing import ClassVar, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
import numpy as np

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class ChestXRayClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    # Chinese translation mapping for pathology names
    PATHOLOGY_CN_MAP: ClassVar[Dict[str, str]] = {
        "Atelectasis": "肺不张",
        "Cardiomegaly": "心脏扩大",
        "Consolidation": "实变",
        "Edema": "水肿",
        "Effusion": "积液",
        "Emphysema": "肺气肿",
        "Enlarged Cardiomediastinum": "纵隔扩大",
        "Fibrosis": "纤维化",
        "Fracture": "骨折",
        "Hernia": "疝",
        "Infiltration": "浸润",
        "Lung Lesion": "肺部病变",
        "Lung Opacity": "肺部阴影",
        "Mass": "肿块",
        "Nodule": "结节",
        "Pleural Thickening": "胸膜增厚",
        "Pneumonia": "肺炎",
        "Pneumothorax": "气胸",
    }

    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    model: xrv.models.DenseNet = None
    device: Optional[str] = "cuda"
    transform: torchvision.transforms.Compose = None

    def __init__(self, model_name: str = "densenet121-res224-all", device: Optional[str] = "cuda"):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)
        # Add resize to 224x224 to match model input size and avoid warnings
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, normalizes it, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = img[None, :, :]
        img = self.transform(img)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(self.device)

        return img

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image for multiple pathologies.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        try:
            img = self._process_image(image_path)

            with torch.inference_mode():
                preds = self.model(img).cpu()[0]

            # Convert numpy array to Python native types for JSON serialization
            preds_np = preds.numpy()
            # Create output with both English and Chinese names
            output = {}
            for pathology, prob in zip(xrv.datasets.default_pathologies, preds_np):
                cn_name = self.PATHOLOGY_CN_MAP.get(pathology, pathology)
                output[f"{pathology} ({cn_name})"] = float(prob)
            
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": "概率值范围0-1，数值越高表示该病理存在的可能性越大。",
            }
            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path)
