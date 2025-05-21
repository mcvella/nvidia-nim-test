import sys
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Dict, Any
import asyncio
import base64
import io
import aiohttp
import re # Added import for regular expressions

from PIL import Image

from typing_extensions import Self
from viam.media.video import ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection,
                                       GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.components.camera import Camera
from viam.utils import ValueTypes, dict_to_struct, struct_to_dict


class NvidiaNimTest(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("nvidia-demo", "service"), "nvidia-nim-test"
    )

    # Configurable attributes
    api_key: str
    model_name: str
    default_question: str
    invoke_url: str
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool = False # stream is not used in the classification logic yet.

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:        
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        attrs = struct_to_dict(config.attributes)
        api_key = attrs.get("api_key", "")
        if not api_key:
            raise Exception("api_key attribute is required for NvidiaNimTest service.")
        
        # Extract camera dependencies
        camera_names =  attrs.get("cameras", [])
        return camera_names


    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        py_attrs = struct_to_dict(config.attributes)

        # api_key is validated to exist by validate_config
        self.api_key = py_attrs["api_key"]

        self.model_name = py_attrs.get("model_name", "meta/llama-3.2-11b-vision-instruct")
        self.default_question = py_attrs.get("default_question", "describe this image")
        self.invoke_url = py_attrs.get("invoke_url", "https://integrate.api.nvidia.com/v1/chat/completions")
        self.max_tokens = int(py_attrs.get("max_tokens", 512))
        self.temperature = float(py_attrs.get("temperature", 1.0))
        self.top_p = float(py_attrs.get("top_p", 1.0))
        
        self.cameras = {}
        camera_names_from_config = py_attrs.get("cameras", [])
        for name in camera_names_from_config:
            camera = dependencies.get(Camera.get_resource_name(name))
            if not camera:
                # This case should ideally be caught by validate_config's dependency return,
                # but good to have a check here too.
                self.logger.warn(f"Camera dependency '{name}' configured but not found in dependencies map.")
                continue
            self.cameras[name] = camera


    async def _call_nvidia_api(self, image_b64: str, question: str) -> List[Classification]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json" 
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f'{question} <img src="data:image/jpeg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream 
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.invoke_url, headers=headers, json=payload) as response:
                response.raise_for_status() # Raise an exception for bad status codes
                response_data = await response.json()
        
        self.logger.debug(f"NVIDIA API Response: {response_data}")

        classifications = []
        if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
            api_content = response_data["choices"][0]["message"]["content"]

            # Check for the specific question pattern: "Is this a <description>, answer YES or NO."
            # Regex is case-insensitive for the fixed parts of the question.
            # Allows for ',' or '?' before 'answer YES or NO'.
            pattern = r"Is this an? (.*?)(?:,|\?)\s*answer YES or NO\.?"
            match = re.fullmatch(pattern, question, re.IGNORECASE)

            if match:
                description = match.group(1).strip() # This is the <description> part
                
                # Enhanced normalization:
                # 1. Lowercase
                # 2. Strip leading/trailing whitespace
                # 3. Strip common trailing punctuation
                # 4. Strip leading/trailing markdown bold/italic markers (* and _)
                normalized_api_response = api_content.lower().strip().rstrip('.!?')
                normalized_api_response = normalized_api_response.strip('*_')
                
                # Check if the normalized response STARTS WITH "yes"
                if normalized_api_response.startswith("yes"):
                    classifications.append(Classification(class_name=description, confidence=1.0))
                # If not starting with "yes" (e.g., starts with "no" or other), classifications list remains empty.
            else:
                # Original behavior: use the full API response as the classification if question pattern doesn't match
                classifications.append(Classification(class_name=api_content.strip(), confidence=1.0))
        
        return classifications

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        if camera_name not in self.cameras:
            raise ValueError(f"Camera '{camera_name}' not found in configured dependencies.")

        camera = self.cameras[camera_name]
        captured_image = await camera.get_image()

        classifications = []
        if return_classifications:
            # The `count` parameter for get_classifications is not strictly used by the NVIDIA API
            # but is part of the Viam Vision service interface. We pass 1 as a placeholder.
            classifications = await self.get_classifications(captured_image, 1, extra=extra, timeout=timeout)

        # Detections and ObjectPointClouds are not supported by this service
        detections = []
        object_point_clouds = []

        return CaptureAllResult(
            image=captured_image if return_image else None,
            classifications=classifications if return_classifications else None
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections` is not implemented")
        raise NotImplementedError()

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int, # count is not directly used with the current API, which returns one primary classification
        *,extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        if camera_name not in self.cameras:
            raise Exception(f"Camera {camera_name} not found in dependencies.")
        
        camera = self.cameras[camera_name]
        image = await camera.get_image()
        return await self.get_classifications(image, count, extra=extra, timeout=timeout)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int, # count is not directly used with the current API
        *,        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        question = self.default_question
        if extra and "question" in extra:
            question = str(extra["question"])

        # Convert ViamImage to PIL Image using viam_to_pil_image
        pil_image = viam_to_pil_image(image)

        # Save PIL image to BytesIO buffer as JPEG
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Basic size check, similar to the example script
        # You might want to make this limit configurable or handle it more gracefully
        assert len(image_b64) < 180_000, \
            "To upload larger images, use the assets API (see NVIDIA docs)"

        return await self._call_nvidia_api(image_b64, question)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

    async def get_properties(
        self,       *,       extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

