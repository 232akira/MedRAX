from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


def _patch_dynamic_cache_compat() -> None:
    """
    Patch transformers DynamicCache for mixed-version compatibility.

    Some model remote code expects `DynamicCache.seen_tokens`, while newer
    transformers variants may expose only `_seen_tokens` or rely on
    `get_seq_length()`. We add a compatible property at runtime to avoid
    touching installed core packages.
    """
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore
    except Exception:
        return

    def _get_seen_tokens(cache_obj):
        if hasattr(cache_obj, "_seen_tokens"):
            return cache_obj._seen_tokens
        if hasattr(cache_obj, "get_seq_length"):
            try:
                return int(cache_obj.get_seq_length())
            except Exception:
                pass
        return 0

    def _set_seen_tokens(cache_obj, value):
        cache_obj._seen_tokens = value

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)

    # Older remote code may call cache.get_max_length()
    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(cache_obj, *args, **kwargs):
            # Returning None means "no explicit cap" in HF generation internals.
            if hasattr(cache_obj, "max_cache_len"):
                return getattr(cache_obj, "max_cache_len")
            return None
        DynamicCache.get_max_length = _get_max_length

    # Some generation paths call get_usable_length(new_seq_len)
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(cache_obj, *args, **kwargs):
            # Old signature compatibility: get_usable_length(new_seq_length, layer_idx=0)
            new_seq_length = kwargs.get("new_seq_length", None)
            layer_idx = kwargs.get("layer_idx", 0)
            if len(args) >= 1 and new_seq_length is None:
                new_seq_length = args[0]
            if len(args) >= 2:
                layer_idx = args[1]
            if new_seq_length is None:
                new_seq_length = 0
            try:
                if hasattr(cache_obj, "get_seq_length"):
                    current = int(cache_obj.get_seq_length(layer_idx))
                else:
                    current = 0
            except Exception:
                current = 0
            max_len = cache_obj.get_max_length() if hasattr(cache_obj, "get_max_length") else None
            if max_len is None:
                return current
            return max(0, min(current, int(max_len) - int(new_seq_length)))
        DynamicCache.get_usable_length = _get_usable_length


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""

    image_paths: List[str] = Field(
        ..., description="List of paths to chest X-ray images to analyze"
    )
    prompt: str = Field(..., description="Question or instruction about the chest X-ray images")
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )


class XRayVQATool(BaseTool):
    """Tool that leverages CheXagent for comprehensive chest X-ray analysis."""

    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the XRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        # Dangerous code, but works for now
        import transformers

        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"
        _patch_dynamic_cache_compat()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="eager",
        )
        self.model = self.model.to(dtype=self.dtype)
        # Force a conservative attention backend for compatibility on mixed stacks.
        if hasattr(self.model, "config"):
            try:
                self.model.config._attn_implementation = "eager"
            except Exception:
                pass
        self.model.eval()

        transformers.__version__ = original_transformers_version

    def _generate_response(self, image_paths: List[str], prompt: str, max_new_tokens: int) -> str:
        """Generate response using CheXagent model.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
        Returns:
            str: Model's response
        """
        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device=self.device)

        def _decode_sequences(generated: Any) -> Optional[str]:
            if generated is None:
                return None
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            if torch.is_tensor(sequences):
                seq = sequences[0] if sequences.dim() > 1 else sequences
            elif isinstance(sequences, (list, tuple)) and len(sequences) > 0:
                first = sequences[0]
                if torch.is_tensor(first):
                    seq = first
                else:
                    return None
            else:
                return None

            decoded = self.tokenizer.decode(seq[input_ids.size(1):], skip_special_tokens=True).strip()
            return decoded if decoded else None

        errors: List[str] = []
        # Prefer cache path first (CheXagent remote code commonly assumes cached generation),
        # then fallback to non-cache mode for compatibility.
        for use_cache in (True, False):
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        input_ids,
                        do_sample=False,
                        num_beams=1,
                        temperature=1.0,
                        top_p=1.0,
                        use_cache=use_cache,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                decoded = _decode_sequences(generated)
                if decoded is not None:
                    return decoded
                errors.append(f"use_cache={use_cache}: empty output")
            except Exception as e:
                errors.append(f"use_cache={use_cache}: {e}")

        raise RuntimeError("CheXagent generation failed; " + " | ".join(errors))

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the chest X-ray analysis.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary
        """
        try:
            # Verify image paths
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response = self._generate_response(image_paths, prompt, max_new_tokens)

            output = {
                "response": response,
            }

            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)
