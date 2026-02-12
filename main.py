import os
import warnings
from pathlib import Path
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()

def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="Qwen3-VL-30B-A3B-Instruct",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),                                         # 1.图像可视化
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),                          # 2.DICOM文件处理
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),                    # 3.病理分类
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),                # 4.结构分割
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(                         # 5.报告生成
            cache_dir=model_dir, device=device
        ),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),                       # 6.分析问答
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=False), # 7.医学视觉问答
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(                                   # 8.医学短语定位
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=False, device=device
        ),
        # "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(                                     # 9.胸片图像生成
        #     model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        # ),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",          # 1.图像可视化
        "DicomProcessorTool",           # 2.DICOM文件处理
        "ChestXRayClassifierTool",      # 3.病理分类
        "ChestXRaySegmentationTool",    # 4.结构分割
        "ChestXRayReportGeneratorTool", # 5.报告生成
        "XRayVQATool",                  # 6.分析问答
        "LlavaMedTool",                 # 7.医学视觉问答
        "XRayPhraseGroundingTool",      # 8.医学短语定位
        # "ChestXRayGeneratorTool",       # 9.胸片图像生成
    ]

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    prompt_path = (Path(__file__).resolve().parent / "medrax" / "docs" / "system_prompts.txt")
    agent, tools_dict = initialize_agent(
        str(prompt_path),
        tools_to_use=selected_tools,
        model_dir="/model-weights",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="Qwen3-VL-30B-A3B-Instruct",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=False)
