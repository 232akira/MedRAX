import re
import base64
import ast
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage


class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For LLM (.dcm or other)
        self.display_file_path = None  # For UI (always viewable format)
        # Avoid re-sending the same image_url in every turn.
        self._image_sent_in_thread = False

    def _to_existing_abs_path(self, path_value: Optional[str]) -> Optional[str]:
        """Convert a path to absolute form and ensure it exists."""
        if not path_value or not isinstance(path_value, str):
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        return str(candidate) if candidate.exists() else None

    def _current_display_path(self, display_image: Optional[str]) -> Optional[str]:
        """Return the best currently available display image path."""
        for candidate in (
            self.display_file_path,
            display_image,
            self.original_file_path,
        ):
            existing = self._to_existing_abs_path(candidate)
            if existing:
                suffix = Path(existing).suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
                    return existing
        return None

    def _extract_display_path(self, tool_output, tool_metadata) -> Optional[str]:
        """Extract a valid image path from tool output/metadata."""
        candidate_keys = ("segmentation_image_path", "image_path")
        candidates = []

        if isinstance(tool_output, dict):
            for key in candidate_keys:
                value = tool_output.get(key)
                if isinstance(value, str):
                    candidates.append(value)

        if isinstance(tool_metadata, dict):
            for key in candidate_keys:
                value = tool_metadata.get(key)
                if isinstance(value, str):
                    candidates.append(value)

        for path in candidates:
            existing = self._to_existing_abs_path(path)
            if existing:
                return existing
        return None

    def _persist_runtime_image(self, path_value: Optional[str]) -> Optional[str]:
        """Copy runtime temp image (e.g. /tmp/gradio/*) into managed temp dir."""
        src = self._to_existing_abs_path(path_value)
        if not src:
            return None
        src_path = Path(src)
        suffix = src_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".dcm"}:
            return None
        timestamp = int(time.time() * 1000)
        dst = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(src_path, dst)
        return str(dst.resolve())

    def _ensure_managed_image_paths(self, display_image: Optional[str]) -> None:
        """
        Ensure tool-facing image path always points to a managed local file.
        This avoids /tmp/gradio paths disappearing during multi-turn chats.
        """
        original_exists = self._to_existing_abs_path(self.original_file_path)
        if original_exists:
            self.original_file_path = original_exists
            return

        persisted = self._persist_runtime_image(display_image)
        if persisted:
            self.original_file_path = persisted
            if not self._to_existing_abs_path(self.display_file_path):
                self.display_file_path = persisted
            self._image_sent_in_thread = False

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path.resolve())

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = self._to_existing_abs_path(output.get("image_path"))
        else:
            self.display_file_path = str(saved_path.resolve())

        # New upload means the current image should be attached again.
        self._image_sent_in_thread = False
        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        chat_history = chat_history or []
        self._ensure_managed_image_paths(display_image)

        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        # Use original path for tools, but display path (converted PNG) for multimodal encoding
        image_path = self.original_file_path or display_image
        display_path_for_encoding = self.display_file_path or display_image
        resolved_display = self._current_display_path(display_image)
        if resolved_display:
            self.display_file_path = resolved_display

        if image_path is not None:
            # Keep a lightweight path hint for tools / prompt context.
            messages.append({"role": "user", "content": f"image_path: {image_path}"})

            # Only attach image_url once per thread to avoid backend multi-image limit errors.
            if not self._image_sent_in_thread:
                if display_path_for_encoding and Path(display_path_for_encoding).exists():
                    image_suffix = Path(display_path_for_encoding).suffix.lower()
                    mime_type = "image/png" if image_suffix == ".png" else "image/jpeg"
                    with open(display_path_for_encoding, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
                                }
                            ],
                        }
                    )
                    self._image_sent_in_thread = True

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self._current_display_path(display_image), ""

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            tool_result = {}
                            tool_metadata = {}
                            try:
                                parsed = ast.literal_eval(message.content)
                                if isinstance(parsed, (list, tuple)) and len(parsed) >= 1:
                                    if isinstance(parsed[0], dict):
                                        tool_result = parsed[0]
                                    if len(parsed) >= 2 and isinstance(parsed[1], dict):
                                        tool_metadata = parsed[1]
                                elif isinstance(parsed, dict):
                                    tool_result = parsed
                            except Exception:
                                tool_result = {"raw_result": message.content}

                            if tool_result:
                                metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                formatted_result = " ".join(
                                    line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    )
                                )

                            new_display_path = self._extract_display_path(
                                tool_result, tool_metadata
                            )
                            if new_display_path:
                                self.display_file_path = new_display_path

                            # Keep inline image bubble for image_visualizer responses.
                            if tool_name == "image_visualizer" and self.display_file_path:
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        # content=gr.Image(value=self.display_file_path),
                                        content={"path": self.display_file_path},
                                    )
                                )

                            yield chat_history, self._current_display_path(display_image), ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self._current_display_path(display_image), ""


def create_demo(agent, tools_dict):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 🏥 MedRAX
            Medical Reasoning Agent for Chest X-ray
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the X-ray...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=700, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        # Event handlers
        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path = None
            interface._image_sent_in_thread = False
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            interface._image_sent_in_thread = False
            return [], interface.display_file_path

        def handle_file_upload(file):
            if file is None:
                return None
            file_path = file.name if hasattr(file, "name") else str(file)
            return interface.handle_upload(file_path)

        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)

        dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
