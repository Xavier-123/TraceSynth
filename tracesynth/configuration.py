from dotenv import load_dotenv
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
import os

load_dotenv()
load_dotenv(".local.env", override=True)


def parse_range(value: str | int) -> tuple[int, int]:
    """Parse range strings like '2~4', '2-4', or single integers."""
    if isinstance(value, int):
        return value, value
    text = str(value).strip()
    for sep in ("~", "-"):
        if sep in text:
            left, right = text.split(sep, 1)
            if left.strip().isdigit() and right.strip().isdigit():
                return int(left.strip()), int(right.strip())
    if text.isdigit():
        n = int(text)
        return n, n
    raise ValueError(f"Invalid range value: {value!r}")


def format_range(value: str | int, unit: str = "") -> str:
    """Format a range for prompt injection, e.g. '2~4 个'."""
    lo, hi = parse_range(value)
    suffix = f" {unit}" if unit else ""
    if lo == hi:
        return f"{lo}{suffix}"
    return f"{lo}~{hi}{suffix}"


class SynthesisComplexity(BaseModel):
    """Controls task complexity and iteration complexity for Agentic RAG data synthesis."""

    num_tools: str = Field(default="4~6", description="Number of RAG tools to design (>=4 to cover step2~5).")
    num_custom_tools: str = Field(default="1", description="Number of custom virtual RAG components.")
    distractor_tools: str = Field(default="1~2", description="Number of distractor tools in tool_check.")
    max_iterations: str = Field(
        default="1~2",
        description="Expected step5 to step2 retrieval iteration rounds in task design and solving.",
    )

    @classmethod
    def from_run_config(cls, run_config: Optional[Dict[str, Any]] = None) -> "SynthesisComplexity":
        """Build complexity settings from YAML run_config (with legacy key support)."""
        if not run_config:
            return cls()

        values: Dict[str, Any] = {}

        synthesis = run_config.get("synthesis") or {}
        task_cfg = synthesis.get("task_complexity") or {}
        iter_cfg = synthesis.get("iteration_complexity") or {}

        field_names = set(cls.model_fields.keys())
        for key, val in {**task_cfg, **iter_cfg}.items():
            if key in field_names and val is not None:
                values[key] = str(val)

        # Legacy: ToolSetGenAgent.num_tools
        legacy = run_config.get("ToolSetGenAgent") or run_config.get("tool_set_gen") or {}
        if "num_tools" in legacy and "num_tools" not in values:
            values["num_tools"] = str(legacy["num_tools"])

        # Legacy: map retrieval_rounds → max_iterations
        if "retrieval_rounds" in iter_cfg and "max_iterations" not in values:
            values["max_iterations"] = str(iter_cfg["retrieval_rounds"])

        return cls(**values)

    def to_prompt_vars(self) -> Dict[str, str]:
        """Return placeholder dict for str.format on prompts."""
        iter_lo, iter_hi = parse_range(self.max_iterations)

        iteration_note = (
            "无需迭代补检，单轮检索即可凑齐回答所需的全部信息。"
            if iter_hi == 0
            else (
                f"须规划 {format_range(self.max_iterations, '轮')} 步骤5到步骤2 检索迭代回路，"
                "每轮评估后根据缺口分析返回步骤2重新优化 Query 并补检。"
            )
        )

        return {
            "num_tools": format_range(self.num_tools, "个"),
            "num_custom_tools": format_range(self.num_custom_tools, "个"),
            "distractor_tools": format_range(self.distractor_tools, "个"),
            "max_iterations": format_range(self.max_iterations, "轮"),
            "min_iterations": str(iter_lo),
            "max_iterations_val": str(iter_hi),
            "iteration_requirement": iteration_note,
            "complexity_summary": (
                f"任务复杂度：设计 {format_range(self.num_tools, '个')} RAG 工具，"
                f"覆盖全部 4 类工具（检索前优化/检索/检索后优化/评估），"
                f"对齐 step2~step5 四个必经步骤，"
                f"含 {format_range(self.num_custom_tools, '个')} 自定义组件、"
                f"{format_range(self.distractor_tools, '个')} 干扰工具；"
                f"迭代复杂度：{iteration_note}"
            ),
        }


class ModelConfiguration(BaseModel):
    """Agent使用模型的配置类。"""
    model_name: str = Field(default=None, description="代理使用的模型名称。")
    api_base: str = Field(default=None, description="可选的 API 地址。")
    api_key: str = Field(default=None, description="可选的 API 密钥。")
    api_key_env: Optional[str] = Field(default=None, description="API 密钥对应的环境变量名。")
    temperature: float = Field(default=0.4, description="模型的温度参数。")
    max_tokens: int = Field(default=8192, description="模型生成的最大 token 数。")
    use_tools: bool = Field(default=True, description="是否使用工具。")
    use_thinking: bool = Field(default=False, description="是否使用思考模式。")
    api_max_retries: int = Field(default=3, description="API 瞬时错误最大尝试次数。")
    api_retry_base: float = Field(default=1.0, description="API 重试指数退避基数（秒）。")
    parse_max_retries: int = Field(default=2, description="输出解析失败后的最大重采样次数。")
    tool_call_max_retries: int = Field(default=3, description="Solver 非法 tool_call 自纠错最大次数。")

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "ModelConfiguration":
        """从 RunnableConfig 创建一个 Configuration 实例。"""
        configurable: dict[str, Any] = config.get("configurable", {}) if config else {}
        fields = getattr(cls, "model_fields", cls.__fields__)

        raw_values: Dict[str, Any] = {
            name: configurable.get(name, field.default)
            for name, field in fields.items()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        model_name = values.get("model_name")

        if "api_base" in values:
            values["api_base"] = os.getenv(values["api_base"], values["api_base"])

        if "api_key" not in values and "api_key_env" in values:
            api_key_env = values.pop("api_key_env")
            if api_key_env in ("", "EMPTY", None):
                values["api_key"] = ""
            elif isinstance(api_key_env, str) and api_key_env.startswith("sk-"):
                raise ValueError(
                    f"api_key_env for model '{model_name}' must be an environment variable name, not a literal key"
                )
            else:
                api_key = os.getenv(api_key_env)
                if api_key is None:
                    raise ValueError(f"Missing environment variable '{api_key_env}' for model '{model_name}'")
                values["api_key"] = api_key
        elif "api_key_env" in values:
            values.pop("api_key_env")

        return cls(**values)

