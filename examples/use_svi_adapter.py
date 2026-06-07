"""Minimal SVI-OmniMem adapter usage.

Run after installing OmniMem and the optional ``omni_memory`` package:
    python examples/use_svi_adapter.py
"""

from omni_memory.orchestrator import OmniMemoryOrchestrator
from svi_omnimem import SVIOmniMemAdapter, SVIConfig


def main() -> None:
    memory = OmniMemoryOrchestrator(data_dir="./omni_memory_data_svi")
    svi = SVIOmniMemAdapter(memory, SVIConfig())

    result = svi.add_image_structured(
        "example.jpg",
        text_context="User shared a desk photo.",
        tags=["demo"],
        force=True,
    )
    print(result.to_dict() if hasattr(result, "to_dict") else result)

    query_result = svi.query_structured_visual(
        "What color is the mug in the desk photo?",
        top_k=5,
        verify=True,
        verification_budget=3,
    )
    print(query_result.answer_context)


if __name__ == "__main__":
    main()
