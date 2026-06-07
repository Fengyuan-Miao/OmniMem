import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from svi_omnimem import (
    IndexedAttribute,
    RetrievalAnchor,
    StructuredVisualCard,
    SVIConfig,
    SVIOmniMemAdapter,
    VerificationResult,
    VisualQueryRequirement,
)
from svi_omnimem.models import OCRObservation
from svi_omnimem.promoter import VerifiedFactPromoter
from svi_omnimem.query_parser import VisualQueryParser
from svi_omnimem.retriever import StructuredVisualRetriever
from svi_omnimem.stores import StructuredVisualStore, VerifiedFactStore
from svi_omnimem.extractor import StructuredVisualExtractor
from memgallery_svi_pipeline import (
    answer_format_instruction,
    direct_answer_from_verified_images,
)


@dataclass
class FakeStorageConfig:
    index_dir: str


@dataclass
class FakeLLMConfig:
    caption_model: str = "fake-vlm"


@dataclass
class FakeConfig:
    storage: FakeStorageConfig
    llm: FakeLLMConfig = field(default_factory=FakeLLMConfig)


class FakeColdStorage:
    def __init__(self, image_bytes):
        self.image_bytes = image_bytes

    def retrieve(self, pointer):
        if pointer == "raw://image":
            return self.image_bytes
        return None


class FakeMauStore:
    def __init__(self):
        self.items = {}

    def get(self, mau_id):
        return self.items.get(mau_id)

    def update(self, mau):
        self.items[mau.id] = mau
        return True


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = None


class FakeCompletions:
    def __init__(self, content):
        if isinstance(content, list):
            self.contents = list(content)
        else:
            self.contents = [content]
        self.index = 0

    def create(self, **kwargs):
        content = self.contents[min(self.index, len(self.contents) - 1)]
        self.index += 1
        return FakeResponse(content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeCompletions(content)


class FakeLLMClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


@dataclass
class FakeMetadata:
    session_id: str = "s1"
    tags: list = field(default_factory=list)
    keywords: list = field(default_factory=list)


@dataclass
class FakeMau:
    id: str = "img_1"
    timestamp: float = 1710000000.0
    summary: str = "A red mug is on the left side of a laptop."
    raw_pointer: str = "raw://image"
    metadata: FakeMetadata = field(default_factory=FakeMetadata)
    details: dict = field(default_factory=dict)
    related: list = field(default_factory=list)

    def add_related(self, mau_id):
        if mau_id not in self.related:
            self.related.append(mau_id)


@dataclass
class FakeProcessingResult:
    success: bool
    mau: FakeMau
    metadata: dict = field(default_factory=dict)


class FakeOrchestrator:
    def __init__(self, tmp_path, image_bytes, planner_response=None):
        self.config = FakeConfig(storage=FakeStorageConfig(index_dir=str(tmp_path)))
        self.cold_storage = FakeColdStorage(image_bytes)
        self.mau_store = FakeMauStore()
        self.session_id = "s1"
        self.planner_response = planner_response
        self._planner_client = (
            FakeLLMClient(planner_response) if planner_response is not None else None
        )

    def add_image(self, image, session_id=None, tags=None, force=False):
        mau = FakeMau()
        mau.metadata.session_id = session_id
        mau.metadata.tags = tags or []
        self.mau_store.items[mau.id] = mau
        return FakeProcessingResult(True, mau)

    def add_text(self, text, session_id=None, tags=None, force=False):
        mau = FakeMau(id="mirror_1", summary=text, raw_pointer="")
        mau.metadata.session_id = session_id
        mau.metadata.tags = tags or []
        self.mau_store.items[mau.id] = mau
        return FakeProcessingResult(True, mau)

    def _get_llm_client(self):
        if self._planner_client is not None:
            return self._planner_client
        raise RuntimeError("LLM should not be called in this test")


def _image_bytes():
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _sample_card():
    return StructuredVisualCard(
        card_id="card_1",
        image_mau_id="img_1",
        session_id="s1",
        turn_id="t1",
        observed_at="2026-01-01T00:00:00+00:00",
        raw_pointer="raw://image",
        global_caption="A red mug is on the left side of a laptop.",
        retrieval_anchors=[
            RetrievalAnchor(
                anchor_id="a1",
                category="mug",
                salient_attributes={
                    "color": IndexedAttribute(value="red")
                },
            ),
            RetrievalAnchor(anchor_id="a2", category="cup"),
            RetrievalAnchor(anchor_id="a3", category="laptop"),
        ],
        ocr_observations=[OCRObservation(text="meeting 3pm", context="paper note")],
        tags=["demo"],
    )


def _image_card(card_id, image_mau_id, observed_at, global_caption, image_tag, turn_id, source_text_context):
    return StructuredVisualCard(
        card_id=card_id,
        image_mau_id=image_mau_id,
        session_id="D1" if image_tag.startswith("D1:") else "D2",
        turn_id=turn_id,
        observed_at=observed_at,
        raw_pointer="raw://image",
        global_caption=global_caption,
        source_text_context=source_text_context,
        tags=[f"image_id:{image_tag}"],
    )


def test_card_roundtrip_and_store_search(tmp_path):
    store = StructuredVisualStore(str(tmp_path))
    store.append(_sample_card())

    loaded = store.get_by_card_id("card_1")
    assert loaded is not None
    assert loaded.to_mirror_text().startswith("[Structured visual index")

    entity_hits = store.search_entity_alias(["cup"])
    assert entity_hits[0][0].card_id == "card_1"

    attr_hits = store.search_attribute(["cup"], ["color"])
    assert attr_hits[0][2].startswith("attribute:")

    ocr_hits = store.search_ocr(["meeting"])
    assert ocr_hits[0][2].startswith("ocr:")


def test_generic_retriever_uses_card_text_without_query_priors(tmp_path):
    visual_store = StructuredVisualStore(str(tmp_path))
    fact_store = VerifiedFactStore(str(tmp_path))
    mug = _sample_card()
    laptop = _sample_card()
    laptop.card_id = "card_2"
    laptop.image_mau_id = "img_2"
    laptop.global_caption = "A silver laptop sits beside a blue notebook."
    laptop.retrieval_anchors = [
        RetrievalAnchor(anchor_id="b1", category="laptop"),
        RetrievalAnchor(anchor_id="b2", category="notebook"),
    ]
    visual_store.append(mug)
    visual_store.append(laptop)

    requirement = VisualQueryParser().parse("Which image shows a red cup?")
    retriever = StructuredVisualRetriever(visual_store, fact_store, SVIConfig())
    candidates, claims = retriever.retrieve(
        "Which image shows a red cup?",
        requirement,
        top_k=2,
    )

    assert candidates[0].image_mau_id == "img_1"
    assert "card_text" in candidates[0].routes
    assert claims


def test_date_text_helpful_for_late_july_visual_query(tmp_path):
    store = StructuredVisualStore(str(tmp_path))
    june = _image_card(
        card_id="card_june",
        image_mau_id="img_june",
        observed_at="2024-06-26",
        global_caption="Plain white T-shirt laid flat on a light gray surface.",
        image_tag="D1:IMG_001",
        turn_id="D1:6",
        source_text_context="User: Looking for a t-shirt to wear with jeans.",
    )
    july = _image_card(
        card_id="card_july",
        image_mau_id="img_july",
        observed_at="2024-07-27",
        global_caption="Plain white T-shirt laid flat on a light gray surface.",
        image_tag="D2:IMG_001",
        turn_id="D2:6",
        source_text_context="User: How about this t-shirt?",
    )
    store.append(june)
    store.append(july)

    results = store.search_all_text(
        "Which image shown in late-July is related to the T-shirt discussed in the conversation?"
    )

    assert results[0][0].card_id == "card_july"


def test_single_image_direct_answer_prefers_earlier_verified_candidate(tmp_path):
    store = StructuredVisualStore(str(tmp_path))
    early = _image_card(
        card_id="card_early",
        image_mau_id="img_early",
        observed_at="2024-07-21",
        global_caption="Young man using laptop in library aisle.",
        image_tag="D1:IMG_003",
        turn_id="D1:12",
        source_text_context="User: I started searching for materials related to psychology in the library to prepare.",
    )
    late = _image_card(
        card_id="card_late",
        image_mau_id="img_late",
        observed_at="2024-07-21",
        global_caption="Classroom lecture with a blackboard announcement.",
        image_tag="D1:IMG_004",
        turn_id="D1:20",
        source_text_context="User: Overall, I’m satisfied with my decision to minor in psychology.",
    )
    store.append(early)
    store.append(late)

    svi = SimpleNamespace(visual_store=store)
    svi_result = SimpleNamespace(
        candidates=[
            SimpleNamespace(card_id="card_late", score=0.572),
            SimpleNamespace(card_id="card_early", score=0.429),
        ],
        verified_evidence=[
            SimpleNamespace(
                source_card_id="card_late",
                supports=True,
                abstained=False,
                answer_fragment="Classroom lecture with blackboard announcement",
                visible_evidence="The image shows a lecture with red Chinese characters on the blackboard.",
                confidence=0.95,
            ),
            SimpleNamespace(
                source_card_id="card_early",
                supports=True,
                abstained=False,
                answer_fragment="Young man using laptop in library",
                visible_evidence="The image shows a young man using a laptop in a library aisle.",
                confidence=0.92,
            ),
        ],
    )
    svi_context = (
        "Verified visual evidence:\n"
        "- image_id: D1:IMG_004; date: 2024-07-21; session: D1; turn: D1:20; "
        "Classroom lecture with blackboard announcement; evidence: The image shows a lecture with red Chinese characters on the blackboard.; confidence: 0.95\n"
        "- image_id: D1:IMG_003; date: 2024-07-21; session: D1; turn: D1:12; "
        "Young man using laptop in library; evidence: The image shows a young man using a laptop in a library aisle.; confidence: 0.92"
    )

    answer = direct_answer_from_verified_images(
        "Which image shown represents Lin's efforts to explore psychology before making the decision?",
        svi_context,
        svi_result=svi_result,
        svi=svi,
    )

    assert answer == "D1:IMG_003"


def test_single_image_format_instruction_is_strict():
    assert "one matching public image id" in answer_format_instruction(
        "Which image shown represents the answer?"
    )


def test_verified_fact_promotion(tmp_path):
    store = VerifiedFactStore(str(tmp_path))
    promoter = VerifiedFactPromoter(store, SVIConfig())
    result = VerificationResult(
        supports=True,
        answer_fragment="red",
        visible_evidence="The mug is visibly red.",
        confidence=0.95,
        source_card_id="card_1",
        source_image_mau_id="img_1",
        raw_pointer="raw://image",
        observation_time="2026-01-01T00:00:00+00:00",
    )
    requirement = VisualQueryRequirement(
        requires_visual_evidence=True,
        requested_attributes=["color"],
        requires_raw_verification=True,
        query_type="attribute",
    )
    facts = promoter.promote([result], requirement)
    assert len(facts) == 1
    assert store.count() == 1


def test_adapter_add_image_structured_falls_back_without_llm(tmp_path):
    orchestrator = FakeOrchestrator(tmp_path, _image_bytes())
    adapter = SVIOmniMemAdapter(orchestrator, SVIConfig())
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))

    result = adapter.add_image_structured(
        image,
        text_context="desk setup",
        session_id="s1",
        tags=["demo"],
        force=True,
    )

    assert result.success is True
    assert "svi_card_id" in result.metadata
    assert adapter.visual_store.count() == 1
    query_result = adapter.query_structured_visual(
        "Which image shows a desk setup?",
        verify=False,
        top_k=3,
    )
    assert query_result.candidates
    assert query_result.plan is None
    assert query_result.execution_trace
    assert "retrieval hints only" in query_result.answer_context


def test_extractor_populates_caption_fallback_when_llm_unavailable(tmp_path):
    orchestrator = FakeOrchestrator(tmp_path, _image_bytes())
    extractor = StructuredVisualExtractor(orchestrator, SVIConfig())
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))

    card = extractor.extract(
        image=image,
        image_mau_id="img_1",
        raw_pointer="raw://image",
        global_caption="A Coca-Cola ad with red branding and a white bottle.",
        text_context="image_caption: Coca-Cola promotion at Esso",
        timestamp="2024-01-01",
        session_id="s1",
        turn_id="r1",
        tags=["demo"],
    )

    assert card.retrieval_anchors
    assert card.extraction_scope == "caption_dialogue_fallback"
    assert "coca" in card.to_mirror_text().lower() or "cola" in card.to_mirror_text().lower()
