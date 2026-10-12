"""Request validation (input guardrails).

* Categorical fields are enums over the OULAD vocabularies (v4 accepted any string of any
  length - and forwarded it into LLM prompts).
* Numeric bounds follow the *real* data ranges (v4 rejected 739 genuine OULAD rows because
  ``last_activity_day`` can be negative; that field is replaced by ``days_since_last_activity``).
* Cross-field rules: min <= mean/weighted <= max scores, submissions <= due, active days <=
  snapshot window, etc.
* Free-text query strings are sanitised and screened for prompt-injection patterns.
* All feature fields are optional: missing values are imputed server-side from module medians
  and reported back (``input_quality``) instead of silently defaulting to 0/50.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ssa.agents.guardrails import injection_signals, sanitize_text
from ssa.data import vocab

Module = Literal["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATION_RE = vocab.PRESENTATION_PATTERN
Gender = Literal["F", "M", "Unknown"]
Region = Literal[vocab.REGIONS + ("Unknown",)]  # type: ignore[valid-type]
Education = Literal[vocab.EDUCATION + ("Unknown",)]  # type: ignore[valid-type]
ImdBand = Literal[vocab.IMD_BANDS + ("10-20", "Unknown")]  # type: ignore[valid-type]
AgeBand = Literal["0-35", "35-55", "55<=", "Unknown"]
Outcome = Literal["Distinction", "Pass", "Fail", "Withdrawn"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StudentRef(Strict):
    id_student: int = Field(gt=0, lt=10_000_000)
    code_module: Module
    code_presentation: str = Field(pattern=PRESENTATION_RE)


class FeatureProfile(Strict):
    """Snapshot features (all optional). Units follow the feature pipeline."""
    id_student: int | None = Field(None, gt=0, lt=10_000_000)
    code_module: Module
    code_presentation: str | None = Field(None, pattern=PRESENTATION_RE)
    gender: Gender | None = None
    region: Region | None = None
    highest_education: Education | None = None
    imd_band: ImdBand | None = None
    age_band: AgeBand | None = None
    num_of_prev_attempts: int | None = Field(None, ge=0, le=10)
    studied_credits: int | None = Field(None, ge=0, le=700)
    registration_days_before_start: float | None = Field(None, ge=-300, le=400)
    late_registration: int | None = Field(None, ge=0, le=1)
    snapshot_day: int | None = Field(None, ge=7, le=300)
    assessments_due: int | None = Field(None, ge=0, le=30)
    submitted_count: int | None = Field(None, ge=0, le=40)
    submitted_due_count: int | None = Field(None, ge=0, le=30)
    missed_count: int | None = Field(None, ge=0, le=30)
    missed_rate: float | None = Field(None, ge=0, le=1)
    no_scores_yet: int | None = Field(None, ge=0, le=1)
    mean_score: float | None = Field(None, ge=0, le=100)
    weighted_score: float | None = Field(None, ge=0, le=100)
    score_std: float | None = Field(None, ge=0, le=60)
    min_score: float | None = Field(None, ge=0, le=100)
    max_score: float | None = Field(None, ge=0, le=100)
    last_score: float | None = Field(None, ge=0, le=100)
    score_slope: float | None = Field(None, ge=-100, le=100)
    late_rate: float | None = Field(None, ge=0, le=1)
    mean_days_early: float | None = Field(None, ge=-30, le=60)
    total_clicks: float | None = Field(None, ge=0, le=200_000)
    active_days: int | None = Field(None, ge=0, le=400)
    resources_used: int | None = Field(None, ge=0, le=2000)
    activity_types_used: int | None = Field(None, ge=0, le=25)
    pre_start_clicks: float | None = Field(None, ge=0, le=100_000)
    days_since_last_activity: float | None = Field(None, ge=0, le=300)
    weeks_active: int | None = Field(None, ge=0, le=45)
    engagement_slope: float | None = Field(None, ge=-5000, le=5000)
    engagement_momentum: float | None = Field(None, ge=-1, le=1000)
    engagement_drop_ratio: float | None = Field(None, ge=0, le=1)
    longest_inactive_streak: int | None = Field(None, ge=0, le=45)
    inactive_weeks_last3: int | None = Field(None, ge=0, le=3)
    clicks_pct: float | None = Field(None, ge=0, le=1)
    score_pct: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def consistent(self):
        def given(*xs):
            return all(x is not None for x in xs)
        if given(self.min_score, self.max_score) and self.min_score > self.max_score:
            raise ValueError("min_score cannot exceed max_score")
        for name in ("mean_score", "weighted_score", "last_score"):
            v = getattr(self, name)
            if v is not None and given(self.min_score) and v < self.min_score - 1e-6:
                raise ValueError(f"{name} cannot be below min_score")
            if v is not None and given(self.max_score) and v > self.max_score + 1e-6:
                raise ValueError(f"{name} cannot exceed max_score")
        if given(self.submitted_due_count, self.assessments_due) and self.submitted_due_count > self.assessments_due:
            raise ValueError("submitted_due_count cannot exceed assessments_due")
        if given(self.missed_count, self.assessments_due) and self.missed_count > self.assessments_due:
            raise ValueError("missed_count cannot exceed assessments_due")
        if given(self.active_days, self.snapshot_day) and self.active_days > self.snapshot_day + 60:
            raise ValueError("active_days cannot exceed the observation window")
        if given(self.no_scores_yet) and self.no_scores_yet == 1 and any(
                v is not None for v in (self.mean_score, self.weighted_score)):
            raise ValueError("no_scores_yet=1 is inconsistent with a supplied score")
        if given(self.total_clicks, self.active_days) and self.total_clicks == 0 and self.active_days > 0:
            raise ValueError("active_days > 0 requires total_clicks > 0")
        return self


class StudentInput(Strict):
    """Either a real student reference, a feature profile, or a reference plus what-if overrides."""
    student: StudentRef | None = None
    profile: FeatureProfile | None = None
    what_if: dict[str, float | int | None] | None = None

    @model_validator(mode="after")
    def one_source(self):
        if self.student is None and self.profile is None:
            raise ValueError("provide 'student' (reference) or 'profile' (features)")
        if self.student is not None and self.profile is not None:
            raise ValueError("provide either 'student' or 'profile', not both")
        if self.what_if is not None:
            if self.student is None:
                raise ValueError("what_if overrides require a 'student' reference")
            allowed = set(FeatureProfile.model_fields) - {"id_student", "code_module", "code_presentation", "gender", "region",
                                                          "highest_education", "imd_band", "age_band"}
            unknown = set(self.what_if) - allowed
            if unknown:
                raise ValueError(f"what_if contains unsupported fields: {sorted(unknown)}")
            try:
                FeatureProfile(code_module=self.student.code_module, **self.what_if)
            except Exception as exc:  # surface nested validation as a field error
                raise ValueError(f"invalid what_if values: {exc}") from None
        return self


class RetrievalFilters(Strict):
    code_module: Module | list[Module] | None = None
    code_presentation: str | list[str] | None = None
    region: Region | list[Region] | None = None
    highest_education: Education | list[Education] | None = None
    age_band: AgeBand | list[AgeBand] | None = None
    studied_credits: int | dict[str, float] | None = None
    num_of_prev_attempts: int | dict[str, float] | None = None
    final_result: Outcome | list[Outcome] | None = None

    @field_validator("studied_credits", "num_of_prev_attempts")
    @classmethod
    def range_keys(cls, v):
        if isinstance(v, dict):
            bad = set(v) - {"min", "max", "eq", "ne"}
            if bad:
                raise ValueError(f"unsupported range operators {sorted(bad)}; use min/max/eq/ne")
            if "min" in v and "max" in v and v["min"] > v["max"]:
                raise ValueError("min cannot exceed max")
        return v


class RetrievalRequest(Strict):
    student: StudentRef | None = None
    profile: FeatureProfile | None = None
    query: str | None = Field(None, min_length=3, max_length=500)
    filters: RetrievalFilters | None = None
    top_k: int = Field(10, ge=1, le=50)
    mode: Literal["hybrid", "behavioural", "lexical", "dense"] = "hybrid"
    historical_only: bool = True

    @field_validator("query")
    @classmethod
    def clean_query(cls, v):
        if v is None:
            return v
        v = sanitize_text(v, 500)
        if hits := injection_signals(v):
            raise ValueError(f"query rejected by input guardrail (instruction-like content: {hits})")
        return v

    @model_validator(mode="after")
    def something(self):
        if self.student is None and self.profile is None and not self.query:
            raise ValueError("provide a student reference, a profile or a text query")
        return self


class AnalyzeRequest(StudentInput):
    use_llm: bool = True
    judge: bool = False


class PrioritizeRequest(Strict):
    code_module: Module | None = None
    code_presentation: str | None = Field(None, pattern=PRESENTATION_RE)
    students: list[StudentRef] | None = Field(None, min_length=1, max_length=2000)
    profiles: list[FeatureProfile] | None = Field(None, min_length=1, max_length=500)
    capacity: float | None = Field(0.2, gt=0, le=1)
    limit: int = Field(100, ge=1, le=2000)

    @model_validator(mode="after")
    def exactly_one(self):
        sources = [self.code_module is not None or self.code_presentation is not None, self.students is not None,
                   self.profiles is not None]
        if sum(sources) != 1:
            raise ValueError("provide exactly one of: code_module+code_presentation, students, profiles")
        if sources[0] and (self.code_module is None or self.code_presentation is None):
            raise ValueError("code_module and code_presentation are both required")
        return self


class ActionFeedback(Strict):
    accepted: bool
    useful: bool | None = None


class FeedbackRequest(Strict):
    recommendation_id: str = Field(min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")
    id_student: int = Field(gt=0)
    advisor_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    rating: int = Field(ge=1, le=5)
    plan_accepted: bool
    risk_assessment_agrees: bool | None = None
    action_feedback: dict[str, ActionFeedback] | None = None
    case_feedback: dict[str, bool] | None = None
    observed_outcome: Outcome | None = None
    intervention_effective: bool | None = None
    notes: str = Field("", max_length=2000)

    @field_validator("notes")
    @classmethod
    def clean_notes(cls, v):
        return sanitize_text(v, 2000)

    @field_validator("action_feedback")
    @classmethod
    def action_ids(cls, v):
        if v and any(not k.startswith("ACT_") for k in v):
            raise ValueError("action_feedback keys must be action ids (ACT_...)")
        return v


# ---------------------------------------------------------------------- v6 additions
def _ckpt():
    return Field(None, ge=0.05, le=0.95, description="Share of the planned course length; nearest built checkpoint is used")


class CheckpointMixin(Strict):
    checkpoint: float | None = _ckpt()


class PredictRequest(StudentInput):
    checkpoint: float | None = _ckpt()


class TimelineRequest(Strict):
    student: StudentRef


class AnalyzeRequestV2(AnalyzeRequest):
    checkpoint: float | None = _ckpt()
    synthesize: bool = True


class RetrievalRequestV2(RetrievalRequest):
    checkpoint: float | None = _ckpt()
    understand: bool = True
    descriptors: list[str] | None = Field(None, max_length=30)


class ParseQueryRequest(Strict):
    query: str = Field(min_length=3, max_length=500)

    @field_validator("query")
    @classmethod
    def clean(cls, v):
        v = sanitize_text(v, 500)
        if hits := injection_signals(v):
            raise ValueError(f"query rejected by input guardrail (instruction-like content: {hits})")
        return v


class PrioritizeRequestV2(PrioritizeRequest):
    checkpoint: float | None = _ckpt()


class BriefingRequest(Strict):
    code_module: Module
    code_presentation: str = Field(pattern=PRESENTATION_RE)
    capacity: float = Field(0.2, gt=0, le=1)
    checkpoint: float | None = _ckpt()


class ChatTurn(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class CopilotRequest(Strict):
    question: str = Field(min_length=2, max_length=1500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    checkpoint: float | None = _ckpt()
    use_llm: bool = True


class OutreachRequest(Strict):
    message: str = Field(min_length=20, max_length=1500)
    channel: Literal["email", "sms", "phone_script"] = "email"
    tone: Literal["warm", "concise", "formal"] = "warm"
    language: Literal["English", "French", "Spanish", "German", "Welsh"] = "English"


class JudgeRequest(Strict):
    recommendation_id: str = Field(min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")


class HumanLabelRequest(StudentRef):
    """An advisor's blind rating of the AI-drafted analysis and plan for one evaluation case."""
    verdict: Literal["PASS", "REVIEW"]
    overall: float = Field(ge=0, le=10)
    notes: str = Field("", max_length=1000)
    reviewer: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.@ -]+$")

    @field_validator("notes")
    @classmethod
    def _clean_notes(cls, v: str) -> str:
        return sanitize_text(v, 1000)


class FeedbackLoopRequest(Strict):
    apply: bool = False
    approved_by: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_.@ -]+$")
