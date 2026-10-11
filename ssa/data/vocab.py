"""Canonical OULAD vocabularies. Used for strict API validation (enums instead of free text)
and for normalising known dataset quirks (e.g. the ``10-20`` IMD band without ``%``)."""
from __future__ import annotations

MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
PRESENTATION_PATTERN = r"^20\d{2}[BJ]$"
GENDERS = ("F", "M")
REGIONS = (
    "East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region",
    "North Western Region", "Scotland", "South East Region", "South Region", "South West Region",
    "Wales", "West Midlands Region", "Yorkshire Region",
)
EDUCATION = (
    "No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification",
    "Post Graduate Qualification",
)
IMD_BANDS = ("0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%")
AGE_BANDS = ("0-35", "35-55", "55<=")
DISABILITY = ("N", "Y")
OUTCOMES = ("Distinction", "Pass", "Fail", "Withdrawn")
UNKNOWN = "Unknown"

# VLE activity types grouped into pedagogically meaningful channels.
ACTIVITY_GROUPS = {
    "content": {"oucontent", "page", "htmlactivity", "dualpane", "subpage", "sharedsubpage", "folder", "glossary"},
    "resource": {"resource", "url", "dataplus", "repeatactivity"},
    "forum": {"forumng", "ouwiki", "oucollaborate", "ouelluminate"},
    "quiz": {"quiz", "externalquiz", "questionnaire"},
    "homepage": {"homepage"},
}


def normalise_imd(value):
    if value is None or (isinstance(value, float) and value != value):
        return UNKNOWN
    v = str(value).strip()
    if v == "10-20":
        return "10-20%"
    return v if v in IMD_BANDS else UNKNOWN


def presentation_sort_key(code: str) -> tuple[int, int]:
    """'2013B' < '2013J' < '2014B' < '2014J' (B starts in February, J in October)."""
    s = str(code)
    return int(s[:4]), 0 if s.endswith("B") else 1
