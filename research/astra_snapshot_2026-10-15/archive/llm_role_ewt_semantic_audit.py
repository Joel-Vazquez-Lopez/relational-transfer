"""Source-only, AI-provisional semantic audit of the frozen EWT items.

This script reads EVENTS.json only. It never loads model predictions. The
judgments below are assistant judgments, not independent human verification.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "llm_role_independent_eval"

# One row for each of the 75 frozen events, in their original order.
# Columns: subject role, object role, semantic agent, semantic patient, note.
# Agent means an intentional or organizational doer; patient means an entity
# that clearly undergoes an effect. Themes, content, possessions, goals and
# created results are recorded separately. Borderline cases are unsure.
LABELS = [
    ("agent", "theme", "yes", "no", "Slide show is published content; grammar is nonstandard ('a excellent')."),
    ("agent", "content", "yes", "no", "Slogans are utterance content; long embedded clause."),
    ("agent", "desired_event", "yes", "no", "Talks are demanded, not affected."),
    ("recipient", "theme", "no", "no", "Cities receive treatment; subject is recipient, not agent."),
    ("agent", "document_theme", "yes", "no", "Opinion is filed content."),
    ("agent", "document_theme", "yes", "no", "Opinion is filed content."),
    ("agent", "document_theme", "yes", "no", "Opinion is filed content."),
    ("agent", "document_theme", "yes", "no", "Opinion is filed content."),
    ("agent", "theme", "yes", "no", "Half of aid is transferred; answer span 'half' is underspecified."),
    ("metonymic_agent", "patient", "unsure", "yes", "Media is a collective/metonym; distinctions are obscured."),
    ("metonymic_agent", "content", "yes", "no", "Movement issues statement; organizational agency."),
    ("agent", "idiomatic_object", "yes", "unsure", "'Pinched nerves' is figurative; subject span omits title President."),
    ("agent", "patient", "yes", "yes", "President Bush is target of criticism; object span omits title."),
    ("title_or_concept", "possession", "no", "no", "Monotheism and Holy War is likely a title; subject span truncated."),
    ("agent", "stimulus", "yes", "no", "Attack is rejected proposition/plan, not affected person."),
    ("book_as_source", "idiomatic_content", "no", "no", "'Makes the case' is an idiom; title is not an agent."),
    ("agent", "patient", "yes", "yes", "Dick Clarke undergoes demotion."),
    ("agent", "event_result", "yes", "no", "Coup is an event carried out, not affected entity; title omitted."),
    ("collective_agent", "opponent", "yes", "unsure", "Sports opponent, not a prototypical affected patient."),
    ("possessor", "possession", "no", "no", "Subject omits coordinated Mary Nell Browning; having ideas is possession."),
    ("agent", "path_or_strategy", "yes", "no", "Trying a route is figurative."),
    ("agent", "computed_value", "yes", "no", "Reserve is calculated, not physically affected."),
    ("experiencer", "predicted_content", "unsure", "no", "Gurus predict a value; subject span omits coordinated geniuses."),
    ("agent", "result", "yes", "no", "Adjustments are made/created."),
    ("set_or_container", "member", "no", "no", "Inclusion relation, no agent or patient."),
    ("causer", "patient", "no", "yes", "Results cause reduction in working gas; neither intentional agent nor human actor."),
    ("source", "theme", "no", "no", "Presentation gives a starting point figuratively."),
    ("agent", "theme", "yes", "no", "Taking responsibility is figurative transfer."),
    ("collective_agent", "supported_issue", "yes", "no", "Marriage is the issue supported."),
    ("agent", "patient", "yes", "yes", "Man receives a naming; source span 'K K Paul' differs from surface 'KK Paul'."),
    ("causer", "patient", "no", "yes", "Systems control pattern; non-intentional causal subject."),
    ("event_cause", "patient", "no", "yes", "Alteration leaves area vulnerable; subject span omits coordination."),
    ("event_cause", "patient", "no", "yes", "Results exacerbate destructive power; subject span omits coordination."),
    ("experiencer", "stimulus", "no", "no", "Seeing a story is perception, not acting on a patient."),
    ("metonymic_agent", "opponent", "yes", "unsure", "Company defies country/metonym; headline format."),
    ("metonymic_agent", "opposed_event", "yes", "no", "Google resists attempt; attempt is event/stimulus."),
    ("agent", "goal_or_visited_person", "yes", "no", "Hu Jintao is visited participant, not affected patient; other allies omitted."),
    ("agent", "body_part_in_idiom", "yes", "unsure", "Shaking hands is reciprocal phrase; subject omits title."),
    ("agent", "achieved_goal", "yes", "no", "Goals are accomplished, not an affected patient."),
    ("possessor", "possessed_event", "no", "no", "Having an exhibition does not uniquely identify an agent action."),
    ("causer_or_source", "theme", "unsure", "no", "Art combines painting and calligraphy; object span omits coordination."),
    ("metonymic_agent", "event_result", "yes", "no", "Google makes a move; organizational agency."),
    ("metonymic_agent", "evaluated_option", "yes", "no", "Company evaluates ways; subject omits Wikimedia Foundation coordination."),
    ("agent", "visited_place", "yes", "no", "School is a destination, not patient."),
    ("agent", "patient", "yes", "yes", "Hand changes position; possessive phrase linked to actor."),
    ("agent", "patient", "yes", "yes", "Hand changes position; subject span omits location modifier."),
    ("possessor", "possession", "no", "no", "Having an idea is possession."),
    ("possessor", "possession", "no", "no", "Having an impediment is possession/state."),
    ("source", "provided_theme", "no", "no", "Coffee metaphorically provides food groups; non-agentive source."),
    ("possessor", "feature", "no", "no", "Mail has spam protection; possession/feature."),
    ("agent", "rated_content", "yes", "unsure", "'Thumbs upping' affects ratings; actor phrase may omit 'of mine'."),
    ("agent", "traced_theme", "yes", "no", "Phone is located/traced, not clearly changed; interrogative."),
    ("metonymic_agent", "product_result", "yes", "no", "Company makes systems; organization agency."),
    ("possessor", "feature", "no", "no", "PCs have a feature; 'sleep' is truncated from 'sleep & charge'."),
    ("experiencer", "needed_theme", "no", "no", "Need relation, not acting upon lamp."),
    ("experiencer", "needed_theme", "no", "no", "Need relation, not acting upon heat source."),
    ("possessor", "possession", "no", "no", "Ribs have food; sentence is informal and semantically odd."),
    ("measurement_instrument", "measured_property", "no", "no", "Ounces are units, not agents; weight is measured property."),
    ("possessor", "feature", "no", "no", "Hackney's possession; subject span lacks possessive suffix."),
    ("involuntary_source", "expelled_theme", "no", "unsure", "Dog vomits bile; nonvolitional action and ungrammatical text."),
    ("agent", "created_result", "yes", "no", "Marble object is made/created, not pre-existing patient."),
    ("agent", "patient", "yes", "yes", "Food undergoes cooking."),
    ("experiencer", "known_content", "no", "no", "Knowledge state; subject omits Doctor title."),
    ("agent", "learned_subject", "yes", "no", "Taking Customer Service 101 is education, not affected patient."),
    ("agent", "event_result", "yes", "no", "Doing a job is event/performance; subject omits assistant coordination."),
    ("causer", "patient", "no", "yes", "Service saves situation figuratively; sentence fragment."),
    ("possessor", "possession", "no", "no", "Having a gift is possession/state."),
    ("experiencer", "known_content", "no", "no", "Knowing stuff is mental state, not physical agency."),
    ("possessor", "feature", "no", "no", "Rice has flavor; possession/quality."),
    ("metonymic_agent", "patient", "yes", "yes", "Company transforms back yard; subject span omits Inc."),
    ("source_or_provider", "offered_service", "unsure", "no", "Place stands for business; manicure is offered service."),
    ("causer", "patient", "no", "yes", "Location makes stay convenient; subject omits coordinated parking."),
    ("agent", "event_result", "yes", "no", "Doing work creates/performs result."),
    ("agent", "theme", "yes", "no", "Bucks are transferred/invested, not patient."),
    ("agent", "offered_service", "yes", "no", "Massage is provided service, not patient."),
]


def main():
    events = json.loads((ROOT / "EVENTS.json").read_text())
    assert len(events) == len(LABELS) == 75
    path = ROOT / "SEMANTIC_AUDIT_AI_PROVISIONAL.csv"
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "sentence", "target_verb", "grammatical_subject_UD",
                         "grammatical_object_UD", "subject_semantic_role_AI",
                         "object_semantic_role_AI", "semantic_agent_AI",
                         "semantic_patient_AI", "audit_note_AI", "human_verification"])
        for event, (sr, orole, agent, patient, note) in zip(events, LABELS, strict=True):
            writer.writerow([event["id"], event["sentence"], event["verb"],
                             event["actor_span"], event["other_span"], sr, orole,
                             agent, patient, note, "pending"])
    print({"items": len(events),
           "agent": dict(Counter(x[2] for x in LABELS)),
           "patient": dict(Counter(x[3] for x in LABELS)),
           "human_verified": 0})


if __name__ == "__main__":
    main()
