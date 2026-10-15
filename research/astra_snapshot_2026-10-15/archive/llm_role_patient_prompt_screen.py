"""Exploratory prompt screen on previously used names, excluded from later tests."""
import json

import torch

import llm_role_internal_v1 as core


def main():
    tokenizer, model = core.load_model()
    sentences = [
        ("Grace greeted Henry.", "Grace", "Henry"),
        ("Henry was greeted by Grace.", "Grace", "Henry"),
        ("Henry greeted Grace.", "Henry", "Grace"),
        ("Grace was greeted by Henry.", "Henry", "Grace"),
    ]
    questions = [
        "Who was greeted?",
        "Who received the action?",
        "Who was the action done to?",
        "Which person was acted on?",
        "Who is the patient, meaning the person the action was done to?",
        "Name the person affected by the action, rather than the person who performed it.",
        "Who was greeted by the other person?",
    ]
    results = []
    with torch.inference_mode():
        for sentence, actor, patient in sentences:
            for question in questions:
                row = {"sentence": sentence, "question": question,
                       "actor": actor, "patient": patient}
                inputs = tokenizer(core.make_prompt(tokenizer, row), return_tensors="pt")
                output = model(**inputs, use_cache=False)
                answer = tokenizer.decode([int(output.logits[0, -1].argmax())]).strip()
                results.append({"sentence": sentence, "question": question,
                                "answer": answer, "correct": answer == patient})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
