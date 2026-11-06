from transformers import AutoTokenizer

for mid in ["CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"]:
    tok = AutoTokenizer.from_pretrained(mid)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    print(mid, "-> templated prompt tokens:", ids["input_ids"].shape[1])
