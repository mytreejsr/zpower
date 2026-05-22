"""
ZPower Example 4 — HuggingFace model augmentation
Requires: pip install zpower[hf]
"""
import zpower as zp

# Uncomment to run (requires transformers + torch)
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model     = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
#
# # One line — attach ZPower to any HuggingFace model
# zp_model = zp.compat.augment(
#     model,
#     stabilize    = True,
#     weight_vault = True,
#     auto_heal    = True,
# )
#
# # generate() passthrough — unchanged
# inputs = tokenizer("Hello world", return_tensors="pt")
# output = zp_model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(output[0]))
#
# # Full health report
# print(zp_model.health_report())

print("See comments in this file — requires pip install zpower[hf]")
