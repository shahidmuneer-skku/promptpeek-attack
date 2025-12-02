# victim_simulator.py
import asyncio
import aiohttp
import json
import time
import random
from typing import List
from datasets import load_dataset

# ==================== CONFIGURATION ====================
SGLANG_URL = "http://localhost:30000/generate"
DATASET_NAME = "fka/awesome-chatgpt-prompts"
NUM_USERS = 1  # Single victim user as per paper's evaluation
REQUESTS_PER_USER = 40  # As mentioned in paper
REQUEST_FREQUENCY = 0.004  # requests per second (0.004 = 40 requests per 3 hours)

# ==================== PROMPT TEMPLATE SETUP ====================
class VictimPromptService:
    def __init__(self):
        # Load awesome-chatgpt-prompts dataset
        self.dataset = load_dataset(DATASET_NAME, split="train")
        self.templates = self._extract_templates()
        
    def _extract_templates(self):
        """Extract role-based templates from dataset"""
        templates = []
        for item in self.dataset:
            prompt = item.get("prompt", "")
          
            # Extract the role definition part (template)
            if "I want you to act as" in prompt:
                # Split at "My first request is" or similar endings
                if "My first request is" in prompt:
                    template = prompt.split("My first request is")[0].strip()
                elif "My first sentence is" in prompt:
                    template = prompt.split("My first sentence is")[0].strip()
                else:
                    template = prompt
                templates.append(template)
       
        return templates
    
    def get_random_template(self) -> str:
        """Get a random template from dataset"""
        return random.choice(self.templates)
    
    def create_user_input(self, template: str) -> str:
        """Create a realistic user input for the given template"""
        # Extract role from template
        if "act as" in template.lower():
            role_start = template.lower().find("act as") + 6
            role_end = template.find(".", role_start)
            if role_end == -1:
                role_end = template.find(",", role_start)
            if role_end == -1:
                role_end = len(template)
            role = template[role_start:role_end].strip()
            
            # Generate context-appropriate input based on role
            inputs = {
                "linux terminal": "Please list all files in the current directory and their permissions.",
                "english translator": "Translate 'Hello, how are you?' to French.",
                "travel guide": "What are the top 3 attractions in Paris?",
                "storyteller": "Tell me a short story about a brave knight.",
                "motivational coach": "I'm feeling unmotivated to exercise. What should I do?",
                "debater": "Argue for the benefits of renewable energy.",
                "poet": "Write a haiku about the changing seasons.",
                "interviewer": "Ask me 5 common interview questions for a software engineer position.",
                "stand-up comedian": "Tell me a joke about programming.",
                "financial advisor": "How should I budget $5000 monthly income?",
            }
            
            for key, value in inputs.items():
                if key in role.lower():
                    return value
            
            # Default input
            return f"Please provide information about {role}."
        
        return "Please proceed with the task."
    
    def get_full_prompt(self, template: str, user_input: str) -> str:
        """Combine template with user input"""
        if "My first request is" in template:
            return f"{template} My first request is \"{user_input}\""
        elif "My first sentence is" in template:
            return f"{template} My first sentence is \"{user_input}\""
        else:
            return f"{template} {user_input}"

# ==================== SGLANG CLIENT ====================
class SGLangClient:
    def __init__(self, url: str):
        self.url = url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async def send_request(self, prompt: str, max_tokens: int = 128) -> str:
        """Send a single request to SGLang server"""
      
        payload = {
            "text": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        
        try:
            async with self.session.post(self.url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("text", "")
                else:
                    print(f"Error: {response.status}")
                    return ""
        except Exception as e:
            print(f"Request failed: {e}")
            return ""

# ==================== VICTIM USER ====================
class VictimUser:
    def __init__(self, user_id: int, client: SGLangClient, prompt_service: VictimPromptService):
        self.user_id = user_id
        self.client = client
        self.prompt_service = prompt_service
        self.template = None
        self.user_input = None
        self.full_prompt = None
        
    def prepare_prompt(self):
        """Prepare a new prompt for this user"""
        self.template = self.prompt_service.get_random_template()
        self.user_input = self.prompt_service.create_user_input(self.template)
        self.full_prompt = self.prompt_service.get_full_prompt(self.template, self.user_input)
        return self.full_prompt
    
    async def send_query(self) -> str:
        """Send the prepared query to SGLang server"""
        if not self.full_prompt:
            self.prepare_prompt()
            
        print(f"[Victim {self.user_id}] Sending prompt: {self.full_prompt[:80]}...")
        response = await self.client.send_request(self.full_prompt)
        print(f"[Victim {self.user_id}] Response length: {len(response)} chars")
        return response

# ==================== MAIN SIMULATION ====================
async def simulate_victim_behavior():
    """Simulate victim users sending queries as described in the paper"""
    
    # Initialize services
    prompt_service = VictimPromptService()
    
    async with SGLangClient(SGLANG_URL) as client:
        # Create victim users
        users = [VictimUser(i, client, prompt_service) for i in range(NUM_USERS)]
        
        print(f"Starting victim simulation with {NUM_USERS} users")
        print(f"Each user will send {REQUESTS_PER_USER} requests")
        print(f"Request frequency: {REQUEST_FREQUENCY} requests/sec per user")
        
        # Track requests per user
        user_requests = {user.user_id: 0 for user in users}
        
        # Continuous simulation (simplified - in paper it's 3 hours)
        start_time = time.time()
        duration = 10800  # 3 hours in seconds
        
        while time.time() - start_time < duration:
            for user in users:
                if user_requests[user.user_id] < REQUESTS_PER_USER:
                    # Send request
                    await user.send_query()
                    user_requests[user.user_id] += 1
                    
                    # Prepare new prompt for next request
                    user.prepare_prompt()
                    
                    # Log progress
                    total_requests = sum(user_requests.values())
                    print(f"\n[Progress] Total requests: {total_requests}/{NUM_USERS * REQUESTS_PER_USER}")
                    print(f"          Time elapsed: {time.time() - start_time:.1f}s / {duration}s")
                    
                # Sleep to maintain frequency
                await asyncio.sleep(1 / REQUEST_FREQUENCY)
                
            # Check if all requests are done
            if all(req >= REQUESTS_PER_USER for req in user_requests.values()):
                break
        
        print("\n" + "="*50)
        print("Victim simulation completed!")
        print(f"Total requests sent: {sum(user_requests.values())}")
        print(f"Simulation time: {time.time() - start_time:.1f} seconds")

async def quick_simulation():
    """Quick simulation for testing - sends 40 requests rapidly"""
    prompt_service = VictimPromptService()
    
    async with SGLangClient(SGLANG_URL) as client:
        user = VictimUser(0, client, prompt_service)
        
        print(f"Starting quick simulation - sending {REQUESTS_PER_USER} requests")
        
        for i in range(REQUESTS_PER_USER):
            response = await user.send_query()
            print(f"Request {i+1}/{REQUESTS_PER_USER} completed")
            
            # Prepare for next request
            user.prepare_prompt()
            
            # Small delay to simulate real usage
            await asyncio.sleep(0.5)
        
        print("\nQuick simulation completed!")

if __name__ == "__main__":
    # For quick testing, use quick_simulation
    # For full 3-hour simulation, use simulate_victim_behavior
    
    # asyncio.run(simulate_victim_behavior())  # Full simulation
    asyncio.run(quick_simulation())  # Quick test