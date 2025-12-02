# launch_experiment.py
import subprocess
import time
import asyncio
import sys

def start_sglang_server():
    """Start SGLang server (simulated)"""
    print("Starting SGLang server...")
    # In real setup, you would run: sglang-launch --model meta-llama/Llama-2-13b --port 30000
    print("Note: In production, manually start SGLang server first")
    print("Command: sglang-launch --model meta-llama/Llama-2-13b --port 30000")
    time.sleep(2)
    return True

async def run_experiment():
    """Run the complete experiment"""
    print("="*70)
    print("PROMPTPEEK EXPERIMENT - Multi-tenant LLM Serving Attack")
    print("="*70)
    
    # Check if SGLang server is running
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:30000/health") as resp:
                if resp.status != 200:
                    print("ERROR: SGLang server not running on port 30000")
                    print("Please start it first:")
                    print("  sglang-launch --model meta-llama/Llama-2-13b --port 30000")
                    return
    except:
        print("ERROR: Cannot connect to SGLang server")
        print("Make sure it's running on http://localhost:30000")
        return
    
    print("\n1. Starting victim simulation...")
    victim_proc = subprocess.Popen(
        [sys.executable, "victim_simulator.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for victim to establish some KV cache
    print("Waiting 10 seconds for victim to send initial requests...")
    time.sleep(10)
    
    print("\n2. Starting PromptPeek attacker...")
    attacker_proc = subprocess.Popen(
        [sys.executable, "promptpeek_attacker.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for attack to complete
    print("\n3. Running attack for 60 seconds...")
    time.sleep(60)
    
    # Terminate processes
    print("\n4. Terminating experiment...")
    victim_proc.terminate()
    attacker_proc.terminate()
    
    # Get outputs
    victim_out, victim_err = victim_proc.communicate()
    attacker_out, attacker_err = attacker_proc.communicate()
    
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    
    print("\nVictim Output (last 10 lines):")
    for line in victim_out.decode().split('\n')[-10:]:
        print(line)
    
    print("\nAttacker Output (last 20 lines):")
    for line in attacker_out.decode().split('\n')[-20:]:
        print(line)
    
    print("\n" + "="*70)
    print("Experiment completed!")
    print("Check 'reconstructed_templates.json' for attack results")

if __name__ == "__main__":
    asyncio.run(run_experiment())