"""Quick API test for Groq - run this first to confirm everything works."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.config import GROQ_API_KEY, GROQ_MODEL
import base64
from groq import Groq

print(f"Model   : {GROQ_MODEL}")
print(f"API Key : {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-4:] if GROQ_API_KEY else 'NOT FOUND'}")

if not GROQ_API_KEY:
    print("\nERROR: GROQ_API_KEY not found in .env file!")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)

test_image = "data/1208_REDBUD_ST_CHARLOTTE_NC/2022-10__zzh9-nBI3xOwp_wIMWCRA.jpg"
print(f"\nTesting with: {test_image}")

with open(test_image, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

try:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "What type of building is this? One sentence."}
        ]}],
        temperature=0.0, max_tokens=100
    )
    print(f"\nAPI Response: {resp.choices[0].message.content.strip()}")
    print("\nSUCCESS - Groq API is working correctly!")
except Exception as e:
    print(f"\nFAILED: {e}")
    sys.exit(1)
