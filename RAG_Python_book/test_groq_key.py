import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_key():
    """Test if the Groq API key is valid and working."""
    api_key = os.getenv("Groq_public_Key")
    
    if not api_key:
        print("❌ ERROR: Groq_public_Key not found in .env file")
        return False
    
    print(f"✓ API Key found: {api_key[:20]}...")
    print("\n🔍 Testing API connection...")
    
    try:
        client = Groq(api_key=api_key)
        
        # Simple test request
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Say 'Hello, API is working!' in one sentence."}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ SUCCESS! API is working!")
        print(f"Response: {result}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        print("Possible issues:")
        print("  1. Invalid or expired API key")
        print("  2. Network connection problem")
        print("  3. Groq service is down")
        print("\nGet a new API key at: https://console.groq.com/keys\n")
        return False

if __name__ == "__main__":
    test_groq_key()
