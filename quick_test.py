# quick_test.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_all_features():
    print("🧪 Testing All Application Features")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"   ✅ Status: {health['status']}")
    print(f"   ✅ Model: {'Loaded' if health['model_loaded'] else 'Not Loaded'}")
    print(f"   ✅ Twitter: {'Connected' if health['twitter_connected'] else 'Not Connected'}")
    
    # Test 2: Single Text Analysis
    print("\n2. Testing Single Text Analysis...")
    texts = [
        "I absolutely love this product! It's fantastic!",
        "This is the worst experience ever.",
        "It's okay, nothing special.",
        "I'm extremely angry about this service!",
        "This is absolutely wonderful and amazing!"
    ]
    
    for i, text in enumerate(texts, 1):
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        result = response.json()
        print(f"   {i}. '{text}'")
        print(f"      → {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    # Test 3: Sentiment Labels
    print("\n3. Testing Sentiment Labels...")
    labels = requests.get(f"{BASE_URL}/sentiment-labels").json()
    print("   Available sentiment classes:")
    for class_id, info in labels['mapping'].items():
        print(f"      {class_id}: {info['label']} {info['emoji']}")
    
    print("\n🎉 All Tests Completed Successfully!")
    print("\n🌐 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/api/docs")

if __name__ == "__main__":
    test_all_features()