"""
API Testing Script for Indian House Price Prediction

This script tests the Flask API endpoints with sample requests.

Usage:
    python test_api.py
"""

import requests
import json


# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test the health check endpoint."""
    print("\n" + "=" * 70)
    print("Testing Health Check Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Health check passed!")
            return True
        else:
            print("✗ Health check failed!")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        print("Make sure the API is running on http://localhost:5000")
        return False


def test_predict_2bhk():
    """Test prediction for a 2 BHK property."""
    print("\n" + "=" * 70)
    print("Test 1: Predict 2 BHK in Whitefield")
    print("=" * 70)
    
    payload = {
        "bhk": 2,
        "area_sqft": 1200,
        "city": "Bangalore",
        "locality": "Whitefield",
        "age": 5,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "parking": 1,
        "lift": 1
    }
    
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Prediction successful!")
            print(f"   Predicted Price: {data['formatted']}")
            print(f"   Range: {data['lower_formatted']} - {data['upper_formatted']}")
            print(f"   Model: {data['model']}")
            return True
        else:
            print(f"✗ Prediction failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_3bhk():
    """Test prediction for a 3 BHK property."""
    print("\n" + "=" * 70)
    print("Test 2: Predict 3 BHK in Indiranagar")
    print("=" * 70)
    
    payload = {
        "bhk": 3,
        "area_sqft": 1800,
        "city": "Bangalore",
        "locality": "Indiranagar",
        "age": 2,
        "floor": 2,
        "bathrooms": 3,
        "balconies": 2,
        "parking": 2,
        "lift": 1
    }
    
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Prediction successful!")
            print(f"   Predicted Price: {data['formatted']}")
            print(f"   Range: {data['lower_formatted']} - {data['upper_formatted']}")
            return True
        else:
            print(f"✗ Prediction failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_1bhk():
    """Test prediction for a 1 BHK property."""
    print("\n" + "=" * 70)
    print("Test 3: Predict 1 BHK in Marathahalli")
    print("=" * 70)
    
    payload = {
        "bhk": 1,
        "area_sqft": 850,
        "city": "Bangalore",
        "locality": "Marathahalli",
        "age": 10,
        "floor": 4,
        "bathrooms": 1,
        "balconies": 1,
        "parking": 0,
        "lift": 1
    }
    
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Prediction successful!")
            print(f"   Predicted Price: {data['formatted']}")
            return True
        else:
            print(f"✗ Prediction failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False


def test_invalid_input():
    """Test API with invalid input."""
    print("\n" + "=" * 70)
    print("Test 4: Invalid Input (Missing Fields)")
    print("=" * 70)
    
    payload = {
        "bhk": 2,
        "area_sqft": 1200
        # Missing other required fields
    }
    
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 400:
            print(f"\n✓ API correctly rejected invalid input!")
            return True
        else:
            print(f"✗ API should reject invalid input with 400 status!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False


def test_curl_example():
    """Print cURL example."""
    print("\n" + "=" * 70)
    print("cURL Example")
    print("=" * 70)
    
    curl_command = """
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "bhk": 2,
    "area_sqft": 1200,
    "city": "Bangalore",
    "locality": "Whitefield",
    "age": 5,
    "floor": 3,
    "bathrooms": 2,
    "balconies": 1,
    "parking": 1,
    "lift": 1
  }'
"""
    
    print(curl_command)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("API Testing Suite - Indian House Price Prediction")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("2 BHK Prediction", test_predict_2bhk()))
    results.append(("3 BHK Prediction", test_predict_3bhk()))
    results.append(("1 BHK Prediction", test_predict_1bhk()))
    results.append(("Invalid Input", test_invalid_input()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    # Print cURL example
    test_curl_example()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
