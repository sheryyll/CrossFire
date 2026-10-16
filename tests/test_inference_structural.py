import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import LLMClient, build_client_from_config


def test_unknown_provider_raises():
    try:
        LLMClient(provider="madeup", model="x")
        assert False, "should have raised"
    except ValueError as e:
        assert "Unknown provider" in str(e)
    print("test_unknown_provider_raises PASSED")


def test_missing_api_key_raises():
    os.environ.pop("GROQ_API_KEY", None)
    try:
        LLMClient(provider="groq", model="llama-3.3-70b-versatile")
        assert False, "should have raised"
    except ValueError as e:
        assert "GROQ_API_KEY" in str(e)
    print("test_missing_api_key_raises PASSED")


def test_client_builds_with_explicit_key():
    client = LLMClient(provider="groq", model="llama-3.3-70b-versatile", api_key="fake-key-for-test")
    assert client.provider == "groq"
    assert client.model == "llama-3.3-70b-versatile"
    print("test_client_builds_with_explicit_key PASSED")


def test_ollama_needs_no_key():
    # Local Ollama should build fine with no key at all.
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    assert client.provider == "ollama"
    print("test_ollama_needs_no_key PASSED")


def test_build_from_config_dict():
    config = {"provider": "openrouter", "model": "some/free-model", "api_key": "fake", "temperature": 0.5}
    client = build_client_from_config(config)
    assert client.temperature == 0.5
    print("test_build_from_config_dict PASSED")


if __name__ == "__main__":
    test_unknown_provider_raises()
    test_missing_api_key_raises()
    test_client_builds_with_explicit_key()
    test_ollama_needs_no_key()
    test_build_from_config_dict()
    print("\nAll inference structural tests passed.")
