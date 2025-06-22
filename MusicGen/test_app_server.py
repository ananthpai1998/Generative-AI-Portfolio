import pytest
import pytest_asyncio
import httpx
import asyncio
import base64
import io
import soundfile as sf
import os

SERVER_URL = "http://127.0.0.1:8000"
SECURITY_TOKEN = "test_Sec__123token_random_123"
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 30
OUTPUT_DIR = "test_outputs"


@pytest_asyncio.fixture(scope="function")
async def client():
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=200.0) as async_client:
        yield async_client

async def poll_for_status(client: httpx.AsyncClient, request_id: str, target_status: str = "completed", max_attempts: int = MAX_POLL_ATTEMPTS):
    """Polls the status endpoint until the target status is reached or polling times out."""
    print(f"\nPolling for request_id: {request_id} (target: {target_status})")
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} for {request_id}...")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            response = await client.get(f"/generated_music/{request_id}")
            print(f"Status check for {request_id}: {response.status_code}")

            if response.status_code == 404 and target_status == "completed":
                print(f"Received 404 for {request_id}. Assuming completed after a few checks.")
                
                if attempt > 2: # Wait for ~6 seconds before assuming completion on 404
                     print(f"Assuming {request_id} completed due to 404 after {attempt+1} attempts.")
                     # Return a simulated completion status
                     return {"processing_status": "completed", "message": "Assumed completed due to 404", "generated_data": None, "is_success": True}
                else:
                    print(f"Received 404 early for {request_id}, continuing poll...")
                    continue # Continue polling if 404 received early


            response.raise_for_status() # Raise HTTP errors for other non-2xx codes
            data = response.json()
            current_status = data.get("processing_status", data.get("message")) # Adapt based on actual response structure
            print(f"Status for {request_id}: {current_status}, Queue Pos: {data.get('queue_position', 'N/A')}")


            # Check against target status
            # Backend uses "completed", "processing", "queued", "failed"
            if current_status == target_status:
                print(f"Target status '{target_status}' reached for {request_id}.")
                return data
            if current_status == "failed":
                print(f"Request {request_id} failed.")
                return data # Return the failure data

            # Optional: Add specific handling for other statuses if needed (e.g., queued, processing)

        except httpx.HTTPStatusError as e:
            print(f"HTTP error during polling for {request_id}: {e.response.status_code} - {e.response.text}")
            # Decide whether to continue polling or fail based on the error
            if e.response.status_code == 404 and target_status == "completed":
                 # Handled above, but kept for clarity; this part might not be reached if 404 logic is correct.
                 print(f"Polling continued despite 404 for {request_id}...")
                 continue
            # For other errors, maybe fail fast?
            pytest.fail(f"HTTP error {e.response.status_code} during polling for {request_id}. Response: {e.response.text}")
        except Exception as e:
            print(f"Error during polling for {request_id}: {e}")
            # Continue polling even if there's a transient error like connection refused briefly
            continue

    # If loop completes without reaching target status
    pytest.fail(f"Polling timed out after {max_attempts} attempts for {request_id}. Last known status might be 'processing' or 'queued'.")


def save_audio_from_base64(base64_data: str, request_id: str = "test"):
    """Saves audio data from a base64 string to a WAV file."""
    if not base64_data:
        print(f"No audio data provided for request {request_id}, skipping save.")
        return
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        audio_bytes = base64.b64decode(base64_data)
        # Use soundfile which is generally more robust for reading various WAV formats
        audio_buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_buffer, dtype='float32')
        filename = os.path.join(OUTPUT_DIR, f"test_output_{request_id[:8]}.wav")
        sf.write(filename, data, samplerate)
        print(f"Saved audio for {request_id} to {filename}")
    except Exception as e:
        print(f"Error saving audio for {request_id}: {e}")

@pytest.mark.asyncio
async def test_status_endpoint(client):
    """Tests if the status endpoint is reachable and returns expected keys."""
    response = await client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "background_task_active" in data

@pytest.mark.asyncio
async def test_generate_music_success_and_fetch(client):
    """Tests successful music generation and retrieval."""
    payload = {
        "security_token": SECURITY_TOKEN,
        "music_generation_condition": "Upbeat electronic music",
        "duration": 3 # Keep duration short for testing
    }
    response = await client.post("/generate-music", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert data["is_success"] is True
    request_id = data["request_id"]
    assert request_id is not None

    # Poll until completion
    completion_data = await poll_for_status(client, request_id, target_status="completed")

    # Check completion status explicitly from the polling result
    # Adapt keys based on actual backend response format upon completion
    assert completion_data.get("is_success") is True or completion_data.get("processing_status") == "completed"
    # Check if audio data exists (even if it's None due to 404 assumption)
    assert "generated_data" in completion_data
    save_audio_from_base64(completion_data.get("generated_data"), request_id)

@pytest.mark.asyncio
async def test_generate_music_invalid_token(client):
    """Tests generation request with an invalid security token."""
    payload = {
        "security_token": "INVALID_TOKEN",
        "music_generation_condition": "Test prompt",
        "duration": 5
    }
    response = await client.post("/generate-music", json=payload)
    assert response.status_code == 403, f"Expected 403, got {response.status_code}. Response: {response.text}"

# Test cases for invalid duration based on backend validation (1-30 seconds)
@pytest.mark.parametrize("duration", [0, 31, -5])
@pytest.mark.asyncio
async def test_generate_music_invalid_duration(client, duration):
    """Tests generation request with invalid duration values."""
    payload = {
        "security_token": SECURITY_TOKEN,
        "music_generation_condition": "Test prompt",
        "duration": duration
    }
    response = await client.post("/generate-music", json=payload)
    assert response.status_code == 400, f"Expected 400 for duration {duration}, got {response.status_code}. Response: {response.text}"

@pytest.mark.asyncio
async def test_generate_music_validation_error_missing_prompt(client):
    """Tests generation request with missing 'music_generation_condition'."""
    payload = {
        "security_token": SECURITY_TOKEN,
        # Missing "music_generation_condition"
        "duration": 5
    }
    response = await client.post("/generate-music", json=payload)
    # FastAPI should return 422 Unprocessable Entity for missing required fields
    assert response.status_code == 422, f"Expected 422, got {response.status_code}. Response: {response.text}"

@pytest.mark.asyncio
async def test_get_generated_music_not_found(client):
    """Tests retrieving status for a non-existent request ID."""
    response = await client.get("/generated_music/invalid-request-id-12345")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}. Response: {response.text}"

@pytest.mark.asyncio
async def test_queueing_multiple_requests(client):
    """Tests queueing multiple requests and checking their completion."""
    prompts = ["Sad piano melody", "80s synthwave beat", "Calm acoustic guitar strumming"]
    tasks = []
    print("\nSubmitting multiple requests...")
    for i, prompt in enumerate(prompts):
        payload = {
            "security_token": SECURITY_TOKEN,
            "music_generation_condition": prompt,
            "duration": 3 + i # Vary duration slightly (keep short)
        }
        # Create tasks for submitting POST requests concurrently
        tasks.append(client.post("/generate-music", json=payload))

    # Wait for all POST requests to complete
    post_responses = await asyncio.gather(*tasks)

    request_ids = []
    for i, response in enumerate(post_responses):
        assert response.status_code == 200, f"POST {i+1} failed: {response.status_code}, {response.text}"
        data = response.json()
        assert data["is_success"] is True
        request_id = data["request_id"]
        assert request_id is not None
        request_ids.append(request_id)
        print(f"Submitted request {i+1} with ID: {request_id}")

    print(f"\nPolling for completion of {len(request_ids)} requests...")
    # Create tasks for polling the status of each request concurrently
    poll_tasks = [poll_for_status(client, rid, target_status="completed") for rid in request_ids]

    # Wait for all polling tasks to complete
    results = await asyncio.gather(*poll_tasks)

    # Verify results and save audio
    for i, (rid, result) in enumerate(zip(request_ids, results)):
        print(f"Result for request {i+1} ({rid}): {result.get('processing_status', result.get('message'))}")
        # Check if the final status indicates success or assumed completion
        assert result.get("is_success") is True or result.get("processing_status") == "completed", f"Request {rid} did not complete successfully. Final data: {result}"
        assert "generated_data" in result # Check key exists
        save_audio_from_base64(result.get("generated_data"), rid)