from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
# Add HTMLResponse import
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import base64
import scipy
import io
import uuid
import pandas as pd
import os # Import os to construct file path safely


FRONTEND_FILE_PATH = '/app/front_end.HTML'


class MusicQueueDF:
    def __init__(self):
        self.df = pd.DataFrame(columns=["request_id",
                                        "type",
                                        "processing_status",
                                        "queue_position",
                                        "generated_data",
                                        "music_generation_condition",
                                        "duration",
                                        "timestamp"])

    def add_to_queue(self, request_id, type, music_generation_condition, duration):
        try:
            new_row = {"request_id": request_id,
                       "type": type,
                       "processing_status": "queued",
                       "queue_position": len(self.df), # Simple queue position
                       "generated_data": "",
                       "music_generation_condition": music_generation_condition,
                       "duration": duration,
                       "timestamp": pd.NaT} # Use NaT for missing timestamps
            # Use pd.concat instead of append
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self.update_queue_positions() # Update positions after adding

        except Exception as e:
            print(f"Error adding to queue: {e}")

    def update_processing_status(self, request_id, processing_status, generated_data):
        try:
            idx = self.df[self.df["request_id"] == request_id].index
            if not idx.empty:
                self.df.loc[idx, "processing_status"] = processing_status
                self.df.loc[idx, "generated_data"] = generated_data
                if processing_status == "completed":
                    self.df.loc[idx, "timestamp"] = pd.Timestamp.now()
                self.delete_expired_requests() 

        except Exception as e:
            print(f"Error updating processing status for {request_id}: {e}")

    def update_queue_positions(self):
        """ Recalculates queue positions for all queued items. """
        try:
            queued_mask = self.df["processing_status"] == "queued"
            # Assign positions based on current order for queued items
            self.df.loc[queued_mask, "queue_position"] = range(queued_mask.sum())
        except Exception as e:
            print(f"Error updating queue positions: {e}")

    def get_queue_position(self, request_id):
        try:
            row = self.df[self.df["request_id"] == request_id]
            if not row.empty and row["processing_status"].iloc[0] == 'queued':
                return row["queue_position"].iloc[0]
            return None # Return None if not found or not queued
        except Exception as e:
            print(f"Error getting queue position for {request_id}: {e}")
            return None


    def delete_from_queue(self, request_id):
        try:
            initial_len = len(self.df)
            self.df = self.df[self.df["request_id"] != request_id].copy() # Use copy to avoid SettingWithCopyWarning
            if len(self.df) < initial_len:
                 print(f"Deleted request ID: {request_id}")
                 self.update_queue_positions() # Update positions after deletion
            # self.delete_expired_requests()
        except Exception as e:
            print(f"Error deleting from queue {request_id}: {e}")

    def delete_expired_requests(self):
        # This can be called periodically if needed, e.g., via a background task
        try:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors='coerce')
            expiration_time = pd.Timestamp.now() - pd.Timedelta(minutes=1) # 1 minute expiry
            expired_mask = (self.df["processing_status"] == "completed") & (self.df["timestamp"] < expiration_time)

            expired_ids = self.df.loc[expired_mask, "request_id"].tolist()
            if expired_ids:
                 print(f"Deleting expired completed requests: {expired_ids}")
                 self.df = self.df[~expired_mask].copy() # Use copy
                 self.update_queue_positions() # Update positions if any deletion occurred

        except Exception as e:
            print(f"Error deleting expired requests: {e}")


global BackgroundTask_Activity
BackgroundTask_Activity = False

requests_queue = MusicQueueDF()

# --- Model Loading --- 
print("Loading model...")
try:
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load model. Error: {e}")
    # Optionally exit or prevent app startup if model load fails
    # exit()
# ---------------------


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicInput(BaseModel):
    security_token: str
    music_generation_condition: str
    duration: int

# --- NEW ENDPOINT TO SERVE FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the front_end.HTML file."""
    print(f"Attempting to serve frontend file from: {FRONTEND_FILE_PATH}")
    try:
        if not os.path.exists(FRONTEND_FILE_PATH):
             print(f"Error: Frontend file not found at {FRONTEND_FILE_PATH}")
             raise HTTPException(status_code=404, detail="Frontend HTML file not found.")

        with open(FRONTEND_FILE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        print(f"Error serving frontend: {e}")
        # Avoid raising HTTPException here for generic errors if possible,
        # return a fallback error message instead.
        return HTMLResponse(content="<html><body><h1>Internal Server Error</h1><p>Sorry, could not load the frontend.</p></body></html>", status_code=500)
# ---------------------------------------

@app.get("/status")
async def check_status():
    return {"status": "ok", "background_task_active": BackgroundTask_Activity}

def music_generation_model(duration, music_generation_condition, request_id):
    """Generates music using the preloaded model."""
    try:
        print(f"[{request_id}] Generating music: '{music_generation_condition}', duration: {duration}s")
        # Musicgen duration calculation might differ, check model docs. Using 50Hz as approximation.
        # duration_tokens = duration * 50 # Example, adjust based on model specifics
        # Check model docs for max_new_tokens relationship to seconds
        
        max_tokens = int(duration * model.config.audio_encoder.frame_rate) # Approx tokens based on frame rate
        max_supported_tokens = model.config.max_length # e.g., 1500 for small

        if duration > 30 : # Safety cap based on typical model limits
             print(f"Warning: Requested duration {duration}s exceeds typical max (30s). Capping generation.")
             duration = 30
             max_tokens = max_supported_tokens

        print(f"[{request_id}] Calculated max_tokens: {max_tokens}")


        inputs = processor(
                    text=[music_generation_condition],
                    padding=True,
                    return_tensors="pt",
                ).to(device)

        # Generate audio - use max_new_tokens if applicable, or rely on duration logic
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_tokens)
        audio_values = audio_values.cpu() # Move to CPU before scipy/numpy

        sampling_rate = model.config.audio_encoder.sampling_rate
        print(f"[{request_id}] Sampling rate: {sampling_rate}")
        # Ensure audio_values has the expected shape before processing
        # Shape typically [batch_size, num_channels, num_samples]
        if audio_values.ndim == 3 and audio_values.shape[0] == 1 and audio_values.shape[1] == 1:
             audio_data = audio_values[0, 0].numpy() # Extract the raw audio waveform
             print(f"[{request_id}] Generated audio data shape: {audio_data.shape}")
        else:
             print(f"[{request_id}] Error: Unexpected audio tensor shape: {audio_values.shape}")
             raise ValueError("Unexpected audio tensor shape from model")

        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, rate=sampling_rate, data=audio_data)
        wav_buffer.seek(0) # Rewind buffer to the beginning

        encoded_audio = base64.b64encode(wav_buffer.read()).decode("utf-8")
        print(f"[{request_id}] Music generated successfully.")
        requests_queue.update_processing_status(request_id, "completed", encoded_audio)

    except Exception as e:
        print(f"[{request_id}] Error generating music: {e}")
        requests_queue.update_processing_status(request_id, "failed", "")
        import traceback
        traceback.print_exc() # Print full error details to console


def generation_task():
    """Background task to process queued music generation requests."""
    global BackgroundTask_Activity
    print("Generation task started.")
    while True:
        # Get the next queued request (FIFO)
        queued_requests = requests_queue.df[requests_queue.df["processing_status"] == "queued"].sort_values(by="queue_position")

        if queued_requests.empty:
            print("No more queued requests. Stopping generation task.")
            BackgroundTask_Activity = False
            break # Exit the loop if queue is empty

        # Process the first request in the queue
        next_request = queued_requests.iloc[0]
        request_id = next_request["request_id"]
        music_condition = next_request["music_generation_condition"]
        duration = next_request["duration"]

        print(f"Processing request ID: {request_id} (Queue Pos: {next_request['queue_position']})")
        requests_queue.update_processing_status(request_id, "processing", "")

        # Run the generation model
        music_generation_model(duration, music_condition, request_id)

        

    print("Generation task finished.")
    BackgroundTask_Activity = False 


@app.get("/generated_music/{request_id}")
async def get_generated_music(request_id: str):
    """Gets the status or result of a music generation request."""
    try:
        request_row = requests_queue.df[requests_queue.df["request_id"] == request_id]

        if request_row.empty:
            print(f"Status check failed: Request ID {request_id} not found.")
            return JSONResponse(
                status_code=404,
                content={"message": "Request ID not found", "is_success": False, "generated_data": "", "queue_position": ""}
            )

        status_info = request_row.iloc[0]
        request_status = status_info["processing_status"]
        queue_pos = requests_queue.get_queue_position(request_id) # Use dedicated function

        print(f"Status check for {request_id}: {request_status}, Queue Pos: {queue_pos}")


        if request_status == "queued":
            return JSONResponse(content={"message": "Music generation request is queued",
                                         "is_success": False,
                                         "queue_position": queue_pos,
                                         "generated_data": ""})

        elif request_status == "processing":
            return JSONResponse(content={"message": "Music generation request is processing",
                                         "is_success": False,
                                         "generated_data": "",
                                         "queue_position": ""}) # No queue position once processing

        elif request_status == "failed":
            print(f"Request {request_id} failed. Deleting from queue.")
            requests_queue.delete_from_queue(request_id) # Delete failed request
            return JSONResponse(content={"message": "Music generation request failed",
                                         "is_success": False,
                                         "generated_data": "",
                                         "queue_position": ""})

        elif request_status == "completed":
            print(f"Request {request_id} completed. Retrieving data and deleting.")
            encoded_data = status_info["generated_data"]
            requests_queue.delete_from_queue(request_id) # Delete completed request after retrieval
            return JSONResponse(content={"message": "Music generation request completed",
                                         "is_success": True,
                                         "generated_data": encoded_data,
                                         "queue_position": ""})
        else:
            # Should not happen, but handle unknown status
             print(f"Error: Unknown status '{request_status}' for request ID {request_id}")
             return JSONResponse(status_code=500, content={"message": f"Internal error: Unknown status '{request_status}'", "is_success": False})


    except Exception as e:
        print(f"Error getting generated music for {request_id}: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Internal server error retrieving request status", "is_success": False})


@app.api_route("/generate-music", methods=["POST"]) # Only allow POST for generation
async def generate_music(request: Request, background_tasks: BackgroundTasks, input_data: MusicInput):
    """Adds a music generation request to the queue."""
    global BackgroundTask_Activity

    # Generate request ID within the endpoint
    request_id = str(uuid.uuid4())
    print(f"Received generation request. Assigning ID: {request_id}")

    try:
        # --- Input Validation ---
        if not input_data:
            raise HTTPException(status_code=400, detail="Input data is required") # Should be caught by FastAPI if model is correct

        if input_data.security_token != "test_Sec__123token_random_123": # Use proper auth in real app
            raise HTTPException(status_code=403, detail="Invalid security token")

        if not (1 <= input_data.duration <= 30): # Duration check
            raise HTTPException(status_code=400, detail="Duration must be between 1 and 30 seconds")

        if not input_data.music_generation_condition: # Prompt check
            raise HTTPException(status_code=400, detail="Music generation condition is required")
        # ------------------------

        print(f"[{request_id}] Condition: '{input_data.music_generation_condition}', Duration: {input_data.duration}s")

        # Add validated request to the queue
        requests_queue.add_to_queue(request_id, type="music", music_generation_condition=input_data.music_generation_condition, duration=input_data.duration)
        print(f"[{request_id}] Added to queue. Current BackgroundTask_Activity: {BackgroundTask_Activity}")

        # Start background task only if it's not already running
        if not BackgroundTask_Activity:
            print("Starting background generation task.")
            BackgroundTask_Activity = True
            background_tasks.add_task(generation_task)
        else:
            print("Background task already active.")

        return JSONResponse(content={
            "message": "Music queued successfully!",
            "request_id": request_id,
            "is_success": True
        })

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions for FastAPI to handle standard error responses
         raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"[{request_id}] Unexpected error during submission: {e}")
        import traceback
        traceback.print_exc()
        # Return a generic 500 error for unexpected issues
        raise HTTPException(status_code=500, detail="Internal server error processing request.")


if __name__ == "__main__":
    print("Starting Uvicorn server...")
    # Consider adding reload=True for development, but not for production
    uvicorn.run(app, host="0.0.0.0", port=8000)