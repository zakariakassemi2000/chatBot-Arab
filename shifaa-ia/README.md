# SHIFAA IA — AI Hospital Diagnostic System

AI-powered hospital system focused on symptom-to-diagnosis inference for Moroccan healthcare context.

## Setup

1.  **Clone/Open the project**:
    ```bash
    cd shifaa-ia
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment**:
    Edit `.env` and add your `GEMINI_API_KEY` and `OPENAI_API_KEY`.

4.  **Run the application**:
    ```bash
    python main.py
    ```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

## Endpoints

### POST `/api/v1/diagnostic/`

Infers diagnosis from symptoms.

**Input Sample**:
```json
{
  "patient_age": 25,
  "gender": "male",
  "symptoms": ["fever", "cough", "fatigue"],
  "duration_days": 3,
  "medical_history": "Asthma"
}
```
