# NER Streamlit Frontend

A lightweight Streamlit frontend for interacting with a deployed Named Entity Recognition (NER) API.

## Features

- **API-based NER inference** with real-time entity extraction
- **KPI summary** - Total entities, unique labels, average confidence
- **Color-coded highlighting** with intelligent overlap resolution
- **Interactive entity table** with built-in search

## Requirements

- Python 3.9+
- NER API endpoint with the following interface:
```json
POST /predict
{
  "text": "Your input text here"
}
```

Response:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "entities": [
      {
        "label": "PER",
        "text": "Entity text",
        "score": 0.95,
        "start": 10,
        "end": 20
      }
    ]
  }
}
```

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- This is a frontend client only - does not host the NER model
- Requires an active NER API endpoint to function