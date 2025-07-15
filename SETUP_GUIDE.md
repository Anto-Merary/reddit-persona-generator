# Reddit Persona Generator - Setup Guide

This guide provides step-by-step instructions for setting up and running the Reddit Persona Generator with quantized LLM and Flask API.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 4GB VRAM (recommended)
- **RAM**: Minimum 8GB system RAM
- **Storage**: At least 2GB free space for model files
- **CPU**: Modern multi-core processor

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: Compatible CUDA installation (for GPU acceleration)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/reddit-persona-generator.git
cd reddit-persona-generator
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Set Up Reddit API Credentials

1. **Create Reddit App**:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Select "script" as the application type
   - Fill in the required fields:
     - Name: `PersonaGenerator`
     - Description: `AI-powered Reddit persona analysis tool`
     - About URL: Can be left blank
     - Redirect URI: `http://localhost:8080`
   - Click "Create app"

2. **Get Credentials**:
   - Note down the **client ID** (appears under the app name)
   - Note down the **client secret** (appears as "secret")

3. **Configure Environment**:
   ```bash
   # Copy example environment file
   cp env_example.txt .env
   
   # Edit .env file with your credentials
   nano .env  # or use your preferred editor
   ```
   
   Add your credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=PersonaGenerator/1.0
   ```

## Running the Application

### 1. Start the Flask Server
```bash
python reddit_persona_generator.py
```

This will:
1. Initialize the quantized LLM model (may take 1-2 minutes)
2. Test with sample URLs
3. Start the Flask API server on `http://localhost:5000`

### 2. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Test the API
Run the test script to verify everything is working:
```bash
python test_api.py
```

## Usage Examples

### Web Interface
1. Navigate to `http://localhost:5000`
2. Enter a Reddit profile URL (e.g., `https://www.reddit.com/user/kojied/`)
3. Click "Generate Persona"
4. Wait for the analysis to complete

### API Endpoint
```bash
curl -X POST http://localhost:5000/generate_persona \
  -H "Content-Type: application/json" \
  -d '{"profile_url": "https://www.reddit.com/user/kojied/"}'
```

### Direct Python Usage
```python
from reddit_persona_generator import QuantizedLLMPersonaGenerator

# Initialize generator
generator = QuantizedLLMPersonaGenerator(
    reddit_client_id="your_client_id",
    reddit_client_secret="your_client_secret",
    reddit_user_agent="PersonaGenerator/1.0"
)

# Generate persona
result = generator.generate_persona_report("https://www.reddit.com/user/kojied/")
print(result)
```

## Testing

### Test URLs
The following URLs are included for testing:
- `https://www.reddit.com/user/kojied/`
- `https://www.reddit.com/user/Hungry-Move-6603/`

### Run Tests
```bash
# Test the Flask API
python test_api.py

# Test individual components
python test_persona_generator.py
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU-only mode by setting `device_map="cpu"` in the model loading code.

#### 2. Model Download Issues
```
OSError: Can't load tokenizer for 'facebook/opt-125m'
```
**Solution**: Check internet connection and try:
```bash
pip install --upgrade transformers
```

#### 3. Reddit API Rate Limits
```
Error: 429 Too Many Requests
```
**Solution**: Wait a few minutes and try again. Reddit API has rate limits.

#### 4. spaCy Model Not Found
```
OSError: Can't find model 'en_core_web_sm'
```
**Solution**:
```bash
python -m spacy download en_core_web_sm
```

#### 5. Flask Server Won't Start
```
Address already in use
```
**Solution**: Kill existing process or use different port:
```bash
# Kill existing process
pkill -f "python reddit_persona_generator.py"

# Or modify port in code
app.run(host='0.0.0.0', port=5001, debug=False)
```

### Performance Optimization

#### 1. GPU Memory Optimization
- Use smaller batch sizes
- Enable gradient checkpointing
- Use mixed precision training

#### 2. CPU-Only Mode
If you don't have a GPU, modify the model loading:
```python
# In _setup_quantized_llm method
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
```

#### 3. Model Caching
The model files are cached in `~/.cache/huggingface/` after first download.

## File Structure

```
reddit-persona-generator/
├── reddit_persona_generator.py    # Main application
├── test_api.py                    # API test script
├── test_persona_generator.py      # Component tests
├── requirements.txt               # Dependencies
├── env_example.txt               # Environment template
├── README.md                     # Documentation
├── SETUP_GUIDE.md               # This file
├── output/                      # Generated personas
└── .env                        # Your credentials (create this)
```

## API Endpoints

### GET /
- **Description**: Web interface
- **Response**: HTML page

### GET /health
- **Description**: Health check
- **Response**: JSON with status information

### POST /generate_persona
- **Description**: Generate persona from Reddit URL
- **Request Body**: `{"profile_url": "https://www.reddit.com/user/username/"}`
- **Response**: JSON with persona data and statistics

## Output Format

Generated personas follow this structure:
```
User Persona for [username]:
Generated on: 2024-01-15 14:30:25
Profile URL: https://www.reddit.com/user/[username]/

============================================================

- Age: [LLM analysis]
  (Citations: [URLs])

- Interests: [LLM analysis]
  (Citations: [URLs])

- Behavior: [LLM analysis]
  (Citations: [URLs])

- Goals: [LLM analysis]
  (Citations: [URLs])

============================================================
USER STATISTICS:
[Account information and NLP analysis]
```

## Development

### Code Structure
- `QuantizedLLMPersonaGenerator`: Main class
- `initialize_generator()`: Setup function
- Flask routes: `/`, `/health`, `/generate_persona`
- Helper methods: Text cleaning, NLP analysis, LLM querying

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## Support

For issues and questions:
1. Check this setup guide
2. Review the main README.md
3. Check GitHub issues
4. Create a new issue with detailed information

## License

This project is for educational and research purposes. Please respect Reddit's terms of service and user privacy. 