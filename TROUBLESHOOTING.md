# Reddit Persona Generator - Troubleshooting Guide

## Common Issues and Solutions

### 1. Protobuf Error: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`

**This is the most common issue** and occurs due to version conflicts between protobuf and transformers libraries.

### 2. BitsAndBytesConfig Import Error: `ImportError: cannot import name 'BitsAndBytesConfig' from 'transformers'`

**This occurs when your transformers version is too old** (< 4.20.0). BitsAndBytesConfig was added in transformers 4.20+.

#### Solution:
```bash
# Update transformers to a newer version
pip install transformers>=4.24.0

# Or use the complete fix
pip install -r requirements_fixed.txt
```

#### Quick Fix (Recommended):
```bash
python fix_protobuf.py
```

#### Manual Fix:
```bash
# Uninstall conflicting versions
pip uninstall protobuf transformers tokenizers -y

# Install compatible versions
pip install protobuf==3.20.3
pip install transformers==4.24.0
pip install tokenizers==0.13.3

# Restart your Python environment
```

#### Alternative Fix:
```bash
# Use the fixed requirements file
pip install -r requirements_fixed.txt
```

### 3. GPU/CUDA Issues

**Error**: CUDA out of memory or GPU not detected

#### Solution:
- The app will automatically fallback to CPU mode
- Reduce model size by using a smaller model
- Close other GPU-intensive applications

### 4. spaCy Model Missing

**Error**: `OSError: [E050] Can't find model 'en_core_web_sm'`

#### Solution:
```bash
python -m spacy download en_core_web_sm
```

### 5. Reddit API Issues

**Error**: Reddit API authentication errors

#### Solution:
1. Create a `.env` file with your Reddit credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=PersonaGenerator/1.0
```

2. Get credentials from [Reddit App Preferences](https://www.reddit.com/prefs/apps)

### 6. Memory Issues

**Error**: Out of memory errors during model loading

#### Solution:
- Use CPU mode instead of GPU
- Close other applications
- Use a smaller model (change model_name in the code)

### 7. Flask Port Issues

**Error**: Port 5000 already in use

#### Solution:
Change the port in the main function:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Use port 5001 instead
```

## Testing Your Installation

### Quick Test
Run the test script to verify everything is working:
```bash
python test_imports.py
```

This will check:
- All required dependencies are installed
- Correct versions are present
- Tokenizer loading works
- CUDA availability (if applicable)
- spaCy model is available

### Manual Tests
```python
# Test basic imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import google.protobuf

# Test tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
print("✅ All imports successful!")
```

## Environment Setup Tips

### Virtual Environment (Recommended):
```bash
# Create virtual environment
python -m venv reddit_persona_env

# Activate it
# Windows:
reddit_persona_env\Scripts\activate
# macOS/Linux:
source reddit_persona_env/bin/activate

# Install dependencies
pip install -r requirements_fixed.txt
```

### Conda Environment:
```bash
# Create conda environment
conda create -n reddit_persona python=3.9

# Activate it
conda activate reddit_persona

# Install dependencies
pip install -r requirements_fixed.txt
```

## Performance Optimization

### For Better Performance:
1. **Use GPU**: Ensure CUDA is properly installed
2. **Increase RAM**: Close unnecessary applications
3. **SSD Storage**: Store models on SSD for faster loading
4. **Batch Processing**: Process multiple users at once (if implementing)

### For Lower Resource Usage:
1. **Use CPU Mode**: Set environment variable `CUDA_VISIBLE_DEVICES=""`
2. **Smaller Model**: Use `distilgpt2` instead of `facebook/opt-125m`
3. **Reduce Max Length**: Lower the `max_length` parameter in queries

## Debugging Tips

### Enable Detailed Logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Components:
```python
# Test Reddit API
from reddit_persona_generator import QuantizedLLMPersonaGenerator
generator = QuantizedLLMPersonaGenerator(...)
user_data = generator.fetch_user_data("username")

# Test LLM
response = generator._query_llm("Test prompt")
```

## Platform-Specific Issues

### Windows:
- Use Command Prompt or PowerShell as Administrator
- Ensure Visual Studio Build Tools are installed for bitsandbytes

### macOS:
- Use Terminal
- May need to install Xcode Command Line Tools: `xcode-select --install`

### Linux:
- Ensure CUDA drivers are properly installed for GPU support
- May need to install build-essential: `sudo apt install build-essential`

## Getting Help

If you're still experiencing issues:

1. **Check the error message** for specific clues
2. **Restart your Python environment** after installing packages
3. **Use a virtual environment** to avoid conflicts
4. **Update your graphics drivers** for GPU issues
5. **Check system requirements** (Python 3.7+, 8GB+ RAM recommended)

## System Requirements

### Minimum:
- Python 3.7+
- 8GB RAM
- 10GB free disk space

### Recommended:
- Python 3.9+
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 20GB free disk space (for models and cache)

## Contact

If none of these solutions work, please provide:
1. Your operating system and Python version
2. The complete error message
3. What you were trying to do when the error occurred
4. Whether you're using a virtual environment 