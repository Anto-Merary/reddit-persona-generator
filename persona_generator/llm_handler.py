"""
LLM Handler module for the Persona Generator.

This module handles all interactions with the quantized Large Language Model,
including model loading, text generation, and response processing.
"""

import logging
import warnings
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from .utils import setup_protobuf_compatibility

# Try to import BitsAndBytesConfig with fallback
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    QUANTIZATION_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class LLMHandler:
    """Handler for quantized Large Language Model interactions."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the LLM handler.
        
        Args:
            model_name: Name of the model to load (optional, uses config default)
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup protobuf compatibility
        setup_protobuf_compatibility()
        
        # GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # Get configuration
        self.llm_config = Config.get_llm_config()
        self.model_name = model_name or self.llm_config['model_name']
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the quantized LLM model with optimizations."""
        try:
            self.logger.info(f"Loading quantized model: {self.model_name}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.logger.warning("No GPU detected, will use CPU")
            
            # Load tokenizer first
            self._load_tokenizer()
            
            # Load model with quantization if available
            self._load_quantized_model()
            
        except Exception as e:
            self.logger.error(f"Error loading LLM: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer with error handling."""
        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("[SUCCESS] Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            if "protobuf" in str(e).lower():
                self.logger.error("Protobuf compatibility issue. Please install compatible versions.")
            raise
    
    def _load_quantized_model(self):
        """Load the model with quantization optimizations."""
        # Try to load with 4-bit quantization first (optimal for RTX Cards)
        if torch.cuda.is_available() and QUANTIZATION_AVAILABLE:
            try:
                self.logger.info("Attempting to load model with 4-bit quantization on GPU...")
                
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.llm_config['quantization_config']['load_in_4bit'],
                    bnb_4bit_compute_dtype=getattr(torch, self.llm_config['quantization_config']['bnb_4bit_compute_dtype']),
                    bnb_4bit_quant_type=self.llm_config['quantization_config']['bnb_4bit_quant_type'],
                    bnb_4bit_use_double_quant=self.llm_config['quantization_config']['bnb_4bit_use_double_quant'],
                )
                
                # Load model with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                self.logger.info("[SUCCESS] Model loaded successfully with 4-bit quantization on GPU")
                self.device = "cuda"
                return
                
            except Exception as e:
                self.logger.warning(f"4-bit quantization failed: {e}")
                self.logger.info("Trying regular GPU loading...")
        
        # Fallback to regular GPU loading
        if torch.cuda.is_available():
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                self.logger.info("[SUCCESS] Model loaded successfully on GPU without quantization")
                self.device = "cuda"
                return
                
            except Exception as e:
                self.logger.warning(f"GPU loading failed: {e}")
                self.logger.info("Falling back to CPU loading...")
        
        # Final fallback to CPU
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            self.logger.info("[SUCCESS] Model loaded successfully on CPU")
            self.device = "cpu"
            
        except Exception as e:
            self.logger.error(f"CPU loading failed: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None, for_quote: bool = False) -> str:
        """
        Generate text using the quantized LLM.
        
        Args:
            prompt: Input prompt for generation
            max_length: Maximum response length (optional)
            for_quote: Whether this is for quote generation (uses different config)
            
        Returns:
            Generated text response
        """
        try:
            # Select configuration based on generation type
            generation_config = (
                self.llm_config['quote_generation_config'] if for_quote 
                else self.llm_config['generation_config']
            )
            
            # Use appropriate max length
            if max_length is None:
                max_length = (
                    self.llm_config['max_quote_length'] if for_quote 
                    else self.llm_config['max_response_length']
                )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=800
            )
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response, for_quote)
            
            return response if response else ("Every day is a chance to learn something new." if for_quote else "Unable to generate response.")
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error generating response: {e}"
    
    def _clean_response(self, response: str, for_quote: bool = False) -> str:
        """
        Clean up LLM response by removing artifacts.
        
        Args:
            response: Raw response from LLM
            for_quote: Whether this is a quote response
            
        Returns:
            Cleaned response
        """
        # Remove common artifacts
        response = response.replace('<|endoftext|>', '')
        response = response.replace('</s>', '')
        response = response.strip()
        
        if for_quote:
            return self._clean_quote_response(response)
        else:
            return self._clean_general_response(response)
    
    def _clean_quote_response(self, response: str) -> str:
        """Clean quote-specific responses."""
        # Remove quotes if they wrap the entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        
        # Take only the first sentence for quotes
        sentences = response.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if first_sentence and len(first_sentence) > 8:
                return first_sentence
        
        # Take only the first line if multiple lines
        lines = response.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) > 8:
                return first_line
        
        return response.strip()
    
    def _clean_general_response(self, response: str) -> str:
        """Clean general responses."""
        # Remove placeholder text and generic instructions
        artifact_patterns = [
            'this content is being provided for informational purposes',
            'please check with your local community',
            'set goals by category',
            'add objectives by year',
            'include other categories if you have them',
            'create a user profile that has',
            'some things to consider',
            'quote text here',
            'format as:',
            'format as',
            'based on the analysis above',
            'detailed analysis',
            'evidence from their behavior',
            'word cards',
            'posters that highlight',
            'customize to give a clear picture',
            'identify 3-5 specific',
            'generate 3-5 specific',
            'critical instructions',
            'consider the following'
        ]
        
        # Filter out lines containing artifacts
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that contain artifact patterns
            is_artifact = any(pattern in line_lower for pattern in artifact_patterns)
            if is_artifact:
                continue
                
            # Skip repetitive lines
            if line not in cleaned_lines:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def is_available(self) -> bool:
        """Check if the LLM is loaded and available."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_available():
            return {'status': 'Not loaded'}
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'quantization_available': QUANTIZATION_AVAILABLE,
            'cuda_available': torch.cuda.is_available(),
            'status': 'Ready'
        }
    
    def cleanup(self):
        """Clean up GPU memory and resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("LLM handler cleanup completed") 