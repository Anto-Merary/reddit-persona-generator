#!/usr/bin/env python3
"""
Reddit Persona Generator Flask Application

A modern, modular Flask application for generating comprehensive user personas
from Reddit profiles using advanced NLP and quantized LLM technology.

Features:
- Modular architecture with separated concerns
- Professional REST API endpoints
- Beautiful web interface
- Comprehensive error handling
- Health monitoring and system information

Author: Anto Merary S
Date: 2025
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Fix console encoding for Windows
if sys.platform.startswith('win'):
    try:
        # Try to set console to UTF-8 mode
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        # Fallback if UTF-8 setup fails
        pass

# Import our modular components
from config import Config
from persona_generator import PersonaGenerator
from persona_generator.utils import setup_logging, setup_protobuf_compatibility


class PersonaGeneratorApp:
    """Main Flask application class for the Persona Generator."""
    
    def __init__(self):
        """Initialize the Flask application."""
        # Setup protobuf compatibility
        setup_protobuf_compatibility()
        
        # Setup logging
        self.logger = setup_logging()
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web interface
        
        # Initialize persona generator
        self.persona_generator = None
        self._initialize_generator()
        
        # Register routes
        self._register_routes()
        
        self.logger.info("Flask application initialized successfully")
    
    def _initialize_generator(self):
        """Initialize the persona generator with error handling."""
        try:
            self.persona_generator = PersonaGenerator()
            self.logger.info("[SUCCESS] PersonaGenerator initialized successfully")
        except Exception as e:
            self.logger.error(f"[ERROR] Error initializing PersonaGenerator: {e}")
            if "protobuf" in str(e).lower():
                self.logger.error("[WARNING] PROTOBUF COMPATIBILITY ISSUE DETECTED!")
                self.logger.error("Please ensure you have compatible protobuf version installed")
            self.persona_generator = None
    
    def _register_routes(self):
        """Register all Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main web interface."""
            return render_template_string(self._get_web_interface_template())
        
        @self.app.route('/generate_persona', methods=['POST'])
        def generate_persona_api():
            """API endpoint to generate persona from Reddit profile URL."""
            try:
                # Check if generator is available
                if not self.persona_generator:
                    return jsonify({
                        'error': 'PersonaGenerator not initialized. Please check system logs.'
                    }), 500
                
                # Get JSON data from request
                data = request.get_json()
                
                if not data or 'profile_url' not in data:
                    return jsonify({
                        'error': 'Missing profile_url in request body'
                    }), 400
                
                profile_url = data['profile_url']
                
                # Validate URL format
                if not profile_url or 'reddit.com/user/' not in profile_url:
                    return jsonify({
                        'error': 'Invalid Reddit profile URL format'
                    }), 400
                
                # Generate persona
                result = self.persona_generator.generate_persona_from_url(profile_url)
                
                if 'error' in result:
                    return jsonify(result), 400
                
                return jsonify(result), 200
                
            except Exception as e:
                self.logger.error(f"Error in generate_persona_api: {e}")
                return jsonify({
                    'error': f'Internal server error: {str(e)}'
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint with system information."""
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'generator_initialized': self.persona_generator is not None
            }
            
            if self.persona_generator:
                try:
                    health_data['system_info'] = self.persona_generator.get_system_info()
                except Exception as e:
                    health_data['system_info_error'] = str(e)
            
            return jsonify(health_data)
        
        @self.app.route('/api/info', methods=['GET'])
        def api_info():
            """Get API information and available endpoints."""
            return jsonify({
                'name': 'Reddit Persona Generator API',
                'version': '2.0.0',
                'endpoints': {
                    'POST /generate_persona': 'Generate persona from Reddit URL',
                    'GET /health': 'Health check and system status',
                    'GET /api/info': 'API information',
                    'GET /': 'Web interface'
                },
                'required_payload': {
                    'profile_url': 'Reddit profile URL (e.g., https://www.reddit.com/user/username/)'
                }
            })
    
    def _get_web_interface_template(self) -> str:
        """Get the HTML template for the web interface."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reddit Persona Generator</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: #f8f9fa; 
                    color: #333;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                h1 { 
                    color: #2c3e50; 
                    text-align: center; 
                    margin-bottom: 30px;
                }
                .subtitle {
                    text-align: center;
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 1.1em;
                }
                .form-section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }
                input[type="text"] { 
                    width: 100%; 
                    padding: 12px; 
                    margin: 10px 0; 
                    border: 2px solid #e9ecef;
                    border-radius: 6px;
                    font-size: 16px;
                    box-sizing: border-box;
                }
                button { 
                    background: #e67e22; 
                    color: white; 
                    padding: 12px 24px; 
                    border: none; 
                    border-radius: 6px; 
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 600;
                }
                button:hover { background: #d35400; }
                button:disabled { background: #bdc3c7; cursor: not-allowed; }
                .persona-card {
                    display: none;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    margin-top: 30px;
                    overflow: hidden;
                }
                .persona-header {
                    background: linear-gradient(135deg, #e67e22, #f39c12);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }
                .persona-name {
                    font-size: 2.5em;
                    font-weight: 700;
                    margin-bottom: 10px;
                }
                .persona-content {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    padding: 30px;
                }
                .persona-section {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                }
                .section-title {
                    font-size: 1.4em;
                    font-weight: 600;
                    color: #e67e22;
                    margin-bottom: 15px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .demographics-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                    margin-bottom: 20px;
                }
                .demo-item {
                    background: white;
                    padding: 10px;
                    border-radius: 6px;
                    border-left: 4px solid #e67e22;
                }
                .demo-label {
                    font-weight: 600;
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .demo-value {
                    font-size: 1.1em;
                    color: #2c3e50;
                    margin-top: 4px;
                }
                .motivation-bar {
                    display: flex;
                    align-items: center;
                    margin-bottom: 12px;
                }
                .motivation-label {
                    width: 120px;
                    font-weight: 600;
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                }
                .motivation-progress {
                    flex: 1;
                    height: 8px;
                    background: #e9ecef;
                    border-radius: 4px;
                    margin: 0 15px;
                    overflow: hidden;
                }
                .motivation-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #e67e22, #f39c12);
                    border-radius: 4px;
                    transition: width 0.3s ease;
                }
                .motivation-value {
                    font-weight: 600;
                    color: #2c3e50;
                    min-width: 25px;
                }
                .personality-trait {
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }
                .trait-left {
                    width: 90px;
                    text-align: right;
                    font-size: 0.9em;
                    color: #666;
                    font-weight: 600;
                }
                .trait-slider {
                    flex: 1;
                    height: 20px;
                    background: #e9ecef;
                    border-radius: 10px;
                    margin: 0 15px;
                    position: relative;
                    overflow: hidden;
                }
                .trait-indicator {
                    position: absolute;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 16px;
                    height: 16px;
                    background: #e67e22;
                    border-radius: 50%;
                    border: 2px solid white;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                }
                .trait-right {
                    width: 90px;
                    font-size: 0.9em;
                    color: #666;
                    font-weight: 600;
                }
                .bullet-list {
                    list-style: none;
                    padding: 0;
                }
                .bullet-list li {
                    background: white;
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    border-left: 4px solid #e67e22;
                    position: relative;
                }
                .bullet-list li::before {
                    content: "•";
                    color: #e67e22;
                    font-weight: bold;
                    position: absolute;
                    left: -8px;
                }
                .quote-section {
                    grid-column: 1 / -1;
                    background: #2c3e50;
                    color: white;
                    padding: 30px;
                    border-radius: 8px;
                    text-align: center;
                    font-style: italic;
                    font-size: 1.3em;
                    position: relative;
                }
                .quote-section::before {
                    content: '"';
                    font-size: 4em;
                    color: #e67e22;
                    position: absolute;
                    top: -10px;
                    left: 20px;
                }
                .quote-section::after {
                    content: '"';
                    font-size: 4em;
                    color: #e67e22;
                    position: absolute;
                    bottom: -40px;
                    right: 20px;
                }
                .error { 
                    background: #ffe6e6; 
                    color: #cc0000; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin-top: 20px;
                }
                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #666;
                }
                .loading-spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #e67e22;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 20px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .avatar-placeholder {
                    width: 120px;
                    height: 120px;
                    background: #bdc3c7;
                    border-radius: 50%;
                    margin: 0 auto 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 3em;
                    color: white;
                }
                .footer {
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #e9ecef;
                    color: #666;
                }
                @media (max-width: 768px) {
                    .persona-content {
                        grid-template-columns: 1fr;
                    }
                    .demographics-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reddit Persona Generator</h1>
                <p class="subtitle">
                    Generate comprehensive AI-powered persona analysis from Reddit user profiles
                </p>
                
                <div class="form-section">
                    <form id="personaForm">
                        <input type="text" id="profileUrl" placeholder="https://www.reddit.com/user/username/" required>
                        <button type="submit" id="generateButton">Generate Persona</button>
                    </form>
                </div>
                
                <div id="result"></div>
                
                <div class="footer">
                    <p>Powered by Advanced NLP and Quantized LLM Technology</p>
                </div>
            </div>
            
            <script>
                document.getElementById('personaForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const url = document.getElementById('profileUrl').value;
                    const resultDiv = document.getElementById('result');
                    const generateButton = document.getElementById('generateButton');
                    
                    // Show loading state
                    generateButton.disabled = true;
                    generateButton.textContent = 'Generating...';
                    resultDiv.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Analyzing Reddit profile and generating persona... This may take a few minutes.</div>';
                    
                    try {
                        const response = await fetch('/generate_persona', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ profile_url: url })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${data.error}</div>`;
                        } else {
                            displayPersonaCard(data);
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
                    } finally {
                        generateButton.disabled = false;
                        generateButton.textContent = 'Generate Persona';
                    }
                });
                
                function displayPersonaCard(data) {
                    const resultDiv = document.getElementById('result');
                    const persona = data.persona;
                    
                    // Extract quote text
                    let quoteText = persona.quote;
                    if (typeof quoteText === 'object' && quoteText.description) {
                        quoteText = quoteText.description;
                    }
                    
                    const personaHtml = `
                        <div class="persona-card" style="display: block;">
                            <div class="persona-header">
                                <div class="avatar-placeholder">
                                    ${data.username.charAt(0).toUpperCase()}
                                </div>
                                <div class="persona-name">${data.username}</div>
                            </div>
                            
                            <div class="persona-content">
                                <div class="persona-section">
                                    <div class="section-title">Demographics</div>
                                    <div class="demographics-grid">
                                        <div class="demo-item">
                                            <div class="demo-label">Age</div>
                                            <div class="demo-value">${persona.demographics.age}</div>
                                        </div>
                                        <div class="demo-item">
                                            <div class="demo-label">Occupation</div>
                                            <div class="demo-value">${persona.demographics.occupation}</div>
                                        </div>
                                        <div class="demo-item">
                                            <div class="demo-label">Status</div>
                                            <div class="demo-value">${persona.demographics.status}</div>
                                        </div>
                                        <div class="demo-item">
                                            <div class="demo-label">Location</div>
                                            <div class="demo-value">${persona.demographics.location}</div>
                                        </div>
                                        <div class="demo-item">
                                            <div class="demo-label">Tier</div>
                                            <div class="demo-value">${persona.demographics.tier}</div>
                                        </div>
                                        <div class="demo-item">
                                            <div class="demo-label">Archetype</div>
                                            <div class="demo-value">${persona.demographics.archetype}</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="persona-section">
                                    <div class="section-title">Motivations</div>
                                    ${Object.entries(persona.motivations).map(([key, value]) => `
                                        <div class="motivation-bar">
                                            <div class="motivation-label">${key}</div>
                                            <div class="motivation-progress">
                                                <div class="motivation-fill" style="width: ${value * 10}%"></div>
                                            </div>
                                            <div class="motivation-value">${value}</div>
                                        </div>
                                    `).join('')}
                                </div>
                                
                                <div class="persona-section">
                                    <div class="section-title">Personality</div>
                                    ${createPersonalityTraits(persona.personality)}
                                </div>
                                
                                <div class="persona-section">
                                    <div class="section-title">Behaviour & Habits</div>
                                    <ul class="bullet-list">
                                        ${persona.behavior.items ? persona.behavior.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific behaviors identified</li>'}
                                    </ul>
                                </div>
                                
                                <div class="persona-section">
                                    <div class="section-title">Frustrations</div>
                                    <ul class="bullet-list">
                                        ${persona.frustrations.items ? persona.frustrations.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific frustrations identified</li>'}
                                    </ul>
                                </div>
                                
                                <div class="persona-section">
                                    <div class="section-title">Goals & Needs</div>
                                    <ul class="bullet-list">
                                        ${persona.goals.items ? persona.goals.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific goals identified</li>'}
                                    </ul>
                                </div>
                                
                                <div class="quote-section">
                                    ${quoteText}
                                </div>
                            </div>
                        </div>
                    `;
                    
                    resultDiv.innerHTML = personaHtml;
                }
                
                function createPersonalityTraits(personality) {
                    const traits = [
                        { left: 'Introvert', right: 'Extrovert', key: 'extraversion' },
                        { left: 'Intuition', right: 'Sensing', key: 'sensing' },
                        { left: 'Feeling', right: 'Thinking', key: 'thinking' },
                        { left: 'Perceiving', right: 'Judging', key: 'judging' }
                    ];
                    
                    return traits.map(trait => `
                        <div class="personality-trait">
                            <div class="trait-left">${trait.left}</div>
                            <div class="trait-slider">
                                <div class="trait-indicator" style="left: ${((personality[trait.key] || 5) - 1) * 10}%"></div>
                            </div>
                            <div class="trait-right">${trait.right}</div>
                        </div>
                    `).join('');
                }
            </script>
        </body>
        </html>
        """
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the Flask application."""
        flask_config = Config.get_flask_config()
        
        self.app.run(
            host=host or flask_config['host'],
            port=port or flask_config['port'],
            debug=debug if debug is not None else flask_config['debug']
        )


def main():
    """Main function to run the application."""
    print("Reddit Persona Generator v2.0")
    print("=" * 50)
    
    # Create and run the application
    app = PersonaGeneratorApp()
    
    print("\nStarting Flask application...")
    print("Web Interface: http://localhost:5000")
    print("API Endpoint: POST /generate_persona")
    print("Health Check: GET /health")
    print("API Info: GET /api/info")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        if app.persona_generator:
            app.persona_generator.cleanup()
        print("[SUCCESS] Application stopped")


if __name__ == "__main__":
    main() 