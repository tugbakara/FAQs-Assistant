"""
FAQ Chatbot Application - Main Dash Application
================================================
This module contains the main Dash application for the FAQ chatbot interface.
It handles UI rendering, user interactions, and real-time chat functionality.
"""
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from chatbot import ChatbotEngine
from services.config_service import config_service
import base64
import os
import logging
import time

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load UI configuration
ui_config = config_service.get_ui_config()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = ui_config.app_title

# Initialize chatbot engine
chatbot = ChatbotEngine()


def load_icon(icon_path: str) -> str:
    """
    Load and encode an icon image to base64 format.
    
    Args:
        icon_path (str): Path to the icon file
        
    Returns:
        str: Base64 encoded data URI string, or empty string if file not found
    """
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    return ""


def load_font(font_path: str) -> str:
    """
    Load and encode a font file to base64 format.
    
    Args:
        font_path (str): Path to the font file
        
    Returns:
        str: Base64 encoded data URI string, or empty string if file not found
    """
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:font/truetype;base64,{encoded}"
    return ""


def get_background_images() -> list:
    """
    Load all background images from the configured folder.
    
    Returns:
        list: List of base64 encoded image data URIs
    """
    bg_folder = ui_config.background_images_folder
    bg_images = []
    if os.path.exists(bg_folder):
        for filename in os.listdir(bg_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                file_path = os.path.join(bg_folder, filename)
                try:
                    with open(file_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()
                    bg_images.append(f"data:image/{filename.split('.')[-1]};base64,{encoded}")
                except Exception:
                    continue
    return bg_images


def create_welcome_message() -> html.Div:
    """
    Create the initial welcome message component.
    
    Returns:
        html.Div: Dash HTML component containing the welcome message
    """
    return html.Div([
        html.Div([
            html.Div([
                html.Div(className="ai-avatar"),
                html.Div([
                    "Hello! I'm your FAQs assistant. Ask me any question and I'll help you find answers from my knowledge base."
                ], className="message-content")
            ], className="message-layout")
        ], className="message-inner")
    ], className="message-container bot-message")


def create_message(content: str, is_user: bool = False, message_id: int = None) -> html.Div:
    """
    Create a chat message component.
    
    Args:
        content (str): The message text content
        is_user (bool): True if message is from user, False if from bot
        message_id (int, optional): Unique identifier for the message
        
    Returns:
        html.Div: Dash HTML component containing the formatted message
    """
    avatar_class = "user-avatar" if is_user else "ai-avatar"
    message_class = "user-message" if is_user else "bot-message"
    
    if is_user:
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(content, className="message-content"),
                    html.Div(className=avatar_class)
                ], className="message-layout")
            ], className="message-inner")
        ], className=f"message-container {message_class}")
    else: 
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(className=avatar_class),
                    html.Div([
                        html.Div(content, className="message-text")
                    ], className="message-content")
                ], className="message-layout")
            ], className="message-inner")
        ], className=f"message-container {message_class}")


def generate_background_css() -> str:
    """
    Generate CSS for background animation or slideshow.
    
    Returns:
        str: CSS string for background styling
    """
    if not background_images:
        return """
            body {
                background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
                background-size: 400% 400%;
                animation: gradientMove 15s ease infinite;
            }
            
            @keyframes gradientMove {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
        """
    
    keyframes = "@keyframes bgSlideshow {\n"
    step = 100 / len(background_images)
    
    for i, img in enumerate(background_images):
        percentage = i * step
        keyframes += f"    {percentage:.1f}% {{ background-image: url('{img}'); opacity: 1; }}\n"
        if i < len(background_images) - 1:
            next_percentage = (i + 1) * step - 2
            keyframes += f"    {next_percentage:.1f}% {{ opacity: 0; }}\n"

    keyframes += f"    100% {{ background-image: url('{background_images[0]}'); opacity: 1; }}\n"
    keyframes += "}\n"
    
    return f"""
        {keyframes}
        
        body::before {{
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            animation: bgSlideshow {len(background_images) * 8}s infinite ease-in-out !important;
            z-index: -2;
        }}
    """


# Load assets once at module level
enter_icon = load_icon(ui_config.icon_enter_path)
ai_icon = load_icon(ui_config.icon_ai_path)
user_icon = load_icon(ui_config.icon_user_path)
playfair_font = load_font(ui_config.font_playfair_path)
century_font = load_font(ui_config.font_century_path)
background_images = get_background_images()

# Define app layout
app.layout = html.Div([
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="initialization-status", data={"initialized": False}),
    
    html.Div([
        html.Div(id="progress-indicator"),
        dcc.Interval(id="progress-interval", interval=1000, n_intervals=0)
    ], style={"position": "fixed", "top": "10px", "right": "10px", 
              "background": "rgba(0,0,0,0.8)", "color": "white", 
              "padding": "10px", "border-radius": "5px", "z-index": "1000"}),
    
    html.Div([
        html.H2("FAQs Assistant", className="header-title")
    ], className="header-container"),
    
    html.Div([
        html.Div(id="chat-messages", children=[
            create_welcome_message() 
        ], className="chat-area")
    ], className="chat-container"),
    
    html.Div([
        html.Div([
            html.Div([
                dcc.Textarea(
                    id="user-input",
                    placeholder="Send a message...",
                    className="text-input",
                    rows=1
                ),
                html.Button([
                    html.Img(src=enter_icon, className="icon") if enter_icon else "Send"
                ], id="send-button", className="send-button")
            ], className="input-container")
        ], className="input-wrapper")
    ], className="input-area"),
    
], className="app-container")


@app.callback(
    Output("progress-indicator", "children"),
    Input("progress-interval", "n_intervals")
)
def update_progress(n: int) -> html.Div:
    """
    Update the initialization progress indicator.
    
    Args:
        n (int): Number of intervals elapsed
        
    Returns:
        html.Div: Progress indicator component
    """
    status = chatbot.get_initialization_status()
    
    if status["completed"]:
        return html.Div([
            "System Ready ✓"
        ], style={"color": "green", "font-weight": "bold"})
    elif status["error"]:
        return html.Div([
            f"Error: {status['error'][:50]}"
        ], style={"color": "red"})
    elif status["started"]:
        progress_text = f"{status['progress']} ({status['stage']}/{status['total_stages']})"
        return html.Div([
            dcc.Loading([
                html.Span(progress_text)
            ], type="default")
        ])
    else:
        return html.Div("Starting...")


@app.callback(
    Output("chat-messages", "style"),
    Input("chat-messages", "children"),
    prevent_initial_call=False
)
def auto_scroll_trigger(children: list) -> dict:
    """
    Trigger auto-scroll behavior for the chat area.
    
    Args:
        children (list): Current chat message components
        
    Returns:
        dict: Style dictionary with scroll trigger
    """
    return {"data-scroll-trigger": str(time.time())}


@app.callback(
    Output("chat-messages", "children"),
    Output("user-input", "value"),
    Output("chat-history", "data"),
    [Input("send-button", "n_clicks"),
     Input("user-input", "n_submit")],
    [State("user-input", "value"),
     State("chat-history", "data"),
     State("initialization-status", "data")],
    prevent_initial_call=True
)
def update_chat(send_clicks: int, input_submit: int, user_input: str, 
                chat_history: list, init_status: dict) -> tuple:
    """
    Handle chat message submission and update chat display.
    
    Args:
        send_clicks (int): Number of send button clicks
        input_submit (int): Number of input submissions
        user_input (str): User's input text
        chat_history (list): Current chat history
        init_status (dict): Initialization status
        
    Returns:
        tuple: (messages list, cleared input value, updated chat history)
    """
    status = chatbot.get_initialization_status()
    
    if not status["completed"]:
        if status["error"]:
            return [create_welcome_message(), 
                    html.Div("System initialization failed. Please restart the application.", 
                            className="error-message")], dash.no_update, dash.no_update
        else:
            return [create_welcome_message(), 
                    html.Div(f"System is initializing: {status['progress']}", 
                            className="info-message")], dash.no_update, dash.no_update
    
    if not user_input or user_input.strip() == "":
        return dash.no_update, dash.no_update, dash.no_update
    
    if chat_history is None:
        chat_history = []
    
    response = chatbot.get_response(user_input.strip())
    
    chat_history.append({"type": "user", "message": user_input.strip()})
    chat_history.append({"type": "bot", "message": response})
    
    messages = [create_welcome_message()]
    
    for i, msg in enumerate(chat_history):
        is_user = msg["type"] == "user"
        messages.append(create_message(msg["message"], is_user=is_user, message_id=i if not is_user else None))
    
    return messages, "", chat_history


# Generate custom HTML with embedded styles and scripts
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            @font-face {{
                font-family: 'PlayfairDisplay';
                src: url('{playfair_font}') format('truetype');
                font-weight: 600;
                font-display: swap;
            }}
            
            @font-face {{
                font-family: 'CenturyGothic';
                src: url('{century_font}') format('truetype');
                font-weight: normal;
                font-display: swap;
            }}
            
            body {{
                margin: 0;
                font-family: 'CenturyGothic', -apple-system, BlinkMacSystemFont, sans-serif;
                color: #e8e8e8;
                font-size: 17px;
                line-height: 1.7;
            }}
            
            {generate_background_css()}
            
            .app-container {{
                height: 100vh;
                display: flex;
                flex-direction: column;
                background: rgba(26, 26, 26, 0.85);
                backdrop-filter: blur(10px);
                position: relative;
                z-index: 1;
            }}
            
            .header-container {{
                padding: 20px;
                border-bottom: 1px solid #404040;
                background: rgba(45, 45, 45, 0.9);
                text-align: center;
                backdrop-filter: blur(10px);
            }}
            
            .header-title {{
                margin: 0; color: #f0f0f0; font-family: 'PlayfairDisplay', serif;
                font-weight: 600; font-size: 32px; letter-spacing: 0.5px;
            }}
            
            .chat-container {{ flex: 1; display: flex; flex-direction: column; min-height: 0; }}
            .chat-area {{ 
                flex: 1; 
                overflow-y: auto; 
                padding: 0; 
                max-width: 950px; 
                margin: 0 auto; 
                width: 100%; 
            }}
            
            .message-container {{
                padding: 18px 25px;
                display: flex;
            }}
            .message-inner {{ width: 100%; }}
            .message-layout {{ display: flex; align-items: flex-end; gap: 15px; }}
            
            .message-content {{
                padding: 20px 26px;
                border-radius: 28px;
                max-width: 85%;
                color: #e8e8e8;
                line-height: 1.8;
                font-size: 18px;
                position: relative;
            }}
            
            .ai-avatar, .user-avatar {{
                width: 50px; 
                height: 50px; 
                border-radius: 50%;
                flex-shrink: 0;
                background-size: cover;
                background-position: center;
            }}

            .ai-avatar {{
                background-image: url('{ai_icon}');
            }}
            .user-avatar {{
                background-image: url('{user_icon}');
            }}
            
            .bot-message {{ justify-content: flex-start; }}
            .bot-message .message-content {{
                background: #3a3a3a;
                border-bottom-left-radius: 8px;
            }}
            .bot-message .message-layout {{ justify-content: flex-start; }}
            
            .user-message {{ justify-content: flex-end; }}
            .user-message .message-content {{
                background: linear-gradient(135deg, #6a1b9a, #9c27b0);
                border-bottom-right-radius: 8px;
            }}
            .user-message .message-layout {{
                justify-content: flex-end;
            }}

            .message-text {{ 
                white-space: pre-wrap; 
                user-select: text;
                -webkit-user-select: text;
                -moz-user-select: text;
                -ms-user-select: text;
            }}
            
            .input-area {{
                border-top: 1px solid #404040; background: rgba(45, 45, 45, 0.9);
                padding: 25px; backdrop-filter: blur(10px);
            }}
            .input-wrapper {{ max-width: 850px; margin: 0 auto; width: 100%; }}
            
            .input-container {{
                display: flex; align-items: flex-end; background: rgba(26, 26, 26, 0.8);
                border: 1px solid #9c27b0; border-radius: 28px; padding: 16px 20px;
                gap: 15px; transition: border-color 0.2s ease;
            }}
            .input-container:focus-within {{ border-color: #e91e63; box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.1); }}
            
            .text-input {{
                flex: 1; border: none; outline: none; resize: none; background: transparent;
                font-size: 18px; line-height: 28px; max-height: 140px; min-height: 28px;
                color: #e8e8e8; font-family: 'CenturyGothic', sans-serif;
            }}
            .text-input::placeholder {{ color: #888; }}
            
            .send-button {{
                background: linear-gradient(135deg, #9c27b0, #e91e63); border: none;
                border-radius: 14px; padding: 12px; cursor: pointer; display: flex;
                align-items: center; justify-content: center; transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(156, 39, 176, 0.3);
            }}
            .send-button:hover {{ background: linear-gradient(135deg, #7b1fa2, #c2185b); box-shadow: 0 4px 12px rgba(156, 39, 176, 0.4); }}
            
            .icon {{ width: 18px; height: 18px; filter: brightness(0) invert(1); }}
            
            .error-message {{
                background: rgba(244, 67, 54, 0.2);
                color: #f44336;
                padding: 18px;
                border-radius: 12px;
                margin: 25px;
                text-align: center;
                font-size: 16px;
            }}
            
            .info-message {{
                background: rgba(33, 150, 243, 0.2);
                color: #2196f3;
                padding: 18px;
                border-radius: 12px;
                margin: 25px;
                text-align: center;
                font-size: 16px;
            }}
            
            ::-webkit-scrollbar {{ width: 10px; }}
            ::-webkit-scrollbar-track {{ background: rgba(26, 26, 26, 0.5); }}
            ::-webkit-scrollbar-thumb {{ background: rgba(156, 39, 176, 0.6); border-radius: 5px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: rgba(156, 39, 176, 0.8); }}
        </style>
        <script>
            function autoScrollToBottom() {{
                const chatArea = document.querySelector('.chat-area');
                if (chatArea) {{
                    chatArea.scrollTo({{
                        top: chatArea.scrollHeight,
                        behavior: 'smooth'
                    }});
                }}
            }}

            function setupAutoScroll() {{
                const chatMessages = document.getElementById('chat-messages');
                if (chatMessages) {{
                    const observer = new MutationObserver((mutations) => {{
                        mutations.forEach((mutation) => {{
                            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {{
                                setTimeout(autoScrollToBottom, 100);
                            }}
                        }});
                    }});
                    
                    observer.observe(chatMessages, {{
                        childList: true,
                        subtree: true
                    }});
                    
                    setTimeout(autoScrollToBottom, 200);
                }}
            }}

            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', setupAutoScroll);
            }} else {{
                setupAutoScroll();
            }}

            setInterval(() => {{
                const trigger = document.querySelector('[data-scroll-trigger]');
                if (trigger && trigger.dataset.scrollTrigger !== window.lastScrollTrigger) {{
                    window.lastScrollTrigger = trigger.dataset.scrollTrigger;
                    autoScrollToBottom();
                }}
            }}, 100);
        </script>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

if __name__ == "__main__":
    # Start initialization only when running as main script
    chatbot.initialize()
    app.run(host=ui_config.app_host, port=ui_config.app_port, debug=ui_config.debug_mode)