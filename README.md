# FAQs Chatbot

AI-powered FAQ chatbot application built with Dash and Sentence Transformers. This chatbot uses semantic search to provide accurate answers from a large FAQ knowledge base.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/dash-latest-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<img width="1851" height="932" alt="Screenshot from 2025-10-01 15-04-48" src="https://github.com/user-attachments/assets/fabd3551-5c9b-44d8-b383-ce209b1b3815" />


## Quick Start

### Prerequisites

- Python 3.8 or higher
- (Optional) CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/faq-chatbot.git
   cd faq-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**
   
   Edit `config.json` to customize settings:
   ```json
   {
     "model": {
       "dataset_name": "vishal-burman/c4-faqs",
       "model_name": "sentence-transformers/all-mpnet-base-v2",
       "similarity_threshold": 0.78,
       "use_gpu": true
     },
     "ui": {
       "app_title": "FAQ Chatbot",
       "app_port": 8050
     }
   }
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the chatbot**
   
   Open your browser and navigate to: `http://127.0.0.1:8050`

## Configuration

### Model Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_name` | HuggingFace dataset to use | `vishal-burman/c4-faqs` |
| `model_name` | Sentence transformer model | `all-mpnet-base-v2` |
| `similarity_threshold` | Minimum similarity score (0-1) | `0.78` |
| `max_faqs` | Maximum FAQs to load (null = all) | `null` |
| `batch_size` | Embedding batch size | `64` |
| `use_gpu` | Enable GPU acceleration | `true` |


### Cache Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_cache_size` | Max embeddings in memory | `1000` |
| `query_cache_size` | Max cached queries | `5000` |
| `cache_ttl_hours` | Cache time-to-live | `24` |

## Project Structure

```
faq-chatbot/
│
├── app.py                      # Main application entry point
├── chatbot.py                  # Chatbot engine logic
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
│
├── services/                   # Microservices layer
│   ├── config_service.py      # Configuration management
│   ├── cache_service.py       # Caching mechanisms
│   ├── orchestrator_service.py # Service coordination
│   └── search_service.py      # Semantic search engine
│
├── fonts/                      # Custom fonts
│   ├── Playfair_Display/
│   └── CenturyGothic/
│
├── icons/                      # UI icons
│   ├── enter.png
│   ├── ai.png
│   └── user.png
│
├── bg/                         # Background images
│
└── *.pkl                       # Cache files (generated)
```

## Customization

### Custom Fonts

Place your custom fonts in the `fonts/` directory and update paths in `config.json`:

```json
{
  "ui": {
    "font_playfair_path": "fonts/YourFont/font.ttf",
    "font_century_path": "fonts/YourFont2/font.ttf"
  }
}
```

### Custom Icons

Replace icons in the `icons/` directory:
- `ai.png` - Bot avatar (50x50px recommended)
- `user.png` - User avatar (50x50px recommended)
- `enter.png` - Send button icon (18x18px recommended)

### Background Images

Add background images to the `bg/` folder. Supported formats:
- PNG
- JPG/JPEG
- GIF
- WebP

The app will automatically create a slideshow from all images in this folder.

### Optimization Tips

1. **Enable GPU**: Set `"use_gpu": true` for 3-5x faster processing
2. **Adjust Batch Size**: Increase `batch_size` for faster initialization
3. **Cache Configuration**: Tune cache sizes based on your usage patterns

## Troubleshooting

### Common Issues

**Issue**: Model fails to load
- **Solution**: Check internet connection for first-time model download
- **Solution**: Verify sufficient disk space (~500MB for model)

**Issue**: Slow initialization
- **Solution**: Enable GPU acceleration if available
- **Solution**: Reduce `max_faqs` in configuration

**Issue**: High memory usage
- **Solution**: Reduce `embedding_cache_size` and `query_cache_size`
- **Solution**: Limit `max_faqs` to needed amount

**Issue**: Port already in use
- **Solution**: Change `app_port` in config.json
- **Solution**: Kill process using the port: `lsof -ti:8050 | xargs kill -9`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] Multi-language support
- [ ] Custom dataset upload interface
- [ ] Conversation history export
- [ ] Advanced analytics dashboard
- [ ] API endpoint for external integrations
- [ ] Docker containerization
- [ ] Cloud deployment guides
