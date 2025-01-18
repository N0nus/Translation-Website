import torch
from tokenizers import Tokenizer
from transformer import build_transformer
import requests
from io import BytesIO
import os
from pathlib import Path

def get_tokenizer_path(filename):
    """Get absolute path to tokenizer file."""
    current_dir = Path(__file__).parent
    return str(current_dir / 'de2en' / filename)

def translate(sentence: str):
    # Get model URL from environment variable
    model_url = "https://en2de.blob.core.windows.net/en2de-model/models/tmodel_09.pt?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-01-18T23:53:56Z&st=2025-01-18T15:53:56Z&spr=https&sig=4M3dHAVkd5qdBktolmJ6G%2FvDkfyvRtTiiEo6PaL2bM4%3D"
    if not model_url:
        raise ValueError("MODEL_URL environment variable is not set")
    
    # Define the device - prefer CPU in cloud environment unless GPU is specifically configured
    device = torch.device("cpu")
    
    try:
        # Load tokenizers with absolute paths
        tokenizer_src = Tokenizer.from_file(get_tokenizer_path("tokenizer_src.json"))
        tokenizer_tgt = Tokenizer.from_file(get_tokenizer_path("tokenizer_tgt.json"))
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizers: {e}")
    
    # Build the transformer model
    try:
        model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            64,
            64,
            512
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to build transformer model: {e}")
    
    # Download and load the model with proper error handling
    try:
        response = requests.get(model_url, timeout=30)  # Add timeout
        response.raise_for_status()
        
        # Load the model into memory
        try:
            state = torch.load(BytesIO(response.content), map_location=device, weights_only=True)
            model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}")
            
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download model: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading model: {e}")
    
    # Translation logic with error handling
    try:
        seq_len = 64
        model.eval()
        with torch.no_grad():
            # Encode the source sentence
            source = tokenizer_src.encode(sentence)
            if not source:
                raise ValueError("Failed to encode source sentence")

            # Prepare input tensor
            source = torch.cat([
                torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)

            # Create source mask
            source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            
            # Generate translation
            encoder_output = model.encode(source, source_mask)
            decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

            for _ in range(seq_len - 1):
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
                if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                    break

            # Decode the translation
            translation = tokenizer_tgt.decode(decoder_input[0].tolist())
            if not translation:
                raise ValueError("Failed to generate translation")
                
            return translation

    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}")