import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import torchaudio
from rich.console import Console
from rich import print as rprint
from demucs import pretrained
from torch.cuda import is_available as is_cuda_available
import gc

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw.mp3")
BACKGROUND_AUDIO_FILE = os.path.join(AUDIO_DIR, "background.mp3")
VOCAL_AUDIO_FILE = os.path.join(AUDIO_DIR, "vocal.mp3")

def load_audio(path, model_channels, model_samplerate):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > model_channels:
        wav = wav[:model_channels]
    elif wav.shape[0] < model_channels:
        wav = torch.cat([wav, torch.zeros(model_channels - wav.shape[0], wav.shape[1])], dim=0)
    
    if sr != model_samplerate:
        wav = torchaudio.transforms.Resample(sr, model_samplerate)(wav)
    return wav

def save_audio(tensor, path, sample_rate):
    torchaudio.save(path, tensor, sample_rate)

def demucs_main():
    if os.path.exists(VOCAL_AUDIO_FILE) and os.path.exists(BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {VOCAL_AUDIO_FILE} and {BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
    console = Console()
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    console.print("🤖 Loading demucs model...")
    model = pretrained.get_model('htdemucs')
    device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    
    console.print("🎵 Separating audio...")
    # Load and separate audio
    wav = load_audio(RAW_AUDIO_FILE, model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    with torch.no_grad():
        sources = model(wav[None].to(device))[0]
    sources = sources * ref.std() + ref.mean()
    
    sources = list(sources)
    sources = [source.cpu() for source in sources]
    
    console.print("🎤 Saving vocals track...")
    vocals = sources[model.sources.index('vocals')]
    save_audio(vocals, VOCAL_AUDIO_FILE, model.samplerate)
    
    console.print("🎹 Saving background music...")
    background = sum(source for i, source in enumerate(sources) if model.sources[i] != 'vocals')
    save_audio(background, BACKGROUND_AUDIO_FILE, model.samplerate)
    
    # Clean up memory
    del sources, background, model, wav
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_main()
