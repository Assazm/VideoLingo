import os
import sys
import warnings
warnings.filterwarnings("ignore")
import whisperx
import torch
from rich import print as rprint
import tempfile
import subprocess
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import defaultdict, OrderedDict, Counter
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from omegaconf.dictconfig import DictConfig
from torch.serialization import add_safe_globals

# Add all necessary classes to safe globals
add_safe_globals([
    # omegaconf classes
    ListConfig,
    ContainerMetadata,
    DictConfig,
    # typing classes
    Any,
    Optional,
    Union,
    Dict,
    List,
    Tuple,
    # built-in classes
    list,
    dict,
    tuple,
    str,
    int,
    float,
    bool,
    # collections classes
    defaultdict,
    OrderedDict,
    Counter
])

# Monkey patch torch.load to use weights_only=False by default
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import librosa
import subprocess
import time

from core.config_utils import load_key
from core.all_whisper_methods.demucs_vl import demucs_main, RAW_AUDIO_FILE, VOCAL_AUDIO_FILE
from core.all_whisper_methods.whisperX_utils import process_transcription, convert_video_to_audio, split_audio, save_results, save_language, compress_audio, CLEANED_CHUNKS_EXCEL_PATH
from core.step1_ytdlp import find_video_files

MODEL_DIR = load_key("model_dir")
WHISPER_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_PATH = "output/audio/enhanced_vocals.mp3"

def check_hf_mirror() -> str:
    """Check and return the fastest HF mirror"""
    mirrors = {
        'Official': 'huggingface.co',
        'Mirror': 'hf-mirror.com'
    }
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        try:
            if os.name == 'nt':
                cmd = ['ping', '-n', '1', '-w', '3000', domain]
            else:
                cmd = ['ping', '-c', '1', '-W', '3', domain]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            response_time = time.time() - start
            if result.returncode == 0:
                if response_time < best_time:
                    best_time = response_time
                    fastest_url = f"https://{domain}"
                rprint(f"[green] {name}: {response_time:.2f}s[/green]")
        except:
            rprint(f"[red] {name}: Failed to connect[/red]")
    if best_time == float('inf'):
        rprint("[yellow]All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]Selected mirror: {fastest_url} ({best_time:.2f}s)[/cyan]")
    return fastest_url

def transcribe_audio(audio_file: str, start: float, end: float) -> Dict:
    os.environ['HF_ENDPOINT'] = check_hf_mirror() #? don't know if it's working...
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(f"[cyan]GPU memory: {gpu_mem:.2f} GB, Batch size: {batch_size}, Compute type: {compute_type}[/cyan]")
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(f"[cyan]Batch size: {batch_size}, Compute type: {compute_type}[/cyan]")
    rprint(f"[green]Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    try:
        if WHISPER_LANGUAGE == 'zh':
            model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
            local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
        else:
            model_name = load_key("whisper.model")
            local_model = os.path.join(MODEL_DIR, model_name)
            
        if os.path.exists(local_model):
            rprint(f"[green]Loading local WHISPER model: {local_model} ...[/green]")
            model_name = local_model
        else:
            rprint(f"[green]Using WHISPER model from HuggingFace: {model_name} ...[/green]")

        vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}
        asr_options = {
            "beam_size": 5,
            "temperatures": [0],  
            "best_of": 5,
            "patience": 1,
            "length_penalty": 1,
            "suppress_tokens": [-1],
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "prefix": None,
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True
        }
        whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
        rprint("[bold yellow]**You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`**[/bold yellow]")
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

        # Create temp file with wav format for better compatibility
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio segment using ffmpeg
        ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 32000 -ac 1 "{temp_audio_path}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        try:
            # Load audio segment with librosa
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")
        result = model.transcribe(audio_segment, batch_size=batch_size, print_progress=True)

        # Free GPU resources
        del model
        torch.cuda.empty_cache()

        # Save language
        save_language(result['language'])
        if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
            raise ValueError("Please specify the transcription language as zh and try again!")

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_segment, device, return_char_alignments=False)

        # Free GPU resources again
        torch.cuda.empty_cache()
        del model_a

        # Adjust timestamps
        for segment in result['segments']:
            segment['start'] += start
            segment['end'] += start
            for word in segment['words']:
                if 'start' in word:
                    word['start'] += start
                if 'end' in word:
                    word['end'] += start
        return result
    except Exception as e:
        rprint(f"[red]WhisperX processing error: {e}[/red]")
        raise

def enhance_vocals(vocals_ratio=2.50):
    """Enhance vocals audio volume"""
    if not load_key("demucs"):
        return RAW_AUDIO_FILE
        
    try:
        print(f"[cyan]Enhancing vocals with volume ratio: {vocals_ratio}[/cyan]")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{VOCAL_AUDIO_FILE}" '
            f'-filter:a "volume={vocals_ratio}" '
            f'"{ENHANCED_VOCAL_PATH}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        return ENHANCED_VOCAL_PATH
    except subprocess.CalledProcessError as e:
        print(f"[red]Error enhancing vocals: {str(e)}[/red]")
        return VOCAL_AUDIO_FILE  # Fallback to original vocals if enhancement fails
    
def transcribe():
    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH):
        rprint("[yellow]Transcription results already exist, skipping transcription step.[/yellow]")
        return
    
    # step0 Convert video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # step1 Demucs vocal separation:
    if load_key("demucs"):
        demucs_main()
    
    # step2 Compress audio
    choose_audio = enhance_vocals() if load_key("demucs") else RAW_AUDIO_FILE
    whisper_audio = compress_audio(choose_audio, WHISPER_FILE)

    # step3 Extract audio
    segments = split_audio(whisper_audio)
    
    # step4 Transcribe audio
    all_results = []
    for start, end in segments:
        result = transcribe_audio(whisper_audio, start, end)
        all_results.append(result)
    
    # step5 Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # step6 Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()