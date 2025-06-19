import os
import whisper
import pysrt
# Supprimer l'import moviepy pour l'extraction
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import google.generativeai as genai
import json
import ffmpeg # <-- Importer ffmpeg-python
import time # <-- Importer time pour la pause
import re # Ajouter cet import en haut du fichier pour la sanitization
from PIL import Image, ImageTk

# Ajouter ces imports en haut du fichier
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from threading import Thread

def transcribe_video(video_path, output_srt="transcription.srt", language="en"):
    model = whisper.load_model("medium")
    result = model.transcribe(
        video_path,
        language=language,
        fp16=False,
        verbose=False
    )
    with open(output_srt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"{start} --> {end}\n{text}\n\n")
    return output_srt

def analyze_semantic_content(srt_file):
    subs = pysrt.open(srt_file)
    segments_data = [] # Changer pour une liste de dictionnaires
    current = []
    segment_index = 1 # Pour numéroter les segments par défaut

    for sub in subs:
        current.append(sub)
        if len(current) >= 3:  # Regroupement par 3 phrases
            start_time = current[0].start.to_time()
            end_time = current[-1].end.to_time()
            start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1_000_000
            end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1_000_000
            # Ajouter un dictionnaire avec start, end, et un summary par défaut
            segments_data.append({
                "start": start_sec,
                "end": end_sec,
                "summary": f"Segment_{segment_index}" # Summary par défaut
            })
            segment_index += 1
            current = []
    # Gérer les sous-titres restants si nécessaire (optionnel)
    # if current:
    #     start_time = current[0].start.to_time()
    #     end_time = current[-1].end.to_time()
    #     start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1_000_000
    #     end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1_000_000
    #     segments_data.append({
    #         "start": start_sec,
    #         "end": end_sec,
    #         "summary": f"Segment_{segment_index}"
    #     })
    return segments_data

# Configuration AI Studio
# AI_STUDIO_PROJECT_ID = 736935593969 # Commenté ou supprimé car non supporté ici

def analyze_with_gemini(transcript, langue="en"):
    """Semantic analysis with Gemini API for precise, coherent Reels segments"""
    try:
        genai.configure(
            api_key='YOUR API KEY',
            transport='rest'
        )

        prompt = f"""Analyze the following video transcript and split it into segments that are perfectly suited for Reels (short vertical videos for social networks).
The language of the video is: {langue}
For each segment, return a JSON array strictly in the format:
[{{"start": start_sec, "end": end_sec, "summary": "Catchy 5-10 word summary in the language of the video"}}]

Constraints:
- Each segment must be between 20 and 120 seconds long.
- Each segment must start and end at a natural speech boundary (do not cut in the middle of a sentence or while someone is speaking).
- The beginning of each segment should be a hook (question, striking fact, emotion, surprise, etc).
- Each segment should form a complete, self-contained idea or story, understandable on its own.
- The summary must be in the language of the video and make people want to click.
- Segments should be suitable for viral, educational, inspiring, or entertaining content.
- Do NOT include any text outside the JSON array (no comments, no explanations).
- Timestamps must be in seconds as float, with millisecond precision (e.g. 12.345).
- Strictly respect the JSON format, with no unnecessary line breaks or extra text.

Example:
[
  {{"start": 12.123, "end": 38.456, "summary": "How to save time every morning"}},
  {{"start": 40.789, "end": 59.987, "summary": "The tip that changed my life"}}
]

Transcript:
"""
        response = genai.GenerativeModel('gemini-1.5-flash-latest').generate_content(prompt + transcript)
        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            print("Gemini returned an empty response after cleaning.")
            return "[]"
        return cleaned
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return "[]"

def extract_segments(video_path, langue="en"):
    """Extract segments with Gemini analysis"""
    srt_file = transcribe_video(video_path, language=langue)

    with open(srt_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    gemini_response = analyze_with_gemini(transcript, langue=langue)
    print(f"Réponse brute de Gemini: {gemini_response}") # Pour débogage

    try:
        segments_data = json.loads(gemini_response)
        if isinstance(segments_data, list):
             # Filtrer pour s'assurer que chaque segment a start, end et summary
             valid_segments = [
                 seg for seg in segments_data
                 if isinstance(seg, dict) and 'start' in seg and 'end' in seg and 'summary' in seg
             ]
             if not valid_segments:
                 print("Avertissement: Gemini n'a retourné aucun segment valide ou complet. Utilisation de la méthode de secours.")
                 return analyze_semantic_content(srt_file) # Méthode de secours
             # Convertir start/end en float ici pour la cohérence
             for seg in valid_segments:
                 try:
                     seg['start'] = float(seg['start'])
                     seg['end'] = float(seg['end'])
                 except (ValueError, TypeError):
                     print(f"Avertissement: Impossible de convertir start/end en float pour le segment {seg}. Il sera ignoré.")
                     # On pourrait le supprimer ou le laisser et le gérer dans extract_reels
             # Retourner la liste de dictionnaires validés
             return [seg for seg in valid_segments if isinstance(seg.get('start'), float) and isinstance(seg.get('end'), float)]
        else:
            print(f"Avertissement: Réponse de Gemini n'est pas une liste JSON valide: {segments_data}. Utilisation de la méthode de secours.")
            return analyze_semantic_content(srt_file) # Méthode de secours

    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON de la réponse Gemini: {e}. Réponse reçue: '{gemini_response}'. Utilisation de la méthode de secours.")
        return analyze_semantic_content(srt_file)
    except Exception as e:
        print(f"Erreur inattendue l'extraction des segments: {e}. Utilisation de la méthode de secours.")
        return analyze_semantic_content(srt_file)

def sanitize_filename(name):
    """Nettoie une chaîne pour l'utiliser comme nom de fichier."""
    # Supprimer les caractères invalides pour les noms de fichiers Windows/Unix
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Remplacer les espaces par des underscores
    name = name.replace(' ', '_')
    # Limiter la longueur (par exemple à 50 caractères)
    name = name[:50]
    # Supprimer les points ou espaces en fin de nom (problématique sous Windows)
    name = name.strip(' .')
    # Si le nom est vide après nettoyage, utiliser un nom par défaut
    if not name:
        name = "reel_sans_titre"
    return name

def extract_reels(video_path, output_folder="reels", progress_callback=None, langue="en", format_video="vertical"):
    os.makedirs(output_folder, exist_ok=True)
    print("Extraction des segments via Gemini (ou secours)...")

    # Vérification du format vidéo (optionnel mais recommandé)
    if not video_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        print(f"Format vidéo non supporté ou non testé : {os.path.splitext(video_path)[1]}")

    # Récupération de la durée totale avec ffprobe (via ffmpeg-python)
    try:
        print(f"Analyse de la vidéo source: {video_path}")
        probe = ffmpeg.probe(video_path)
        total_duration = float(probe['format']['duration'])
        print(f"Durée totale de la vidéo: {total_duration:.2f}s")
    except ffmpeg.Error as e:
        print(f"Erreur FFprobe lors de l'analyse de {video_path}:")
        print(e.stderr.decode()) # Afficher l'erreur FFmpeg
        return
    except Exception as e:
        print(f"Erreur lecture vidéo avec ffprobe : {str(e)}")
        return

    segments = extract_segments(video_path, langue=langue) # Récupère les segments (Gemini ou secours)

    if not segments:
        print("Aucun segment trouvé pour l'extraction.")
        return

    # Assurer que ce sont bien des dictionnaires avec start/end en float
    segments = [s for s in segments if isinstance(s, dict) and isinstance(s.get('start'), float) and isinstance(s.get('end'), float)]

    print(f"Nombre de segments (dictionnaires) reçus: {len(segments)}")

    # Tri des segments par ordre chronologique
    segments_sorted = sorted(segments, key=lambda x: x['start'])

    # Vérification des chevauchements et ajustement
    previous_end = 0.0
    cleaned_segments = []

    for segment_dict in segments_sorted:
        start = segment_dict['start']
        end = segment_dict['end']
        summary = segment_dict.get('summary', f"Segment_{len(cleaned_segments)+1}") # Utiliser summary ou défaut

        start = max(start, previous_end)
        start = min(start, total_duration)
        end = max(start, end)
        end = min(end, total_duration)

        if (end - start) < 15.0:
            end = min(start + 15.0, total_duration)

        if end > start and start < total_duration:
            # Garder le dictionnaire complet pour avoir le summary plus tard
            cleaned_segments.append({
                "start": start,
                "end": end,
                "summary": summary
            })
            previous_end = end

    print(f"Segments après nettoyage et ajustement: {len(cleaned_segments)}")
    if not cleaned_segments:
        print("Aucun segment valide à extraire après nettoyage.")
        return

    for i, segment_info in enumerate(cleaned_segments):
        start = segment_info['start']
        end = segment_info['end']
        summary = segment_info['summary']

        try:
            sanitized_summary = sanitize_filename(summary)
            targetname = os.path.join(output_folder, f"{i+1}_{sanitized_summary}.mp4")

            print(f"Extraction {i+1}: '{summary}' de {start:.2f}s à {end:.2f}s (durée: {end-start:.2f}s) -> {targetname}")

            if start >= end or start >= total_duration:
                print(f"Segment {i+1} invalide (start={start:.2f}, end={end:.2f}, total_duration={total_duration:.2f}), ignoré.")
                continue

            # --- Application du bon filtre selon le format choisi ---
            if format_video == "vertical":
                vf_filter = "scale=-1:1920,crop=1080:1920"
            elif format_video == "carré":
                vf_filter = "scale=1080:1080,crop=1080:1080"
            elif format_video == "horizontal":
                vf_filter = "scale=1920:1080,crop=1920:1080"
            else:
                vf_filter = None

            ffmpeg_args = dict(vcodec='libx264', acodec='aac', strict='experimental', avoid_negative_ts='make_zero')
            if vf_filter:
                ffmpeg_args['vf'] = vf_filter

            (
                ffmpeg
                .input(video_path, ss=start, to=end)
                .output(targetname, **ffmpeg_args)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Commande FFmpeg exécutée pour reel {i+1} (format {format_video}).")

            # Vérification simple de l'existence et taille du fichier généré
            time.sleep(0.5)
            if os.path.exists(targetname) and os.path.getsize(targetname) > 0:
                 print(f"Reel {i+1} créé avec succès: {targetname}")
            else:
                 print(f"Échec génération ou fichier vide après commande FFmpeg: {targetname}.")

            if progress_callback:
                progress_callback(i + 1, len(cleaned_segments))  # Mettre à jour la progression

        except ffmpeg.Error as e:
            print(f"Erreur FFmpeg lors de l'extraction du reel {i+1}:")
            print(f"stderr: {e.stderr.decode()}")
            if os.path.exists(targetname):
                try:
                    os.remove(targetname)
                    print(f"Fichier potentiellement corrompu supprimé: {targetname}")
                except OSError as oe:
                    print(f"Impossible de supprimer le fichier {targetname}: {oe}")
            continue
        except Exception as e:
            print(f"Erreur inattendue lors du traitement du segment {i+1} ('{summary}', {start:.2f}s - {end:.2f}s): {str(e)}")
            continue

    print("Création des reels terminée.")


def process_video(video_path, progress_callback=None, langue="en", format_video="vertical"):
    try:
        video_path = os.path.abspath(video_path.strip().strip('"'))
        print(f"Début du traitement pour: {video_path}", flush=True)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Fichier introuvable: {video_path}")

        print(f"Étape 1 & 2 : Transcription et Analyse sémantique (via Gemini ou secours)...", flush=True)
        print(f"Étape 3 : Extraction des reels...", flush=True)
        extract_reels(video_path, progress_callback=progress_callback, langue=langue, format_video=format_video)

    except Exception as e:
        print(f"ERREUR globale dans process_video: {str(e)}", flush=True)

class VideoReelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Reels Generator")
        self.geometry("800x700")  # Height increased to 700px

        # --- Language and format variables ---
        self.language_var = tk.StringVar(value="en")
        self.format_var = tk.StringVar(value="vertical")
        # -------------------------------------

        # Logo configuration
        try:
            from PIL import Image, ImageTk
            img = Image.open(r"autoshorts\Videologo.png")
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(img)
            
            logo_frame = ttk.Frame(self)
            self.logo_label = ttk.Label(logo_frame, image=self.logo_image)
            self.logo_label.pack(pady=10, anchor='center')
            logo_frame.pack(fill='x', padx=20)
            
        except Exception as e:
            self.log_message(f"Logo loading error: {str(e)}")
        
        # Main widgets
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Language selection menu ---
        lang_label = ttk.Label(main_frame, text="Video language:")
        lang_label.pack(pady=2)
        lang_combo = ttk.Combobox(main_frame, textvariable=self.language_var, values=["en", "fr", "ar", "es", "de"], state="readonly")
        lang_combo.pack(pady=2)
        # -------------------------------

        # --- Output format selection menu ---
        format_label = ttk.Label(main_frame, text="Output format:")
        format_label.pack(pady=2)
        format_combo = ttk.Combobox(main_frame, textvariable=self.format_var, values=["vertical", "square", "horizontal"], state="readonly")
        format_combo.pack(pady=2)
        # ------------------------------------

        self.file_label = ttk.Label(main_frame, text="No file selected")
        self.select_btn = ttk.Button(main_frame, text="Select a video", command=self.select_file)
        self.output_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=12)
        
        # Bottom fixed frame
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.progress = ttk.Progressbar(bottom_frame, mode='determinate')
        self.start_btn = ttk.Button(bottom_frame, text="Start processing", command=self.start_processing)

        # Layout
        self.file_label.pack(pady=5)
        self.select_btn.pack(pady=5)
        self.output_text.pack(pady=5, fill=tk.BOTH, expand=True)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

    def log_message(self, message):
        """Display messages with timestamp and force update"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.output_text.insert(tk.END, full_msg + "\n")
        self.output_text.see(tk.END)
        self.update_idletasks()  # Force immediate UI update

    def start_processing(self):
        def run():
            import sys
            from io import StringIO
            sys.stdout = StringIO()
            
            try:
                video_path = self.file_label.cget("text")
                if not video_path or video_path == "No file selected":
                    self.log_message("Error: No file selected!")
                    return

                langue = self.language_var.get()
                format_video = self.format_var.get()

                self.progress['value'] = 0
                self.log_message(f"Processing started for: {video_path}")
                self.log_message(f"Selected language: {langue}")
                self.log_message(f"Selected format: {format_video}")

                def progress_wrapper(current, total):
                    self.after(0, self.update_progress, current, total)
                    self.after(0, self.flush_stdout)

                langue = self.language_var.get()
                process_video(video_path, progress_wrapper, langue, format_video)
                self.flush_stdout()

            except Exception as e:
                self.log_message(f"CRITICAL ERROR: {str(e)}")
            finally:
                self.progress.stop()

        Thread(target=run, daemon=True).start()

    def flush_stdout(self):
        """Regularly flush the output buffer"""
        import sys
        from io import StringIO
        
        output = sys.stdout.getvalue()
        if output:
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line)
            sys.stdout = StringIO()  # Reset buffer

    def update_progress(self, current, total):
        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = total
        self.progress['value'] = current
        self.log_message(f"Progress: {current}/{total} segments processed")

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_label.config(text=file_path)

if __name__ == "__main__":
    app = VideoReelApp()
    app.mainloop()