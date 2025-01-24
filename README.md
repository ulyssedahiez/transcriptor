
# Transcription Audio avec Diarisation et Whisper

Ce projet permet de transcrire des fichiers audio avec identification des locuteurs grâce aux modèles **PyAnnote** et **Whisper**. Il segmente l'audio, applique une diarisation (identification des locuteurs), puis génère une transcription associée à chaque locuteur.

## Fonctionnalités
- Découpe des fichiers audio en segments configurables.
- Identification des locuteurs (diarisation).
- Transcription des segments avec Whisper.
- Sauvegarde des transcriptions dans un fichier texte avec les locuteurs identifiés.

---

## Prérequis

### 1. Installation de Python
Le projet fonctionne avec **Python 3.10** ou une version supérieure. Assurez-vous que Python est installé sur votre machine :

```bash
python --version
```

Si ce n'est pas le cas, téléchargez et installez Python depuis le site officiel : [https://www.python.org/downloads/](https://www.python.org/downloads/).

### 2. Installation des dépendances Python
Les bibliothèques nécessaires sont listées ci-dessous. Installez-les avec `pip` :

```bash
pip install setuptools==59.5.0
pip install torch==2.0.1+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install speechbrain==0.5.16 pyannote.audio==3.1.1
pip install faster-whisper==0.10.0
pip install pydub==0.25.1
pip install numpy==1.24.3 protobuf==3.20.3
```

> **Note** : Si vous utilisez un GPU NVIDIA, installez la version CUDA adaptée à votre matériel. Pour CPU, remplacez `+cu118` par `cpu` dans la commande Torch.

### 3. Installation de FFmpeg
FFmpeg est requis pour manipuler les fichiers audio. Installez-le en fonction de votre système d'exploitation.

- **Sur macOS** :
  ```bash
  brew install ffmpeg
  ```

- **Sur Linux (Ubuntu/Debian)** :
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **Sur Windows** :
  1. Téléchargez FFmpeg depuis [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
  2. Extrayez les fichiers téléchargés.
  3. Ajoutez le chemin du dossier `bin` à la variable d'environnement PATH.

Vérifiez l'installation avec :
```bash
ffmpeg -version
```

---

## Configuration

### 1. Obtenir une clé API HuggingFace
Le modèle de diarisation **PyAnnote** nécessite une clé API HuggingFace.

#### Étapes pour obtenir une clé :
1. Créez un compte sur [https://huggingface.co/join](https://huggingface.co/join).
2. Accédez à vos paramètres de compte : [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Cliquez sur "New token" et générez un token avec des permissions d'accès aux modèles (gated repositories).
4. Copiez le token généré.

---

### 2. Configuration de la clé API
La clé API doit être ajoutée dans le fichier principal du script. Ouvrez le fichier et remplacez la ligne suivante :

```python
HF_TOKEN = ""
```

par votre clé API :

```python
HF_TOKEN = "votre_clé_API_HuggingFace"
```

---

## Utilisation

### 1. Préparer le fichier audio
Assurez-vous que le fichier audio est dans un format compatible, comme `.wav`, `.mp3`, ou `.ogg`.

### 2. Lancer le script
Exécutez le script en ligne de commande en fournissant le chemin du fichier audio :

```bash
python votre_script.py
```

Par exemple :
```bash
python main.py
```

Ensuite, entrez le chemin de votre fichier audio lorsqu'il est demandé.

---

## Résultats

- **Segments audio temporaires** : Les segments audio sont créés dans un sous-dossier `segments` dans le même répertoire que le fichier audio original.
- **Fichier de transcription** : Une fois le processus terminé, un fichier `transcriptions.txt` est généré dans le même dossier que le fichier audio. Ce fichier contient la transcription associée à chaque locuteur, formatée comme suit :

```
Speaker 1: Bonjour, comment allez-vous ?

Speaker 2: Très bien, merci.
```

---

## Nettoyage

Après l'exécution du script, les segments temporaires sont automatiquement supprimés. Si vous souhaitez les conserver, commentez cette partie du code dans le script :

```python
os.remove(segment_path)
os.rmdir(segments_dir)
```

---

## Dépannage

### 1. Le script ne trouve pas `torch` ou une autre dépendance :
Assurez-vous d'avoir installé toutes les dépendances avec les bonnes versions.

### 2. `FFmpeg` n'est pas reconnu :
Vérifiez que FFmpeg est correctement installé et que son chemin est ajouté à votre variable d'environnement PATH.

### 3. Le script est lent :
Le traitement peut être long pour des fichiers audio volumineux. Assurez-vous d'utiliser un GPU si possible (CUDA).

---

## Exemple d'exécution

1. Préparez un fichier audio, par exemple : `mon_fichier.mp3`.
2. Exécutez le script :
   ```bash
   python main.py
   ```
3. Entrez le chemin du fichier audio :
   ```
   Chemin du fichier audio : /chemin/vers/mon_fichier.mp3
   ```
4. Une fois terminé, le fichier `transcriptions.txt` est généré dans le même dossier que votre fichier audio.

---

### À propos

Ce projet utilise :
- **PyAnnote** pour la diarisation (identification des locuteurs).
- **Faster-Whisper** pour la transcription rapide et précise.

