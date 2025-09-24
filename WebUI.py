import gradio as gr
import os
import shutil
from PIL import Image
import face_recognition
import pickle
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture
import python_speech_features

IMAGE_DIR = "captured_images"
AUDIO_DIR = "voice_database"
GMM_DIR = "./gmm_models"


def save_user_data(name, img1, img2, img3, audio1, audio2, audio3):
    if not name:
        return "❌ Please provide a user name."

    images = [img1, img2, img3]
    audios = [audio1, audio2, audio3]

    if all(i is None for i in images) and all(a is None for a in audios):
        return "❌ Please provide at least one image or one voice recording to update."

    user_img_dir = os.path.join(IMAGE_DIR, name)
    user_audio_dir = os.path.join(AUDIO_DIR, name)

    os.makedirs(user_img_dir, exist_ok=True)
    os.makedirs(user_audio_dir, exist_ok=True)

    # Save images
    for i, img in enumerate(images, 1):
        if img is not None:
            img.save(os.path.join(user_img_dir, f"img{i}.png"))

    # Save audio
    for i, audio in enumerate(audios, 1):
        if audio is not None:
            sf.write(os.path.join(user_audio_dir, f"audio{i}.wav"), audio[1], audio[0])

    # Save face encodings
    encodings = []
    for fname in os.listdir(user_img_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(user_img_dir, fname)
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if enc:
                encodings.append(enc[0])
    if encodings:
        with open(os.path.join(user_img_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(encodings, f)

    # Prepare data for GMM training if audio was uploaded
    voice_data = [audio for audio in audios if audio is not None]
    if voice_data:
        train_gmm_model(name, voice_data)

    return f"✅ User '{name}' added successfully!"



def refresh_inputs():
    return [None] * 6 + [""]

def load_image_fallback(folder, *bases):
    exts = [".png", ".jpg", ".jpeg"]
    filenames = os.listdir(folder)
    for base in bases:
        for ext in exts:
            for f in filenames:
                if f.lower() == base.lower() + ext:
                    return Image.open(os.path.join(folder, f))
    # Try fuzzy fallback if needed
    for f in filenames:
        if any(base.lower() in f.lower() for base in bases) and f.lower().endswith(tuple(exts)):
            return Image.open(os.path.join(folder, f))
    return None


def load_audio_fallback(folder, *bases):
    exts = [".wav", ".mp3", ".flac"]
    filenames = os.listdir(folder)
    for base in bases:
        for ext in exts:
            for f in filenames:
                if f.lower() == base.lower() + ext:
                    path = os.path.join(folder, f)
                    data, rate = sf.read(path)
                    return (rate, data)
    # Fuzzy fallback if exact match not found
    for f in filenames:
        if any(base.lower() in f.lower() for base in bases) and f.lower().endswith(tuple(exts)):
            path = os.path.join(folder, f)
            data, rate = sf.read(path)
            return (rate, data)
    return None


def view_user(name):
    img_dir = os.path.join(IMAGE_DIR, name)
    audio_dir = os.path.join(AUDIO_DIR, name)

    if not os.path.isdir(img_dir) or not os.path.isdir(audio_dir):
        return [None] * 6 + [f"❌ User '{name}' not found."]

    img1 = load_image_fallback(img_dir, "img1", "1")
    img2 = load_image_fallback(img_dir, "img2", "2")
    img3 = load_image_fallback(img_dir, "img3", "3")

    audio1 = load_audio_fallback(audio_dir, "audio1", "1", "voice1")
    audio2 = load_audio_fallback(audio_dir, "audio2", "2", "voice2")
    audio3 = load_audio_fallback(audio_dir, "audio3", "3", "voice3")


    return [img1, img2, img3, audio1, audio2, audio3, f"👁️ User '{name}' data loaded."]

def delete_user(name):
    img_dir = os.path.join(IMAGE_DIR, name)
    audio_dir = os.path.join(AUDIO_DIR, name)
    gmm_path = os.path.join(GMM_DIR, f"{name}.gmm")  # Path to GMM file

    deleted = False
    
    # Delete image directory
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
        deleted = True
    
    # Delete audio directory
    if os.path.isdir(audio_dir):
        shutil.rmtree(audio_dir)
        deleted = True
    
    # Delete GMM model file if exists
    if os.path.exists(gmm_path):
        try:
            os.remove(gmm_path)
            deleted = True
        except Exception as e:
            print(f"Error deleting GMM file: {e}")

    return f"🗑️ User '{name}' deleted." if deleted else f"❌ User '{name}' not found."


def update_user(name, img1, img2, img3, audio1, audio2, audio3):
    return save_user_data(name, img1, img2, img3, audio1, audio2, audio3)


def list_registered_users():
    if not os.path.exists(IMAGE_DIR):
        return "❌ No users found."

    users = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    return "\n".join(users) if users else "❌ No users found."

def add_user(name, img1, img2, img3, audio1, audio2, audio3):
    return save_user_data(name, img1, img2, img3, audio1, audio2, audio3)

def clear_inputs():
    return None, None, None, None, None, None, "", "", ""



def get_user_choices():
    if not os.path.exists(IMAGE_DIR):
        return []
    return [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]

# ---------------------- Voice & Face Matching ----------------------

def extract_mfcc_from_numpy(audio_data, sample_rate):
    try:
        mfcc_feat = python_speech_features.mfcc(audio_data, sample_rate, winlen=0.025, winstep=0.01, numcep=13,
                                                nfilt=26, nfft=2048, lowfreq=0, highfreq=None, preemph=0.97,
                                                ceplifter=22, appendEnergy=True)
        return mfcc_feat
    except Exception as e:
        print(f"MFCC extraction error: {e}")
        return None

def match_face(test_image):
    test_img = np.array(test_image)
    test_encoding = face_recognition.face_encodings(test_img)
    if not test_encoding:
        return None
    test_encoding = test_encoding[0]

    for user in os.listdir(IMAGE_DIR):
        pkl_path = os.path.join(IMAGE_DIR, user, f"{user}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                known_encodings = pickle.load(f)
                results = face_recognition.compare_faces(known_encodings, test_encoding, tolerance=0.5)
                if any(results):
                    return user
    return None

def match_voice(audio_numpy):
    """Returns (username, score) or (None, score) if no match"""
    if not os.path.exists(GMM_DIR):
        return None, -float('inf')
        
    models = []
    speakers = []
    
    try:
        # Load all GMM models
        for file in os.listdir(GMM_DIR):
            if file.endswith(".gmm"):
                with open(os.path.join(GMM_DIR, file), "rb") as f:
                    models.append(pickle.load(f))
                    speakers.append(file.replace(".gmm", ""))
                    
        # Extract features
        mfcc = extract_mfcc_from_numpy(audio_numpy[1], audio_numpy[0])
        if mfcc is None:
            return None, -float('inf')
            
        # Find best match
        best_score = -float('inf')
        best_speaker = None
        
        for i, model in enumerate(models):
            try:
                score = model.score(mfcc).sum()
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_speaker = speakers[i]
            except:
                continue
                
        return (best_speaker, best_score) if best_score > -60 else (None, best_score)
        
    except Exception as e:
        print(f"Voice matching error: {e}")
        return None, -float('inf')


def train_gmm_model(name, audio_files):
    """Adapted for Gradio's audio input format"""
    features = []
    
    for audio in audio_files:
        if audio is None:
            continue
            
        # Gradio provides (sample_rate, audio_data) tuple
        sample_rate, audio_data = audio
        
        # Extract MFCC features using same parameters as Jupyter version
        mfcc = python_speech_features.mfcc(
            audio_data, sample_rate,
            winlen=0.025, winstep=0.01, numcep=13,
            nfilt=26, nfft=2048, lowfreq=0,
            highfreq=None, preemph=0.97,
            ceplifter=22, appendEnergy=True
        )
        if mfcc is not None:
            features.append(mfcc)

    if not features:
        return False

    # Same GMM training as Jupyter
    X = np.vstack(features)
    gmm = GaussianMixture(
        n_components=16,
        covariance_type='diag',
        n_init=3
    )
    gmm.fit(X)

    os.makedirs(GMM_DIR, exist_ok=True)
    with open(f"{GMM_DIR}/{name}.gmm", "wb") as f:
        pickle.dump(gmm, f)
    
    return True


def recognize_voice(audio_input, selected_user=None):
    """Handles both identification and verification"""
    if audio_input is None:
        return None, "❌ No audio input"
        
    sample_rate, audio_data = audio_input
    
    # 1. Feature extraction (same as training)
    mfcc = python_speech_features.mfcc(
        audio_data, sample_rate,
        winlen=0.025, winstep=0.01, numcep=13,
        nfilt=26, nfft=2048, lowfreq=0,
        highfreq=None, preemph=0.97,
        ceplifter=22, appendEnergy=True
    )
    
    # 2. Verification mode (if user specified)
    if selected_user:
        model_path = f"{GMM_DIR}/{selected_user}.gmm"
        if not os.path.exists(model_path):
            return False, "❌ No voice model for user"
            
        with open(model_path, "rb") as f:
            gmm = pickle.load(f)
            
        score = gmm.score(mfcc).sum()
        is_match = score > -60  # Your threshold
        return is_match, f"Score: {score:.2f} ({'Match' if is_match else 'No match'})"
    
    # 3. Identification mode (find best match)
    else:
        best_score = -float('inf')
        best_user = None
        
        for model_file in os.listdir(GMM_DIR):
            if model_file.endswith(".gmm"):
                user = model_file[:-4]
                with open(f"{GMM_DIR}/{model_file}", "rb") as f:
                    gmm = pickle.load(f)
                    score = gmm.score(mfcc).sum()
                    if score > best_score:
                        best_score = score
                        best_user = user
                        
        return best_user, f"Best match: {best_user} (Score: {best_score:.2f})"
    voice_result = recognize_voice(voice_audio)
    voice_user = voice_result[0] if voice_result else None
    


def authenticate_user(selected_user, face_image, voice_audio):
    """Verify biometrics against specific user"""
    if not selected_user:
        return "❌ Please select a user first"
    
    threshold = -60  # Your voice threshold
    results = []
    
    # Face verification
    face_match = False
    if face_image:
        face_user = match_face(face_image)
        face_match = face_user == selected_user if face_user else False
        face_status = f"👤 Face: {'✅ Match' if face_match else '❌ No match'}"
        results.append(face_status)
    
    # Voice verification (now checks score directly)
    voice_match = False
    voice_score = -float('inf')  # Default if no audio provided
    if voice_audio:
        _, voice_score = match_voice(voice_audio)  # Ignore username, just check score
        voice_match = voice_score > threshold
        diff = voice_score - threshold
        closeness = f"{abs(diff):.2f} {'above' if diff >= 0 else 'below'}"
        voice_status = (f"🎙️ Voice: {'✅ Match' if voice_match else '❌ No match'} | "
                       f"Score: {voice_score:.2f} ({closeness} threshold)")
        results.append(voice_status)
    
    # Decision logic
    if face_match and voice_match:
        verdict = f"✅ AUTHENTICATED as {selected_user}"
    elif face_match or voice_match:
        partial = []
        if face_match: partial.append("face")
        if voice_match: partial.append("voice")
        verdict = f"⚠️ PARTIAL: Matched {' & '.join(partial)} only"
    else:
        verdict = "❌ FAILED: No biometrics matched"
    
    return f"{verdict}\n\n" + "\n".join(results)


# ---------------------- UI ----------------------

with gr.Blocks(title="User Registration System") as demo:
    gr.Markdown("## 👤 Biometric Admin Panel")

    with gr.Row():
        name_input = gr.Textbox(label="🆕 New User's Name")
        add_btn = gr.Button("➕ Add User")
        view_btn = gr.Button("👁️ View User")
        del_btn = gr.Button("🗑️ Delete User")
        refresh_btn = gr.Button("🔄 Refresh")
        view_users_btn = gr.Button("👥 View Users")
        user_dropdown = gr.Dropdown(label="📂 Select Existing User", choices=[], interactive=True)

    with gr.Row():
        img1 = gr.Image(label="📸 Face Image 1", type="pil", sources=["webcam"])
        img2 = gr.Image(label="📸 Face Image 2", type="pil", sources=["webcam"])
        img3 = gr.Image(label="📸 Face Image 3", type="pil", sources=["webcam"])

    with gr.Row():
        audio1 = gr.Audio(label="🎙️ Voice 1", type="numpy", sources=["microphone"])
        audio2 = gr.Audio(label="🎙️ Voice 2", type="numpy", sources=["microphone"])
        audio3 = gr.Audio(label="🎙️ Voice 3", type="numpy", sources=["microphone"])

    output = gr.Textbox(label="Status")
    users_list_output = gr.Textbox(label="📋 Registered Users", lines=10, interactive=False)

    # --- Authentication Section ---
    gr.Markdown("## 🔐 Authenticate Existing User")
    with gr.Row():
        test_face = gr.Image(label="🧪 Test Face", type="pil", sources=["webcam"])
        test_voice = gr.Audio(label="🧪 Test Voice", type="numpy", sources=["microphone"])
        auth_btn = gr.Button("🔐 Authenticate User") 
        auth_output = gr.Textbox(label="Authentication Result")
        update_btn = gr.Button("🔄 Update User") 


    add_btn.click(add_user, [name_input, img1, img2, img3, audio1, audio2, audio3], output).then(lambda: gr.update(choices=get_user_choices()), None, user_dropdown)
    del_btn.click(delete_user, user_dropdown, output).then(lambda: gr.update(choices=get_user_choices()), None, user_dropdown)
    refresh_btn.click(clear_inputs,None,[img1, img2, img3, audio1, audio2, audio3, name_input, output, users_list_output]).then(lambda: gr.update(choices=get_user_choices()), None, user_dropdown)
    view_btn.click(view_user, user_dropdown, [img1, img2, img3, audio1, audio2, audio3, output]).then(lambda name: gr.update(value=name), user_dropdown, name_input)
    view_users_btn.click(list_registered_users, None, users_list_output)
    auth_btn.click(authenticate_user,inputs=[user_dropdown, test_face, test_voice], outputs=auth_output )
    update_btn.click(update_user,[name_input, img1, img2, img3, audio1, audio2, audio3],output).then(get_user_choices, None, user_dropdown)
    
    demo.load(lambda: gr.update(choices=get_user_choices()), outputs=user_dropdown)

# ---------------------- Launch App ----------------------
demo.launch()
