import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
from datetime import datetime

# Configuration Streamlit
st.set_page_config(
    page_title="NeuroScan AI - Diagnostic Visionnaire",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Cyberpunk
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');

    * { 
        font-family: 'Exo 2', sans-serif; 
    }

    .main-header { 
        font-family: 'Orbitron', monospace; 
        font-size: 3.5rem; 
        background: linear-gradient(135deg, #00ff88 0%, #00ccff 50%, #ff00ff 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-align: center; 
        margin-bottom: 1rem; 
        font-weight: 900; 
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3); 
        letter-spacing: 2px; 
    }

    .cyber-subtitle { 
        font-family: 'Orbitron', monospace; 
        text-align: center; 
        color: #00ff88; 
        font-size: 1.3rem; 
        margin-bottom: 3rem; 
        font-weight: 400; 
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); 
    }

    .cyber-card { 
        background: rgba(10, 10, 10, 0.9); 
        backdrop-filter: blur(10px); 
        border-radius: 15px; 
        padding: 25px; 
        margin: 15px 0; 
        border: 1px solid rgba(0, 255, 136, 0.3); 
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1); 
        position: relative; 
        overflow: hidden; 
    }

    .cyber-card::before { 
        content: ''; 
        position: absolute; 
        top: 0; 
        left: -100%; 
        width: 100%; 
        height: 100%; 
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.1), transparent); 
        transition: left 0.5s ease; 
    }

    .cyber-card:hover::before { 
        left: 100%; 
    }

    .neural-card { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%); 
        color: white; 
        padding: 25px; 
        border-radius: 15px; 
        margin: 15px 0; 
        border: 1px solid rgba(0, 255, 136, 0.5); 
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.2); 
    }

    .matrix-upload { 
        border: 2px dashed #00ff88; 
        border-radius: 20px; 
        padding: 40px; 
        text-align: center; 
        background: rgba(0, 255, 136, 0.05); 
        transition: all 0.3s ease; 
        position: relative; 
        overflow: hidden; 
    }

    .matrix-upload::before { 
        content: ''; 
        position: absolute; 
        top: 0; 
        left: 0; 
        right: 0; 
        height: 2px; 
        background: linear-gradient(90deg, transparent, #00ff88, transparent); 
        animation: matrixScan 2s linear infinite; 
    }

    @keyframes matrixScan { 
        0% { transform: translateX(-100%); } 
        100% { transform: translateX(100%); } 
    }

    .cyber-metric { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); 
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        border: 1px solid rgba(0, 255, 136, 0.3); 
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.1); 
        transition: all 0.3s ease; 
    }

    .cyber-metric:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 5px 25px rgba(0, 255, 136, 0.2); 
    }

    .cyber-button { 
        background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%); 
        color: #0a0a0a !important; 
        font-weight: 700; 
        border: none; 
        padding: 12px 30px; 
        border-radius: 8px; 
        font-family: 'Orbitron', monospace; 
        letter-spacing: 1px; 
        transition: all 0.3s ease; 
        width: 100%; 
    }

    .cyber-button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.4); 
    }

    .pulse-glow { 
        animation: pulseGlow 2s ease-in-out infinite; 
    }

    @keyframes pulseGlow { 
        0% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); } 
        50% { box-shadow: 0 0 30px rgba(0, 255, 136, 0.6); } 
        100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); } 
    }

    .status-normal { 
        color: #00ff88; 
        font-weight: bold; 
    }

    .status-warning { 
        color: #ffaa00; 
        font-weight: bold; 
    }

    .status-alert { 
        color: #ff0066; 
        font-weight: bold; 
    }

    .risk-low { color: #00ff88; }
    .risk-medium { color: #ffaa00; }
    .risk-high { color: #ff0066; }

    .scan-animation {
        animation: scanLine 3s ease-in-out infinite;
    }

    @keyframes scanLine {
        0% { transform: translateY(-100%); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateY(100%); opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle
MODEL_PATH = "models/lung_model.h5"
CLASS_NAMES = ["COVID-19", "Normal", "Tuberculosis"]
CLASS_DESCRIPTIONS = {
    "COVID-19": "Infection virale respiratoire nécessitant une attention médicale urgente",
    "Normal": "Aucune anomalie pulmonaire détectée",
    "Tuberculosis": "Infection bactérienne pulmonaire nécessitant un traitement spécifique"
}

if not os.path.exists(MODEL_PATH):
    st.error("❌ Modèle non trouvé ! Veuillez placer lung_model.h5 dans le dossier models.")
    model = None
else:
    model = load_model(MODEL_PATH)


# Fonctions
def preprocess_image(image_pil):
    """Prétraite l'image pour l'analyse IA"""
    img = image_pil.resize((224, 224))
    img_array = np.array(img) / 255.0

    if len(img_array.shape) == 2:  # Image en niveaux de gris
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # Image RGBA
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def create_cyber_radar(probabilities):
    """Crée un radar graphique des probabilités"""
    categories = list(probabilities.keys())
    values = [probabilities[cat] * 100 for cat in categories]
    values += values[:1]
    categories_display = [cat.upper() for cat in categories] + [categories[0].upper()]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    ax.plot(angles, values, 'o-', linewidth=2, color='#00ff88', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#00ff88')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_display[:-1], color='#00ccff', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#ff00ff', fontsize=9)
    ax.grid(True, color='#00ccff', alpha=0.3, linestyle='--')
    ax.set_title('CARTE DES PROBABILITÉS', color='#00ff88', fontsize=14, pad=20, fontweight='bold')

    plt.tight_layout()
    return fig


def create_confidence_meter(confidence):
    """Crée une jauge de confiance"""
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    progress = confidence * 100

    # Barre de fond
    ax.barh(0, 100, color='#1a1a2e', alpha=0.7, height=0.3)

    # Barre de progression
    color = '#00ff88' if progress > 70 else '#ffaa00' if progress > 50 else '#ff0066'
    ax.barh(0, progress, color=color, alpha=0.9, height=0.3)

    # Texte
    ax.text(50, 0, f"CONFIDENCE: {progress:.1f}%",
            ha='center', va='center', color='#ffffff',
            fontsize=12, fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.tight_layout()

    return fig


def get_risk_level(prediction, confidence):
    """Détermine le niveau de risque"""
    if prediction == "Normal":
        return "low", "🟢 FAIBLE RISQUE"
    elif confidence > 0.8:
        return "high", "🔴 RISQUE ÉLEVÉ"
    else:
        return "medium", "🟡 RISQUE MODÉRÉ"


def create_progress_bar():
    """Crée une barre de progression animée"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for percent_complete in range(0, 101, 10):
        status_text.text(f"🔍 Analyse en cours... {percent_complete}%")
        progress_bar.progress(percent_complete)
        time.sleep(0.1)

    return progress_bar, status_text


def display_results(result, image_pil):
    """Affiche les résultats de l'analyse"""
    st.markdown("---")

    # En-tête des résultats
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])

    with col_header1:
        st.markdown(f"### 📋 RAPPORT D'ANALYSE - {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    with col_header2:
        risk_level, risk_text = get_risk_level(result['prediction'], result['confidence'])
        st.markdown(
            f"<div class='risk-{risk_level}' style='text-align: center; font-size: 16px; font-weight: bold;'>NIVEAU DE RISQUE: {risk_text}</div>",
            unsafe_allow_html=True)

    with col_header3:
        st.markdown(f"<div style='text-align: center; color: #00ccff; font-size: 14px;'>ID: NS{int(time.time())}</div>",
                    unsafe_allow_html=True)

    # Métriques principales
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 CONFIDENCE")
        st.pyplot(create_confidence_meter(result['confidence']))

    with col2:
        st.markdown("#### 🩺 DIAGNOSTIC")
        is_normal = result['prediction'] == 'Normal'
        status_class = "status-normal" if is_normal else "status-alert"
        status_text = "NORMAL" if is_normal else "ANOMALIE DÉTECTÉE"

        st.markdown(f"""
        <div class='cyber-card' style='text-align:center; height:120px; display:flex; flex-direction:column; justify-content:center;'>
            <div style='font-size:24px; font-weight:bold; color:#00ff88;'>{result['prediction']}</div>
            <div class='{status_class}' style='margin-top:10px;'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Description de la condition
        st.markdown(f"""
        <div class='cyber-card' style='text-align:center; padding:15px;'>
            <div style='color:#00ccff; font-size:12px;'>{CLASS_DESCRIPTIONS[result['prediction']]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### 📊 CARTE IA")
        st.pyplot(create_cyber_radar(result['probabilities']))

    # Détails des probabilités et recommandations
    col_details, col_recommendations = st.columns([2, 1])

    with col_details:
        st.markdown("#### 📈 ANALYSE DÉTAILLÉE")
        prob_cols = st.columns(len(result['probabilities']))

        for idx, (class_name, prob) in enumerate(result['probabilities'].items()):
            with prob_cols[idx]:
                percentage = prob * 100
                color = "#00ff88" if class_name == result['prediction'] else "#ff00ff"

                st.markdown(f"""
                <div class='cyber-metric'>
                    <div style='font-size:14px; color:#00ccff;'>{class_name.upper()}</div>
                    <div style='font-size:24px; color:{color}; font-weight:bold;'>{percentage:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    with col_recommendations:
        st.markdown("#### 💡 RECOMMANDATIONS")

        if result['prediction'] == 'Normal':
            recommendations = [
                "✅ Continuer les examens de routine",
                "✅ Maintenir une hygiène de vie saine",
                "✅ Consulter annuellement pour suivi"
            ]
        else:
            recommendations = [
                "🚨 Consultation médicale urgente recommandée",
                "🔬 Examens complémentaires nécessaires",
                "💊 Traitement spécialisé requis",
                "📋 Suivi médical rapproché"
            ]

        for rec in recommendations:
            st.markdown(f"<div class='cyber-card' style='padding:15px; margin:10px 0;'>{rec}</div>",
                        unsafe_allow_html=True)


# Interface principale
st.markdown('<h1 class="main-header">NEURO SCAN AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="cyber-subtitle">DIAGNOSTIC PULMONAIRE PAR INTELLIGENCE ARTIFICIELLE</p>', unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 🔧 PANEL DE CONTRÔLE")

    st.markdown("""
    <div class='cyber-card'>
        <h4>📊 STATISTIQUES MODÈLE</h4>
        <p>🎯 Précision: 97.1%</p>
        <p>⚡ Performance: Optimale</p>
        <p>🔄 Dernière mise à jour: Aujourd'hui</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='cyber-card'>
        <h4>ℹ️ GUIDE RAPIDE</h4>
        <p>• Téléchargez une radiographie pulmonaire</p>
        <p>• Format: PNG, JPG, JPEG</p>
        <p>• Résolution recommandée: 224x224px</p>
        <p>• Temps d'analyse: ~5 secondes</p>
    </div>
    """, unsafe_allow_html=True)

# Zone d'upload principale
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="matrix-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 GLISSEZ-DÉPOSEZ VOTRE RADIOGRAPHIE",
        type=['png', 'jpg', 'jpeg'],
        help="Formats supportés: PNG, JPG, JPEG"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown("""
    <div class='cyber-card'>
        <h4>👁️ VISUALISEUR D'IMAGE</h4>
        <p>L'image chargée apparaîtra ici avec:</p>
        <p>• Aperçu haute qualité</p>
        <p>• Métadonnées techniques</p>
        <p>• Prévisualisation avant analyse</p>
    </div>
    """, unsafe_allow_html=True)

# Analyse
if uploaded_file is not None:
    # Affichage de l'image avec métadonnées
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_meta = st.columns([2, 1])

    with col_img:
        st.image(image, caption="🖼️ RADIOGRAPHIE CHARGÉE", use_container_width=True)

    with col_meta:
        st.markdown("""
        <div class='cyber-card'>
            <h4>📄 MÉTADONNÉES</h4>
            <p>📏 Dimensions: {}x{}</p>
            <p>🎨 Format: {}</p>
            <p>💾 Taille: {:.1f} KB</p>
            <p>🕒 Date: {}</p>
        </div>
        """.format(
            image.size[0], image.size[1],
            uploaded_file.type,
            len(uploaded_file.getvalue()) / 1024,
            datetime.now().strftime("%d/%m/%Y %H:%M")
        ), unsafe_allow_html=True)

    # Bouton d'analyse
    if st.button("🚀 LANCER L'ANALYSE IA", use_container_width=True):
        if model is None:
            st.error("❌ Modèle non chargé. Vérifiez le fichier lung_model.h5")
        else:
            with st.spinner("🔍 ANALYSE EN COURS... NEUROSCAN TRAITE LES DONNÉES..."):
                try:
                    # Barre de progression animée
                    progress_bar, status_text = create_progress_bar()

                    # Prétraitement et prédiction
                    img_array = preprocess_image(image)
                    preds = model.predict(img_array, verbose=0)

                    class_idx = np.argmax(preds[0])
                    confidence = preds[0][class_idx]

                    # Résultats
                    result = {
                        'prediction': CLASS_NAMES[class_idx],
                        'confidence': confidence,
                        'probabilities': {name: float(p) for name, p in zip(CLASS_NAMES, preds[0])}
                    }

                    # Nettoyage de la barre de progression
                    progress_bar.empty()
                    status_text.empty()

                    # Affichage des résultats
                    display_results(result, image)

                    # Message final
                    if result['prediction'] == 'Normal':
                        st.success("✅ ANALYSE TERMINÉE - ÉTAT PULMONAIRE NORMAL DÉTECTÉ")
                    else:
                        st.warning(
                            f"⚠️ ANALYSE TERMINÉE - {result['prediction'].upper()} DÉTECTÉ - CONSULTEZ UN MÉDECIN")

                except Exception as e:
                    st.error(f"❌ ERREUR D'ANALYSE: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #00ccff; font-size: 12px;'>"
    "NEURO SCAN AI - SYSTÈME DE DIAGNOSTIC MÉDICAL ASSISTÉ PAR IA | "
    "⚠️ CET OUTIL NE REMPLACE PAS UN DIAGNOSTIC MÉDICAL PROFESSIONNEL | "
    f"VERSION 2.0 - {datetime.now().year}"
    "</div>",
    unsafe_allow_html=True
)