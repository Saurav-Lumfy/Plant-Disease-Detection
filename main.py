import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sqlite3
from passlib.hash import pbkdf2_sha256
import datetime
import os
import time
from database import init_db
import random
import time
from passlib.hash import pbkdf2_sha256

# Initialize database
init_db()
conn = sqlite3.connect('plant_disease.db')
c = conn.cursor()

# Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        background: black;
    }
    .tab-content {
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
        margin-top: 10px;
    }
    .chat-container {
        background-color: black;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .disease-card {
        background-color: black;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @media (max-width: 768px) {
        .col-adaptive {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Trained_Model.h5')

model = load_model()

# Class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
    'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Session state management
session_defaults = {
    'user': None,
    'page': 'Home',
    'latest_prediction': None,
    'notification_shown': False,
    'uploaded_file': None,
    'processing': False,
    'prediction_done': False,
    'analyze_clicked': False,
    'chatbot_step': 'plant_selection',
    'selected_plant': None,
    'selected_disease': None
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Disease information database
DISEASE_INFO = {
    # Apple Diseases
    'Apple___Apple_scab': {
    'description': "Apple scab is a fungal disease caused by Venturia inaequalis that primarily affects apple leaves, fruits, and blossoms, resulting in olive-green to black lesions. The disease weakens the plant and reduces fruit quality and yield. Severe infections can cause premature leaf drop and deformed fruits, making them unmarketable.",
    'reasons': [
        "Fungus Venturia inaequalis: Spores overwinter in fallen leaves and produce primary spores (ascospores) in spring, which infect young leaves and fruit.",
        "Cool, wet weather: Ideal conditions for infection occur when temperatures range between 50°F to 75°F (10°C to 24°C) with prolonged leaf wetness or high humidity.",
        "Poor air circulation: Dense foliage traps moisture, providing a favorable environment for fungal growth.",
        "Susceptible apple varieties: Cultivars with low resistance are more prone to scab infection."
    ],
    'precautions': [
        "Remove fallen leaves and plant debris: Rake and destroy infected leaves in the fall to prevent the fungus from overwintering.",
        "Apply dormant sprays: Use lime sulfur or copper-based sprays during dormancy to kill overwintering spores.",
        "Plant resistant varieties: Choose scab-resistant apple varieties such as Liberty, Enterprise, and Freedom.",
        "Ensure proper spacing and pruning: Maintain adequate spacing and prune branches to allow good air circulation and sunlight penetration."
    ],
    'solution': "To control apple scab, apply fungicides containing myclobutanil, captan, or sulfur at regular intervals during the growing season, especially during bud break, bloom, and petal fall. Begin applications in early spring and continue every 7-10 days under wet conditions. Remove and destroy any infected leaves and fruits to break the disease cycle. Improving orchard sanitation and ensuring proper pruning to enhance airflow can significantly reduce disease severity."
},
    'Apple___Black_rot': {
    'description': "Black rot is a fungal disease that affects apple trees, causing dark, sunken lesions on fruits, cankers on branches, and leaf spots. The infected fruits eventually shrivel and mummify, leading to significant yield loss if untreated. Leaf lesions, also known as frogeye spots, develop brown centers with purple margins.",
    'reasons': [
        "Fungus Botryosphaeria obtusa: The primary causal agent that infects apple trees through wounds and natural openings.",
        "Warm, humid conditions: Ideal temperature for fungal growth ranges between 75°F to 85°F (24°C to 29°C), especially during prolonged periods of rain or high humidity.",
        "Infected plant debris: Overwintered fungal spores in fallen leaves, mummified fruits, or deadwood serve as a source of reinfection.",
        "Improper pruning: Unremoved infected branches provide a habitat for the fungus to thrive and spread.",
        "Poor air circulation: Dense foliage and lack of pruning trap moisture, creating an ideal environment for fungal proliferation."
    ],
    'precautions': [
        "Prune and destroy infected branches: Remove all cankered and infected wood during the dormant season to reduce fungal spores.",
        "Ensure good air circulation: Thin the canopy by proper pruning to promote sunlight penetration and air movement, reducing moisture retention.",
        "Apply protective fungicides during bloom: Use fungicides such as captan, mancozeb, or thiophanate-methyl to prevent infection during bloom and fruit development stages.",
        "Sanitize orchard floor: Regularly clean the orchard floor by removing fallen leaves and mummified fruits to prevent overwintering of the pathogen.",
        "Plant resistant varieties: Opt for disease-resistant apple cultivars to minimize susceptibility to black rot."
    ],
    'solution': "To control black rot effectively, apply a series of fungicide treatments during critical growth stages. Use captan or thiophanate-methyl sprays starting at green tip, followed by applications during bloom and petal fall. Infected branches should be pruned and destroyed immediately. Ensure the orchard floor is free of fallen fruits and leaves. Additionally, improve drainage and enhance air circulation to reduce moisture buildup. In severe cases, copper-based fungicides can be applied as a supplementary treatment."
},
    'Apple___Cedar_apple_rust': {
    'description': "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae that affects apple trees and junipers. It causes yellow-orange spots with red halos on apple leaves and fruit, leading to premature leaf drop and reduced fruit quality. Severe infections weaken the tree and reduce yield.",
    'reasons': [
        "Fungus Gymnosporangium juniperi-virginianae: Completes its life cycle between apple trees and juniper/cedar hosts.",
        "Cool, wet spring weather: Moist conditions and temperatures between 50°F to 75°F (10°C to 24°C) promote spore production and infection.",
        "Proximity to juniper hosts: Juniper or cedar trees near apple orchards facilitate the spread of the fungus.",
        "Susceptible apple varieties: Varieties such as Gala, Golden Delicious, and Fuji are more prone to infection."
    ],
    'precautions': [
        "Remove nearby juniper hosts: Cut down or prune juniper and cedar trees within 1,000 feet of apple orchards.",
        "Apply preventive fungicides: Use fungicides such as myclobutanil or mancozeb during early spring to protect apple trees.",
        "Plant resistant apple varieties: Choose resistant cultivars like Redfree, Liberty, or Enterprise to reduce susceptibility.",
        "Prune infected branches: Remove galls and infected twigs from juniper hosts to interrupt the fungal life cycle."
    ],
    'solution': "To control cedar apple rust, apply fungicides such as myclobutanil, propiconazole, or chlorothalonil at intervals of 7-10 days during early spring, especially before and after bloom. Remove any infected leaves and fruit to prevent the spread of spores. Additionally, eliminate nearby juniper or cedar trees to disrupt the fungal life cycle and reduce infection risk. Regular monitoring and timely action can help manage disease outbreaks effectively."
},
    'Apple___healthy': {
    'description': "A healthy apple tree exhibits vibrant green leaves, free from discoloration, spots, or lesions. The tree produces high-quality fruit with a firm texture and smooth skin. The branches and trunk show no signs of cankers, cracks, or pest infestations, and the overall growth is vigorous.",
    'reasons': [
        "Optimal environmental conditions: Proper sunlight (6-8 hours/day), good air circulation, and well-drained soil.",
        "Regular watering and fertilization: Maintaining adequate moisture and nutrient levels to support tree growth and fruit development.",
        "Effective pest and disease management: Timely application of preventive sprays and monitoring to avoid infections.",
        "Proper pruning and maintenance: Regular pruning promotes air circulation and reduces the risk of disease."
    ],
    'precautions': [
        "Maintain proper soil health: Conduct regular soil tests and amend soil as needed to balance pH and nutrient levels.",
        "Ensure regular watering: Deep watering during dry periods, ensuring the soil remains moist but not waterlogged.",
        "Prune regularly: Remove dead or diseased branches to improve air circulation and light penetration.",
        "Monitor for pests and diseases: Inspect the tree regularly and take preventive measures when necessary."
    ],
    'solution': "To maintain a healthy apple tree, provide consistent care through proper watering, fertilizing, and pruning. Apply organic mulch around the base to retain moisture and prevent weed growth. If any signs of stress, nutrient deficiency, or disease are detected, take corrective action promptly. Regularly apply preventive fungicides or insecticides if necessary and ensure good sanitation practices by removing fallen leaves and fruit. Conduct regular inspections to ensure ongoing health and vitality."
},

    # Blueberry
    'Blueberry___healthy': {
    'description': "A healthy blueberry plant features bright green, glossy leaves and firm, plump berries with vibrant color. The plant exhibits vigorous growth, producing multiple shoots and branches with no signs of discoloration, wilting, or pest infestation. The roots are well-developed and free from rot or disease.",
    'reasons': [
        "Optimal environmental conditions: Full sunlight (6-8 hours/day) and well-drained, acidic soil with a pH of 4.5-5.5.",
        "Adequate watering and mulching: Consistent moisture levels without waterlogging, with mulch helping to retain soil moisture.",
        "Proper fertilization: Application of balanced fertilizers (high in nitrogen) during the growing season to support growth and fruiting.",
        "Pest and disease management: Routine monitoring and timely control of pests and diseases to prevent infections."
    ],
    'precautions': [
        "Maintain soil acidity: Regularly test soil pH and apply sulfur or acidic amendments if necessary.",
        "Water regularly: Ensure deep watering, especially during dry spells, to prevent stress and ensure high yields.",
        "Mulch around the base: Apply organic mulch to retain moisture, prevent weed growth, and protect roots.",
        "Prune regularly: Remove dead or weak branches to encourage new growth and improve air circulation."
    ],
    'solution': "To maintain a healthy blueberry plant, ensure proper watering and soil acidity by adjusting pH when needed. Mulch with organic material such as pine bark to keep the soil cool and moist. Fertilize with balanced nutrients designed for acid-loving plants. Regularly monitor for pests and diseases, applying appropriate organic or chemical treatments if necessary. Prune annually to remove old canes and promote new growth for higher yields."
},

    # Cherry Diseases
    'Cherry_(including_sour)___healthy': {
    'description': "A healthy sour cherry plant exhibits vibrant green, glossy leaves, strong branches, and abundant flowering in spring. The cherries develop evenly, turning bright red or deep burgundy when ripe, with firm flesh and minimal fruit drop. The plant shows no signs of fungal infection, pest damage, or nutrient deficiencies.",
    'reasons': [
        "Optimal growing conditions: Full sunlight (6-8 hours/day) and well-drained, loamy soil with a pH of 6.0-6.8.",
        "Consistent watering: Regular irrigation, especially during flowering and fruit development, prevents drought stress.",
        "Proper fertilization: Balanced application of nitrogen, phosphorus, and potassium fertilizers during the growing season.",
        "Pest and disease control: Regular inspection and timely intervention to prevent insect infestations and fungal diseases.",
        "Correct pruning: Annual pruning to remove dead wood and improve air circulation, reducing disease risk."
    ],
    'precautions': [
        "Maintain proper spacing: Ensure 15-20 feet of space between trees for adequate airflow and sunlight.",
        "Mulch around the base: Apply organic mulch to retain moisture, prevent weeds, and maintain consistent soil temperature.",
        "Prune annually: Remove dead or diseased wood to promote healthy growth and better fruit production.",
        "Monitor for pests and diseases: Conduct regular inspections and apply appropriate treatments if needed."
    ],
    'solution': "To maintain a healthy sour cherry plant, provide adequate sunlight, regular watering, and nutrient-rich soil. Apply organic mulch to retain soil moisture and control weeds. Prune annually to remove dead branches and improve airflow. Monitor for pests and fungal infections, applying organic or chemical treatments as required. Fertilize during the growing season with balanced nutrients to support healthy growth and fruiting."
},
    'Cherry_(including_sour)___Powdery_mildew': {
    'description': "Powdery mildew is a fungal disease that forms a white, powdery coating on the leaves, shoots, and sometimes fruit of sour cherry plants. Infected leaves may curl, distort, and drop prematurely, while fruit production can be reduced or impaired.",
    'reasons': [
        "Fungus Podosphaera clandestina, which thrives in warm, dry conditions (60-80°F).",
        "High humidity and poor air circulation create an ideal environment for fungal growth.",
        "Overcrowding of plants or trees limits air movement and promotes fungal spread.",
        "Lack of sunlight due to dense foliage or shaded areas contributes to increased humidity.",
        "Susceptible cherry varieties that lack resistance to powdery mildew."
    ],
    'precautions': [
        "Ensure proper spacing: Plant trees 15-20 feet apart to promote good airflow and reduce humidity.",
        "Prune regularly: Remove excess branches and foliage to increase light penetration and improve air circulation.",
        "Apply preventive fungicides: Use sulfur-based or neem oil sprays before the disease appears, especially during warm and humid periods.",
        "Water at the base: Avoid overhead watering, which increases humidity and favors fungal growth.",
        "Choose resistant varieties: Opt for cherry cultivars that are less prone to powdery mildew."
    ],
    'solution': "If powdery mildew is detected, remove and destroy infected plant parts to prevent further spread. Apply fungicides containing myclobutanil, sulfur, or potassium bicarbonate at the first sign of infection, repeating every 7-10 days as needed. Increase air circulation by thinning branches and ensure the plant receives adequate sunlight. For severe infections, consider systemic fungicides for long-term control. Improve soil drainage and avoid excessive nitrogen fertilization to reduce susceptibility."
},

    # Corn Diseases
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
    'description': "Gray leaf spot (GLS) is a fungal disease that affects corn plants, causing elongated, rectangular, grayish-brown lesions on the leaves. Severe infections can result in premature leaf death, reducing photosynthetic capacity and leading to yield losses.",
    'reasons': [
        "Fungus Cercospora zeae-maydis, which thrives in warm, humid environments (75-85°F).",
        "Extended periods of leaf wetness from dew, rain, or irrigation.",
        "Presence of infected crop debris, which harbors fungal spores and facilitates overwintering.",
        "Dense crop canopy that limits airflow and promotes prolonged moisture retention.",
        "Susceptible maize hybrids with low resistance to Cercospora zeae-maydis."
    ],
    'precautions': [
        "Rotate crops: Avoid planting corn in the same field for consecutive seasons to break the disease cycle.",
        "Remove crop residues: Plow under or remove infected plant debris after harvest to minimize fungal overwintering.",
        "Plant resistant hybrids: Select maize hybrids with high resistance to gray leaf spot.",
        "Manage plant density: Ensure proper spacing between plants to promote air circulation and reduce humidity.",
        "Use drip irrigation: Avoid overhead irrigation that keeps leaves wet and promotes fungal growth."
    ],
    'solution': "If gray leaf spot is detected, apply foliar fungicides such as strobilurins (azoxystrobin, pyraclostrobin) or triazoles (propiconazole, tebuconazole) at the early stages of disease development. Apply fungicides between the V8 and R1 growth stages for best results. Consider using a combination of fungicides with different modes of action to prevent resistance buildup. Scout fields regularly and manage nitrogen levels to promote plant health and reduce susceptibility to infections."
},
'Corn_(maize)___Common_rust_': {
    'description': "Common rust is a fungal disease that affects corn, characterized by the formation of reddish-brown to dark brown pustules (raised bumps) on both the upper and lower surfaces of leaves. Severe infections can lead to premature leaf senescence and reduced yield.",
    'reasons': [
        "Fungus Puccinia sorghi, which thrives in cool, humid conditions (60-77°F).",
        "Windborne urediniospores (spores) that spread from infected fields.",
        "Extended periods of high humidity, dew, and prolonged leaf wetness.",
        "Planting susceptible maize hybrids with low resistance to rust."
    ],
    'precautions': [
        "Plant resistant hybrids: Select maize varieties with genetic resistance to common rust.",
        "Crop rotation: Avoid planting corn in the same field for consecutive seasons to reduce inoculum levels.",
        "Early planting: Plant early to allow the crop to escape severe rust infection during critical growth stages.",
        "Manage plant density: Maintain optimal plant spacing to promote airflow and reduce humidity around leaves."
    ],
    'solution': "If common rust is detected, apply foliar fungicides such as strobilurins (azoxystrobin, pyraclostrobin) or triazoles (propiconazole, tebuconazole) at the early stages of infection. Apply fungicides between the V8 and R1 growth stages for maximum effectiveness. Scout fields regularly to assess disease severity and determine the need for additional applications. Maintain optimal nitrogen levels to support plant health and resistance."
},
'Corn_(maize)___healthy': {
    'description': "Healthy corn plants exhibit vibrant green leaves, robust stalks, and consistent growth without any visible signs of disease, nutrient deficiency, or pest damage. Ears develop uniformly, with kernels maturing fully and maintaining a bright golden color.",
    'reasons': [
        "Optimal environmental conditions, including well-drained soil, adequate sunlight, and consistent rainfall or irrigation.",
        "Proper nutrient management, including balanced nitrogen, phosphorus, and potassium levels.",
        "Effective pest and disease control through regular monitoring and preventive measures.",
        "Good agricultural practices such as crop rotation, maintaining proper plant spacing, and timely weeding."
    ],
    'precautions': [
        "Use certified, disease-free seeds to prevent introducing pathogens into the field.",
        "Rotate crops with non-host species to break the disease cycle and maintain soil health.",
        "Apply appropriate fertilizers based on soil test results to ensure optimal nutrient balance.",
        "Monitor for pest activity and apply integrated pest management (IPM) strategies if needed."
    ],
    'solution': "To maintain plant health, continue regular field scouting to identify early signs of stress or disease. Provide adequate irrigation during dry periods to support crop growth. Apply foliar fertilizers when necessary to address nutrient deficiencies. If minor pest activity is observed, consider using biological or chemical controls to prevent escalation."
},
    'Corn_(maize)___Northern_Leaf_Blight': {
    'description': "Northern Leaf Blight (NLB) is a fungal disease that causes long, elliptical, grayish-green to tan lesions on the leaves of corn plants, eventually leading to premature leaf death and reduced grain yield if left untreated.",
    'reasons': [
        "Caused by the fungus *Exserohilum turcicum*.",
        "Favorable conditions include prolonged periods of high humidity (90-100%) and moderate temperatures between 64°F and 81°F (18°C to 27°C).",
        "Poor air circulation and high plant density create a humid microclimate that promotes fungal growth.",
        "Infection spreads through wind-borne spores that overwinter on infected crop residues."
    ],
    'precautions': [
        "Plant resistant or tolerant hybrids to minimize susceptibility to NLB.",
        "Practice crop rotation with non-host crops like soybeans or wheat to reduce the buildup of fungal spores in the soil.",
        "Avoid excessive nitrogen fertilization, which promotes dense foliage and increases humidity within the canopy.",
        "Remove and destroy infected plant residues after harvest to reduce overwintering spores."
    ],
    'solution': "If Northern Leaf Blight is detected, apply fungicides containing active ingredients such as azoxystrobin, pyraclostrobin, or propiconazole at the early onset of disease symptoms. Ensure proper coverage and reapply as needed according to the product label to protect the developing foliage. Additionally, adjust planting density to improve air circulation and reduce humidity within the canopy."
},

    # Grape Diseases
    'Grape___Black_rot': {
    'description': "Black rot is a severe fungal disease that affects grapevines, causing circular brown lesions on leaves, black shriveled berries, and fruit rot, ultimately leading to significant yield losses if untreated.",
    'reasons': [
        "Caused by the fungus *Guignardia bidwellii*.",
        "Favorable conditions include warm, humid weather with temperatures between 75°F and 85°F (24°C to 29°C).",
        "Prolonged leaf wetness of 6-9 hours promotes fungal spore germination and infection.",
        "Spores spread through rain splash, wind, and insects, infecting healthy tissues.",
        "Infected plant debris and mummified berries left on the vine or soil act as a reservoir for the fungus."
    ],
    'precautions': [
        "Prune and remove infected vines and mummified berries to reduce overwintering inoculum.",
        "Ensure proper vineyard sanitation by removing fallen leaves and debris after harvest.",
        "Promote good air circulation by maintaining proper row spacing and trellis management.",
        "Apply preventive fungicides such as mancozeb, myclobutanil, or captan at critical growth stages (bud break, bloom, and fruit set)."
    ],
    'solution': "If black rot is detected, apply systemic fungicides like tebuconazole or myclobutanil immediately. Repeat applications every 7-10 days, especially during wet and humid conditions. Additionally, improve canopy management to enhance air circulation and reduce leaf wetness duration."
},
    'Grape___Esca_(Black_Measles)': {
    'description': "Esca, also known as Black Measles, is a complex and chronic grapevine disease that affects the vascular system, leading to reduced yield, leaf scorch, berry spotting, and eventual vine death. It often affects older vines and weakens the plant progressively.",
    'reasons': [
        "Caused by a complex of fungal pathogens, including *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*.",
        "Infection usually enters through pruning wounds, allowing the pathogens to colonize the xylem tissue.",
        "Hot and dry climates accelerate symptom development, while moisture during pruning season increases infection risk.",
        "Poor vineyard sanitation, including infected vine debris and old wood, contributes to fungal spread."
    ],
    'precautions': [
        "Use clean and sanitized pruning tools to minimize the risk of infection.",
        "Apply wound protectants or fungicides on pruning cuts to prevent fungal entry.",
        "Remove and burn infected vines to reduce fungal inoculum in the vineyard.",
        "Maintain balanced fertilization and irrigation to avoid vine stress, which can exacerbate infection."
    ],
    'solution': "Currently, no cure exists for Esca once the infection has advanced. However, for early infections, applying fungicides containing tebuconazole or pyraclostrobin to pruning wounds may slow disease progression. Remove and destroy infected vines to prevent the spread of the pathogen. Replant with resistant or certified disease-free planting material."
},
    'Grape___healthy': {
    'description': "A healthy grapevine exhibits vibrant green leaves, free from spots or discoloration, strong stems, and well-formed grape clusters. The plant shows no signs of wilting, pest infestation, or fungal infections.",
    'reasons': [
        "Optimal sunlight exposure, ensuring sufficient photosynthesis.",
        "Proper soil drainage and balanced nutrient levels, especially nitrogen, phosphorus, and potassium.",
        "Adequate spacing between vines to ensure good air circulation and reduce humidity.",
        "Consistent monitoring for pests and diseases to maintain plant health."
    ],
    'precautions': [
        "Implement a regular pruning schedule to improve air circulation and reduce disease risk.",
        "Ensure drip irrigation or proper watering techniques to maintain soil moisture without overwatering.",
        "Apply preventive fungicides and insecticides during the growing season if necessary.",
        "Use mulch around the base of the plant to retain soil moisture and prevent weed growth."
    ],
    'solution': "To maintain a healthy grapevine, practice integrated pest management (IPM), ensure nutrient-rich soil, and monitor for any early signs of stress, disease, or pest infestation. Apply preventive treatments during critical growth stages and prune properly to maintain healthy growth."
},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
    'description': "Isariopsis Leaf Spot is a fungal disease that causes irregular, dark lesions with light centers on the leaves of grapevines. As the disease progresses, the lesions may merge, leading to extensive tissue damage and premature defoliation. In severe cases, it can weaken the plant and reduce grape yield.",
    'reasons': [
        "Fungus Isariopsis griseola",
        "Warm, humid weather (75-85°F)",
        "Rain or irrigation splashing spores onto the leaves",
        "Crowded plantings that promote poor air circulation",
        "Excessive nitrogen fertilization leading to lush, susceptible growth"
    ],
    'precautions': [
        "Avoid overhead irrigation to reduce water splash on leaves",
        "Prune vines to improve air circulation",
        "Plant resistant grape varieties if available",
        "Remove and destroy infected leaves and debris in the fall",
        "Apply fungicide treatments before the disease appears or as soon as symptoms are noticed"
    ],
    'solution': "Apply fungicides containing copper-based compounds, thiophanate-methyl, or chlorothalonil. Make applications every 7-10 days during the growing season, especially during wet weather, and follow up with regular preventive treatments."
},

    # Orange Disease
    'Orange___Huanglongbing_(Citrus_greening)': {
    'description': "Huanglongbing (HLB), also known as citrus greening, is a bacterial disease that affects citrus trees, leading to yellowing of the leaves, stunted growth, and fruit that is small, misshapen, and bitter. Over time, the disease causes premature fruit drop and can kill the tree if left untreated.",
    'reasons': [
        "Bacterium Candidatus Liberibacter asiaticus",
        "Transmission by the Asian citrus psyllid (Diaphorina citri)",
        "Infected psyllids feeding on the tree and spreading the bacteria",
        "Warm temperatures and high humidity promote the spread of the disease"
    ],
    'precautions': [
        "Monitor trees for the presence of the Asian citrus psyllid and other insect vectors",
        "Remove and destroy infected trees to prevent the spread of the bacteria",
        "Use insecticides to control the psyllid population, but be aware of resistance issues",
        "Practice good sanitation by sterilizing pruning tools and equipment",
        "Plant resistant or tolerant citrus varieties if available"
    ],
    'solution': "Currently, there is no cure for Huanglongbing. Management focuses on controlling psyllid populations with insecticides such as imidacloprid or acetamiprid. Additionally, systemic treatments with bactericides like oxytetracycline may help slow disease progression. Careful management practices, including pruning and sanitation, can help manage the disease in affected orchards."
},

    # Peach Diseases
    'Peach___Bacterial_spot': {
    'description': "Bacterial spot is a common bacterial disease affecting peach and other stone fruit trees, characterized by water-soaked lesions on leaves, fruit, and twigs. On leaves, the lesions appear as small, round, dark spots with a yellow halo. Infected fruit may develop sunken spots, making it unmarketable. The disease can lead to premature defoliation and reduce the tree’s vigor and fruit yield.",
    'reasons': [
        "Bacterium Xanthomonas arboricola pv. pruni",
        "Rainy, humid conditions with frequent leaf wetness",
        "Infected plant material, tools, or workers spreading bacteria",
        "Overcrowded trees with poor air circulation increasing moisture retention",
        "Infected seed or nursery stock can introduce the disease to new orchards"
    ],
    'precautions': [
        "Avoid overhead irrigation to reduce moisture on leaves and fruit",
        "Prune trees to ensure proper air circulation and reduce humidity",
        "Remove and destroy infected leaves, twigs, and fruit to limit the spread of the bacteria",
        "Sanitize pruning tools regularly to prevent bacterial spread",
        "Plant resistant or tolerant varieties if available",
        "Avoid planting in areas with heavy rainfall or poor drainage"
    ],
    'solution': "Apply copper-based fungicides or bactericides, such as copper hydroxide, during the growing season, especially before rainy periods. Begin spraying early in the spring when buds begin to swell and continue every 7-14 days during the growing season. Some systemic bactericides may also be effective. Ensure proper orchard sanitation and remove infected plant debris to reduce the bacterial load."
},
    'Peach___Healthy': {
    'description': "Healthy peach trees exhibit strong growth, vibrant green leaves, and produce high-quality fruit. The tree's foliage remains free of major pests and diseases, and fruit development occurs without distortion or lesions. Healthy peach trees typically have a robust root system and are resistant to common environmental stressors like drought and excessive moisture.",
    'reasons': [
        "Proper care including watering, fertilization, and pruning",
        "Planting in well-drained soil with optimal pH (6.0 to 6.5)",
        "Protection from common peach pests and diseases through preventive treatments",
        "Healthy root system that is not stressed by soil compaction or poor drainage",
        "Appropriate climate with sufficient chilling hours during the winter for proper bud development"
    ],
    'precautions': [
        "Regularly monitor for pests and diseases, particularly early in the growing season",
        "Provide adequate watering during dry periods, ensuring deep watering rather than frequent shallow watering",
        "Use proper mulching techniques to retain soil moisture and regulate temperature",
        "Ensure proper spacing between trees to avoid overcrowding and encourage good air circulation",
        "Fertilize according to soil test recommendations to prevent nutrient deficiencies or imbalances",
        "Prune trees annually to remove dead or damaged branches and promote healthy growth"
    ],
    'solution': "To maintain a healthy peach tree, continue regular care practices such as adequate watering, proper fertilization, and pruning. In case of pest or disease outbreaks, use appropriate organic or chemical treatments based on the specific issue. Preventive care, such as applying fungicides and insecticides as recommended by local agricultural extension services, can help keep the tree healthy."
},

    # Pepper Diseases
    'Pepper,_bell___Bacterial_spot': {
    'description': "Bacterial spot is a common disease that affects bell pepper plants, characterized by dark, water-soaked lesions with yellow halos on leaves, stems, and fruit. The spots are irregular and can lead to premature leaf drop, reducing plant vigor and fruit yield. Infected fruit may develop sunken spots, becoming unmarketable. The disease can also weaken plants, making them more susceptible to other stressors.",
    'reasons': [
        "Bacterium Xanthomonas campestris pv. vesicatoria",
        "Warm, humid weather with frequent rainfall",
        "Infected plant material, tools, or workers spreading bacteria",
        "Overcrowding of plants, resulting in poor air circulation",
        "Irrigation that wets the foliage, creating favorable conditions for the bacteria"
    ],
    'precautions': [
        "Avoid overhead irrigation to reduce moisture on leaves and fruit",
        "Ensure proper spacing between plants to allow for good air circulation",
        "Remove and destroy infected plant debris to prevent the spread of the bacteria",
        "Sanitize gardening tools, equipment, and hands when handling plants",
        "Use disease-free seeds or transplants",
        "Practice crop rotation to avoid planting peppers in the same soil for consecutive seasons"
    ],
    'solution': "Apply copper-based bactericides or other recommended bactericides early in the growing season, starting at transplant and continuing through periods of high humidity and rain. Continue applications at 7-10 day intervals during periods of wet weather. Use resistant varieties where available, and practice good sanitation to limit bacterial spread."
},
    'Pepper,_bell___Healthy': {
    'description': "Healthy bell pepper plants are characterized by strong, vigorous growth with bright green leaves, sturdy stems, and high-quality fruit. The leaves are free from lesions or discoloration, and the fruit develops to its full size without any deformities or spots. Healthy plants have a well-developed root system and can effectively take up nutrients and water, promoting optimal growth and fruit production.",
    'reasons': [
        "Proper care including appropriate watering, fertilization, and pruning",
        "Planting in well-drained soil with a pH of 6.0 to 6.8",
        "Adequate sunlight (6-8 hours of direct sunlight per day)",
        "Protection from pests and diseases through regular monitoring and preventive treatments",
        "Consistent irrigation to avoid both drought stress and waterlogging",
        "Good air circulation around plants to minimize the risk of fungal and bacterial diseases"
    ],
    'precautions': [
        "Monitor for pests such as aphids, whiteflies, and spider mites that can damage plants",
        "Ensure proper spacing between plants to encourage good airflow and reduce the risk of disease",
        "Mulch around plants to conserve moisture and reduce weed competition",
        "Fertilize regularly based on soil test recommendations to avoid nutrient deficiencies",
        "Prune dead or damaged leaves and stems to maintain plant health",
        "Practice crop rotation to prevent soil-borne diseases from affecting future crops"
    ],
    'solution': "To maintain healthy bell peppers, continue regular care practices such as proper watering, fertilization, and pest management. If any pest or disease issues arise, address them promptly using appropriate organic or chemical treatments. Ensuring proper environmental conditions and sanitation will promote strong plant growth and high-quality fruit production."
},

    # Potato Diseases
    'Potato___Early_blight': {
    'description': "Early blight is a fungal disease that affects potato plants, causing dark, concentric-ring lesions on the lower leaves, stems, and tubers. The lesions are typically dark brown or black with a yellow halo. As the disease progresses, it can lead to premature leaf drop, reducing photosynthesis and weakening the plant. Severe infections can also affect the potato tubers, causing blemishes that make them unmarketable.",
    'reasons': [
        "Fungus Alternaria solani",
        "Warm, wet conditions (optimum temperature of 70-80°F)",
        "Overhead irrigation or rain that wets the foliage",
        "Infected seed potatoes or plant debris left in the soil",
        "Poor air circulation around plants"
    ],
    'precautions': [
        "Use disease-free seed potatoes to prevent initial infection",
        "Rotate crops every 2-3 years to avoid re-infection from soil-borne spores",
        "Ensure proper spacing between plants to allow for good airflow",
        "Avoid wetting the foliage with irrigation, and use drip irrigation if possible",
        "Remove and destroy infected leaves, stems, and plant debris",
        "Prune and manage the plants to improve air circulation around the canopy"
    ],
    'solution': "Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin every 7-10 days starting when the plants begin to grow. Continue applications until the plants begin to mature. Remove and destroy infected plant debris after harvest to reduce the fungal load for the next season. In areas with frequent rainfall, fungicide applications may need to be more frequent to maintain control."
},
    'Potato___Healthy': {
    'description': "Healthy potato plants exhibit vigorous growth, with strong stems, vibrant green leaves, and healthy tubers. The plants are free of major pests and diseases, and their leaves remain bright green without spots, lesions, or wilting. Healthy potato plants develop well-formed, undamaged tubers and produce high yields with minimal environmental stress.",
    'reasons': [
        "Proper care, including correct watering, fertilization, and pest management",
        "Planting in well-drained, loose, and fertile soil with a pH of 5.5-6.5",
        "Adequate sunlight (at least 6 hours per day) for healthy photosynthesis",
        "Protection from diseases and pests through regular monitoring and preventive treatments",
        "Proper spacing between plants for optimal air circulation and disease prevention",
        "Consistent, balanced irrigation to avoid both drought and waterlogging"
    ],
    'precautions': [
        "Monitor regularly for common pests such as aphids, Colorado potato beetles, and potato tuber moths",
        "Use proper crop rotation practices to reduce the risk of soil-borne diseases",
        "Ensure proper spacing between plants to avoid overcrowding and promote airflow",
        "Fertilize according to soil test recommendations to avoid nutrient imbalances",
        "Mulch around plants to maintain soil moisture, suppress weeds, and protect tubers",
        "Prune any dead or damaged plant material to encourage healthy growth"
    ],
    'solution': "To maintain healthy potato plants, continue regular care practices such as proper watering, fertilization, and pest control. If disease or pest issues arise, use organic or chemical treatments as appropriate. Ensuring optimal environmental conditions and regular maintenance will keep plants vigorous and productive."
},
    'Potato___Late_blight': {
    'description': "Late blight is a destructive fungal disease that affects potatoes and tomatoes, caused by the pathogen *Phytophthora infestans*. It leads to large, irregular, dark lesions on the leaves, stems, and tubers. The lesions have a watery, greasy appearance, and they spread rapidly, causing the leaves to wither and die. The infection can also cause tuber rot, making them unfit for consumption or storage.",
    'reasons': [
        "Fungus-like organism *Phytophthora infestans*",
        "Cool, wet weather (temperatures between 60-70°F and frequent rain)",
        "Overhead irrigation or rain that wets the foliage",
        "Infected plant material, especially seed potatoes",
        "High humidity and poor air circulation, which create ideal conditions for the pathogen"
    ],
    'precautions': [
        "Use certified disease-free seed potatoes to avoid initial infection",
        "Practice crop rotation, not planting potatoes or tomatoes in the same field for consecutive years",
        "Remove and destroy infected plant material, including leaves, stems, and tubers, to reduce the spread of the pathogen",
        "Avoid overhead irrigation and instead use drip irrigation to keep foliage dry",
        "Ensure proper spacing between plants to improve air circulation",
        "Apply protective fungicides as a preventive measure, especially during wet weather"
    ],
    'solution': "Apply fungicides such as chlorothalonil, mancozeb, or mefenoxam, beginning as soon as the plants emerge and continue at regular intervals (every 7-10 days) during wet weather. Infected tubers should be removed and discarded to prevent further spread. Monitor crops regularly and remove any symptomatic plants immediately. Harvest tubers before they become overripe to avoid rot during storage."
},

    # Raspberry
    'Raspberry___healthy': {
    'description': "Healthy raspberry plants have strong canes with vibrant green leaves, producing a high yield of flavorful, well-formed berries. The leaves are free from major diseases, pests, or discoloration, and the plants exhibit robust growth. Healthy raspberry plants typically have a well-developed root system, allowing them to withstand environmental stressors such as drought or heavy rain.",
    'reasons': [
        "Proper care, including correct watering, fertilization, and pruning",
        "Planting in well-drained soil with a pH of 5.5-6.5",
        "Adequate sunlight (6-8 hours per day) for optimal photosynthesis",
        "Consistent and balanced irrigation to avoid both drought stress and waterlogging",
        "Protection from pests and diseases through regular monitoring and preventive treatments",
        "Good air circulation to reduce the risk of fungal diseases"
    ],
    'precautions': [
        "Monitor regularly for pests such as aphids, raspberry beetles, and spider mites",
        "Ensure proper spacing between plants to promote good airflow and reduce disease risk",
        "Mulch around the base of plants to retain moisture and suppress weeds",
        "Prune canes regularly to remove dead or damaged growth and to encourage new growth",
        "Apply balanced fertilizer based on soil test recommendations to avoid nutrient deficiencies",
        "Practice crop rotation to minimize the risk of soil-borne diseases"
    ],
    'solution': "To maintain healthy raspberry plants, continue regular care practices such as proper watering, pruning, and pest management. In case of pest or disease issues, treat early using organic or chemical solutions. Regular maintenance, appropriate environmental conditions, and proper sanitation will help ensure the health and productivity of your raspberry plants."
},

    # Soybean
    'Soybean___healthy': {
    'description': "Healthy soybean plants exhibit strong, upright growth, with vibrant green leaves and well-developed root systems. The plants are free from major pests and diseases, and their leaves remain free from discoloration, wilting, or lesions. Healthy soybeans produce high-quality, plump beans and demonstrate resistance to environmental stress, such as drought or excessive moisture.",
    'reasons': [
        "Proper care including adequate watering, fertilization, and pest management",
        "Planting in well-drained soil with a pH of 6.0-7.0",
        "Full sunlight exposure (6-8 hours of direct sunlight daily)",
        "Consistent, balanced irrigation to avoid both drought stress and waterlogging",
        "Protection from pests and diseases through regular monitoring and preventive treatments",
        "Optimal planting density to ensure good airflow and reduce competition for nutrients"
    ],
    'precautions': [
        "Monitor regularly for pests such as aphids, soybean cyst nematodes, and whiteflies",
        "Ensure proper spacing between plants to promote good airflow and minimize the risk of fungal diseases",
        "Apply appropriate fertilizer based on soil test recommendations to prevent nutrient deficiencies",
        "Practice crop rotation to reduce the risk of soil-borne diseases and pest build-up",
        "Control weeds early to avoid competition for nutrients and water",
        "Ensure proper irrigation practices, avoiding both waterlogging and drought stress"
    ],
    'solution': "To maintain healthy soybean plants, continue regular care practices such as proper watering, fertilization, and pest control. If pests or diseases appear, apply the appropriate organic or chemical treatments as needed. Regular maintenance and proper environmental conditions will ensure the long-term health and productivity of the soybean crop."
},

    # Squash
    'Squash___Powdery_mildew': {
    'description': "Powdery mildew is a fungal disease that affects squash plants, characterized by white, powdery fungal growth on the leaves, stems, and sometimes fruit. This disease typically starts on the undersides of leaves and can spread quickly under warm, dry conditions. Over time, the leaves become yellow and brittle, leading to reduced photosynthesis and, ultimately, stunted plant growth and reduced fruit production.",
    'reasons': [
        "Fungus Podosphaera xanthii (formerly Erysiphe cichoracearum)",
        "Warm, dry conditions with high humidity",
        "Infected seeds or plant debris from previous seasons",
        "Overhead irrigation that wets foliage and creates favorable conditions for fungal growth",
        "Poor air circulation around plants"
    ],
    'precautions': [
        "Space plants properly to ensure good air circulation and reduce humidity around the foliage",
        "Avoid overhead irrigation and opt for drip irrigation to keep the foliage dry",
        "Remove and destroy infected leaves and debris to prevent the spread of the fungus",
        "Use resistant varieties of squash where available",
        "Practice crop rotation to avoid planting squash in the same area year after year",
        "Sterilize tools and equipment regularly to avoid spreading fungal spores"
    ],
    'solution': "Apply fungicides containing active ingredients such as sulfur, neem oil, or potassium bicarbonate as a preventive measure, especially during the early stages of growth. For severe infections, systemic fungicides like myclobutanil can be used. Make sure to follow the label instructions and apply fungicides every 7-14 days during the growing season. Prune affected leaves and practice good sanitation to reduce fungal spore buildup."
},

    # Strawberry Diseases
    'Strawberry___healthy': {
    'description': "Healthy strawberry plants have lush green leaves, strong runners, and produce high-quality fruit. The plants are free from pests and diseases, and the leaves are vibrant, without any yellowing, spots, or wilting. Healthy strawberries show robust growth, with well-formed fruit that ripens evenly. The plants have a strong root system, which allows them to absorb nutrients and water efficiently, supporting both fruit production and overall plant health.",
    'reasons': [
        "Proper care, including adequate watering, fertilization, and pest management",
        "Planting in well-drained, fertile soil with a pH of 5.5-6.5",
        "Optimal sunlight exposure (at least 6-8 hours of direct sunlight daily)",
        "Consistent, balanced irrigation to avoid both drought stress and waterlogging",
        "Protection from pests and diseases through regular monitoring and preventive treatments",
        "Proper spacing between plants to promote good air circulation and prevent overcrowding"
    ],
    'precautions': [
        "Monitor regularly for pests such as aphids, spider mites, and slugs",
        "Ensure proper spacing between plants to allow for adequate airflow and reduce the risk of fungal infections",
        "Mulch around the base of plants to retain moisture, suppress weeds, and protect fruit from soil contact",
        "Prune damaged or diseased leaves and runners regularly to encourage healthy growth",
        "Apply balanced fertilizers based on soil test recommendations to avoid nutrient deficiencies",
        "Practice crop rotation to minimize the risk of soil-borne diseases like verticillium wilt"
    ],
    'solution': "To maintain healthy strawberry plants, continue regular care practices such as proper watering, fertilization, and pest management. If pests or diseases arise, treat early using appropriate organic or chemical solutions. Regular maintenance, appropriate environmental conditions, and proper sanitation will help ensure the health and productivity of your strawberry plants."
},
    'Strawberry___Leaf_scorch': {
    'description': "Leaf scorch is a physiological disorder that affects strawberry plants, characterized by the browning or scorching of the leaf edges, often with yellowing between the veins. This condition typically occurs under hot, dry conditions, causing the plant's leaves to become dehydrated. Leaf scorch can stunt plant growth and reduce fruit yield if not addressed, as the leaves are no longer efficient in photosynthesis.",
    'reasons': [
        "Excessive heat and drought stress",
        "Inconsistent or inadequate watering, leading to dehydration",
        "High levels of direct sunlight, especially during hot summer days",
        "Nutrient imbalances, particularly a lack of potassium or nitrogen",
        "Poor soil drainage, leading to root stress and inadequate water uptake"
    ],
    'precautions': [
        "Water plants consistently, ensuring the soil remains moist but not waterlogged",
        "Provide some shade during the hottest part of the day, especially in regions with high temperatures",
        "Mulch around the base of the plants to conserve moisture and regulate soil temperature",
        "Avoid overhead irrigation, as it can contribute to leaf scorch by causing rapid evaporation",
        "Fertilize based on soil test recommendations to prevent nutrient deficiencies",
        "Ensure proper spacing between plants to reduce overcrowding and improve air circulation"
    ],
    'solution': "To resolve leaf scorch, ensure regular and adequate watering, particularly during dry spells. Use drip irrigation to keep water off the leaves and focus on soil moisture. Apply a balanced fertilizer with appropriate potassium and nitrogen levels to address any nutrient deficiencies. If scorch is severe, remove the affected leaves to help the plant recover, and provide temporary shading during extreme heat to protect the remaining foliage."
},

    # Tomato Diseases
    'Tomato___Bacterial_spot': {
    'description': "Bacterial spot is a disease caused by the bacterium *Xanthomonas vesicatoria*, which affects tomato plants. The disease leads to the formation of small, dark, water-soaked lesions on the leaves, stems, and fruit. These spots can enlarge and develop yellow halos. Infected leaves eventually turn yellow and fall off, weakening the plant and reducing fruit yield. Severe infections can cause premature fruit drop and deformed or unripe tomatoes.",
    'reasons': [
        "Bacterium *Xanthomonas vesicatoria*",
        "Warm, wet conditions (temperatures between 70-85°F with high humidity)",
        "Infected seed or plant debris left in the soil from previous crops",
        "Splashing water that spreads bacterial spores from the soil to the foliage",
        "Poor air circulation around plants"
    ],
    'precautions': [
        "Use disease-free seed and avoid planting tomatoes in areas where the disease was present in previous seasons",
        "Practice crop rotation to avoid planting tomatoes or other solanaceous crops in the same area for 2-3 years",
        "Avoid overhead irrigation and use drip irrigation to minimize water splash on the leaves",
        "Space plants appropriately to allow for good air circulation and reduce humidity around the foliage",
        "Remove and destroy infected plant material, including leaves and fruit, to reduce the spread of the bacterium",
        "Sterilize gardening tools and equipment to prevent transferring the bacteria from plant to plant"
    ],
    'solution': "There is no cure once the plants are infected, but control can be achieved through preventive measures. Apply copper-based bactericides or other labeled products, especially during periods of high moisture or when the disease is first noticed. Maintain proper spacing and watering techniques to reduce the spread of bacteria. If the infection becomes severe, remove and destroy infected plants and their debris to prevent further spread."
},
    'Tomato___Early_blight': {
    'description': "Early blight is a fungal disease caused by *Alternaria solani* that primarily affects the leaves of tomato plants, but can also spread to stems and fruit. The disease begins as small, dark brown or black spots with concentric rings on the lower leaves, which gradually expand and turn yellow. Infected leaves may eventually die and fall off, leading to reduced plant vigor, lower fruit yield, and potential fruit rot in severe cases.",
    'reasons': [
        "Fungus *Alternaria solani*",
        "Warm, wet weather (temperatures between 60-80°F and high humidity)",
        "Infected plant debris or seeds from previous seasons",
        "Overhead irrigation that wets the foliage, encouraging fungal growth",
        "Poor air circulation around plants, especially in crowded gardens"
    ],
    'precautions': [
        "Use disease-free seed and remove any infected plant debris from the previous season",
        "Practice crop rotation, avoiding planting tomatoes or other solanaceous crops in the same area for 2-3 years",
        "Space plants to promote good air circulation and reduce humidity around the plants",
        "Avoid overhead irrigation, and use drip irrigation to keep water off the leaves",
        "Remove and destroy infected leaves and plant parts as soon as possible",
        "Mulch around the base of the plants to prevent soil splash, which can spread spores"
    ],
    'solution': "Apply fungicides containing chlorothalonil, mancozeb, or copper to prevent early blight, beginning when the plants are young and continuing at 7-10 day intervals during the growing season, especially after rain. Infected leaves should be removed and destroyed to reduce the spread of spores. In severe cases, prune affected stems to prevent the fungus from spreading further into the plant."
},
    'Tomato___healthy': {
    'description': "Healthy tomato plants exhibit vigorous growth, with strong stems, vibrant green leaves, and well-developed roots. The plants are free from pests and diseases, and their leaves remain lush, without any discoloration, spots, or wilting. Healthy tomato plants produce high-quality fruit with a rich color, firm texture, and good taste. They grow in favorable environmental conditions, benefiting from proper care and maintenance.",
    'reasons': [
        "Proper care, including correct watering, fertilization, and pest management",
        "Planting in well-drained soil with a pH of 6.0-6.8",
        "Adequate sunlight (6-8 hours per day) for optimal photosynthesis",
        "Consistent and balanced irrigation to avoid both drought stress and waterlogging",
        "Protection from pests and diseases through regular monitoring and preventive treatments",
        "Optimal spacing between plants to promote good air circulation and reduce the risk of diseases"
    ],
    'precautions': [
        "Monitor regularly for common pests such as aphids, whiteflies, and tomato hornworms",
        "Ensure proper spacing between plants to allow for air circulation and prevent overcrowding",
        "Apply balanced fertilizers based on soil test recommendations to avoid nutrient deficiencies",
        "Mulch around the base of the plants to retain moisture and suppress weeds",
        "Prune and remove any dead or diseased plant material to promote healthy growth",
        "Practice crop rotation to avoid planting tomatoes or other solanaceous crops in the same area year after year"
    ],
    'solution': "To maintain healthy tomato plants, continue regular care practices such as proper watering, fertilization, and pest management. If pests or diseases are observed, treat them promptly with organic or chemical solutions. Regular maintenance, optimal growing conditions, and preventive care will help ensure the long-term health and productivity of your tomato plants."
},
    'Tomato___Late_blight': {
    'description': "Late blight is a severe fungal disease caused by the pathogen *Phytophthora infestans*, affecting tomatoes and potatoes. It begins as water-soaked lesions on leaves, which quickly enlarge and turn brown or black. The affected areas may have a white, mold-like growth on the undersides of leaves. Late blight spreads rapidly in cool, moist conditions and can lead to the rapid decay of both leaves and fruit, causing significant crop loss if not managed effectively.",
    'reasons': [
        "Fungus-like organism *Phytophthora infestans*",
        "Cool, wet weather (temperatures between 60-70°F and frequent rain)",
        "High humidity and poor air circulation, which create ideal conditions for the pathogen",
        "Infected seeds, transplants, or plant debris from previous seasons",
        "Overhead irrigation that wets the foliage, increasing the likelihood of fungal growth"
    ],
    'precautions': [
        "Use disease-free seeds and transplants to avoid initial infection",
        "Avoid planting tomatoes in the same location as potatoes or tomatoes from previous years to reduce the buildup of the pathogen in the soil",
        "Ensure proper spacing between plants to improve airflow and reduce humidity",
        "Avoid overhead irrigation, and use drip irrigation to keep the foliage dry",
        "Remove and destroy infected plant material immediately, including leaves, stems, and fruit",
        "Practice crop rotation, and select resistant tomato varieties if available"
    ],
    'solution': "Apply fungicides containing mefenoxam, chlorothalonil, or copper-based solutions at the first sign of infection and continue applications every 7-10 days, especially after rain. Infected plants should be removed and discarded to prevent further spread of the disease. For severe infestations, complete destruction of the affected plants may be necessary to protect surrounding crops. Ensure that proper sanitation practices are followed to avoid cross-contamination with tools and equipment."
},
    'Tomato___Leaf_Mold': {
    'description': "Leaf mold is a fungal disease caused by *Cladosporium fulvum*, which affects tomato plants, primarily the leaves. It begins as yellow or light green spots on the upper side of leaves, which then turn brown or gray. The underside of the leaves develops a fuzzy, olive-green mold. As the disease progresses, the affected leaves may curl, wither, and die. Leaf mold thrives in cool, humid conditions and can significantly reduce plant productivity if left unchecked.",
    'reasons': [
        "Fungus *Cladosporium fulvum*",
        "High humidity and poor air circulation",
        "Excessive watering or overhead irrigation that wets the foliage",
        "Cool temperatures (60-70°F) combined with high moisture levels",
        "Overcrowded planting, preventing proper airflow around the plants"
    ],
    'precautions': [
        "Ensure proper spacing between plants to promote good air circulation and reduce humidity around the foliage",
        "Avoid overhead irrigation and use drip irrigation to keep the leaves dry",
        "Prune and remove infected leaves regularly to reduce the spread of spores",
        "Mulch around the base of the plants to prevent soil splash from spreading the fungus",
        "Grow tomatoes in well-drained soil to prevent excess moisture buildup",
        "Choose resistant varieties if available, and rotate crops to avoid a buildup of the fungus"
    ],
    'solution': "To manage leaf mold, remove and destroy infected leaves and plant debris promptly to reduce the spread of the fungus. Apply fungicides containing copper or chlorothalonil as a preventive measure or at the first signs of infection. Maintain proper spacing, pruning, and irrigation practices to reduce environmental conditions that favor fungal growth. In severe cases, removing the most affected plants may be necessary to prevent the disease from spreading."
},
    'Tomato___Septoria_leaf_spot': {
    'description': "Septoria leaf spot is a fungal disease caused by *Septoria lycopersici* that affects tomato plants, primarily the leaves. It starts as small, dark spots with grayish centers and dark borders on the lower leaves. These spots expand as the disease progresses, often merging into larger lesions. Infected leaves eventually turn yellow and die, leading to defoliation. This reduces the plant’s ability to photosynthesize, which can significantly reduce yield and fruit quality.",
    'reasons': [
        "Fungus *Septoria lycopersici*",
        "Warm, wet conditions with frequent rainfall or high humidity",
        "Infected seeds, transplants, or plant debris from previous crops",
        "Splashing water that spreads fungal spores from the soil to the plant leaves",
        "Poor air circulation around plants, especially in crowded gardens"
    ],
    'precautions': [
        "Use disease-free seed and transplants to reduce the chance of introducing the pathogen",
        "Practice crop rotation, avoiding tomatoes or related solanaceous crops in the same area for 2-3 years",
        "Ensure proper spacing between plants to allow for good air circulation and reduce humidity around the foliage",
        "Avoid overhead irrigation and use drip irrigation to minimize water splash on the leaves",
        "Remove and destroy infected plant material, including fallen leaves, to prevent further spread",
        "Mulch around the base of the plants to prevent soil splash"
    ],
    'solution': "To control Septoria leaf spot, apply fungicides containing chlorothalonil, copper-based products, or mancozeb at the first signs of infection, and continue treatments at 7-10 day intervals. Remove and dispose of infected leaves and debris to prevent fungal spores from spreading. In severe cases, pruning affected stems and leaves can help reduce the disease's impact. Maintain proper spacing and watering techniques to prevent the spread of the disease."
},
    'Tomato___Spider_mites_Two-spotted_spider_mite': {
    'description': "The two-spotted spider mite (*Tetranychus urticae*) is a common pest that affects tomato plants. These tiny arachnids are difficult to see with the naked eye but cause significant damage by feeding on the undersides of leaves, extracting plant juices. The affected leaves develop stippling, turning yellow or bronzed, and may eventually dry out and fall off. In severe infestations, spider mites can cause defoliation, reduced plant growth, and yield loss. They also produce fine webbing around the plants.",
    'reasons': [
        "Two-spotted spider mite (*Tetranychus urticae*)",
        "Hot, dry conditions (temperatures above 80°F) with low humidity",
        "Overcrowded plants that limit air circulation",
        "Poor plant health, making them more susceptible to mite feeding",
        "Use of overhead irrigation that can spread mites to new plants"
    ],
    'precautions': [
        "Provide adequate spacing between plants to promote good air circulation and reduce humidity",
        "Regularly inspect plants for signs of mite infestation, particularly the undersides of leaves",
        "Use insecticidal soap or miticides to control early infestations",
        "Maintain proper watering and humidity levels, as spider mites thrive in dry conditions",
        "Avoid stressing plants through over-fertilization or improper watering",
        "Introduce natural predators, such as ladybugs or predatory mites, to control mite populations"
    ],
    'solution': "To control two-spotted spider mites, apply insecticidal soaps or miticides containing abamectin or neem oil. Treat the plants at the first sign of infestation, repeating every 7-10 days if necessary. Remove and destroy heavily infested leaves to reduce the mite population. Increase humidity around plants by using overhead sprinklers or misting the plants, as higher humidity can reduce mite activity. Regularly monitor plants for new signs of infestation, especially during hot, dry weather."
},
    'Tomato___Target_Spot': {
    'description': "Target spot is a fungal disease caused by *Corynespora cassiicola* that affects tomato plants. It typically starts as small, round, dark lesions with concentric rings or 'target-like' patterns on the leaves. As the disease progresses, the lesions expand and the tissue around them may turn yellow. Infected leaves may eventually die and fall off, leading to reduced plant vigor and fruit yield. This disease can also affect the stems and fruits, leading to further damage.",
    'reasons': [
        "Fungus *Corynespora cassiicola*",
        "Warm, humid conditions (temperatures between 75-85°F)",
        "Infected seeds, transplants, or plant debris from previous crops",
        "Overhead irrigation or water splash, which spreads fungal spores",
        "Crowded plants, leading to poor air circulation and higher humidity levels"
    ],
    'precautions': [
        "Use disease-free seed and transplants to prevent initial infection",
        "Practice crop rotation and avoid planting tomatoes or other solanaceous crops in the same area for 2-3 years",
        "Space plants adequately to promote air circulation and reduce humidity around the foliage",
        "Avoid overhead irrigation and use drip irrigation to minimize water splash on the leaves",
        "Remove and destroy infected leaves and plant debris immediately to prevent fungal spread",
        "Mulch around the base of the plants to prevent soil splash"
    ],
    'solution': "To control target spot, apply fungicides containing copper-based products, chlorothalonil, or mancozeb as a preventive measure or when the first symptoms appear. Treat the plants every 7-10 days during the growing season, particularly after rain. Remove and destroy any infected plant material to reduce the spread of the fungus. Improve plant spacing and adjust irrigation practices to avoid creating conditions favorable to fungal growth."
},
    'Tomato___Tomato_mosaic_virus': {
    'description': "Tomato mosaic virus (ToMV) is a viral disease that affects tomato plants, causing a range of symptoms. Initially, infected plants may show a mosaic pattern of light and dark green patches on the leaves, followed by leaf curling, wilting, and stunting. Infected plants may also display a reduction in fruit size and quality, with fruits developing uneven coloring or deformation. This virus can spread rapidly through infected seeds, transplants, or physical contact, and is often exacerbated by stress conditions like high temperatures and poor plant care.",
    'reasons': [
        "Virus *Tomato mosaic virus* (ToMV)",
        "Infected seeds or transplants",
        "Mechanical transmission through contact with contaminated tools, equipment, or workers' hands",
        "Infected plant material left in the soil or in garden debris",
        "Vectors like aphids or leafhoppers, which can spread the virus from plant to plant"
    ],
    'precautions': [
        "Use virus-free seeds and certified disease-free transplants",
        "Practice good sanitation by regularly disinfecting tools and equipment",
        "Remove and destroy infected plants immediately to prevent the spread of the virus",
        "Avoid handling plants when wet, as moisture can spread the virus more easily",
        "Control aphid populations, which can transmit the virus, using insecticides or introducing natural predators",
        "Rotate crops and avoid planting tomatoes in the same location year after year to reduce virus buildup"
    ],
    'solution': "There is no cure for tomato mosaic virus once a plant is infected, so the focus is on prevention and management. Remove and destroy all infected plants and their debris. Disinfect tools and equipment between uses to avoid spreading the virus. If symptoms are noticed, early removal of infected plants can help reduce the spread to healthy plants. Control aphids and other potential vectors with appropriate pest control methods. To prevent future outbreaks, use resistant tomato varieties if available."
},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
    'description': "Tomato Yellow Leaf Curl Virus (TYLCV) is a viral disease transmitted by whiteflies that causes significant damage to tomato plants. Infected plants show symptoms such as yellowing and curling of the leaves, stunted growth, and reduced fruit yield. The leaves may become cupped and brittle, and the plants may develop a general wilting appearance. Infected plants often produce small, deformed fruits. TYLCV is most commonly found in regions with warm temperatures and where whiteflies are abundant.",
    'reasons': [
        "Virus *Tomato yellow leaf curl virus* (TYLCV)",
        "Transmission by whiteflies, primarily *Bemisia tabaci*",
        "Infected transplants, seeds, or plant debris left in the field",
        "Whiteflies spreading the virus between infected and healthy plants",
        "High temperatures (above 80°F) and high humidity, which favor whitefly activity and virus transmission"
    ],
    'precautions': [
        "Use TYLCV-resistant tomato varieties, if available",
        "Inspect and select disease-free transplants",
        "Control whitefly populations using insecticides or natural predators such as parasitoid wasps (e.g., *Encarsia formosa*)",
        "Avoid planting tomatoes in areas with high whitefly populations or where the virus has been present in the past",
        "Use reflective mulches or row covers to discourage whitefly activity around plants",
        "Remove and destroy infected plants immediately to prevent further spread of the virus"
    ],
    'solution': "There is no cure for Tomato Yellow Leaf Curl Virus once a plant is infected. The primary strategy is to manage whitefly populations and prevent the spread of the virus. Apply insecticides that target whiteflies, or introduce biological controls such as predatory insects. Infected plants should be removed and destroyed to minimize the risk of further spread. Preventative measures such as using resistant varieties, controlling whitefly populations, and selecting clean plant material can help reduce the impact of TYLCV in future plantings."
}
}
# Modified disease_chatbot function with reordered processing
def disease_chatbot():
    st.header("🌱 Plant Disease Chatbot")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Handle conversation flow first to update chat history
    with st.container():
        if st.session_state.chatbot_step == 'plant_selection':
            plant_selection()
        elif st.session_state.chatbot_step == 'disease_selection':
            disease_selection()
        elif st.session_state.chatbot_step == 'disease_info':
            display_disease_info()

    # Display chat history after processing conversation steps
    for message in st.session_state.chat_history:
        if message['type'] == 'bot':
            st.markdown(f"""
            <div class="bot-message">
                <div class="message-content">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="user-message">
                <div class="message-content">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)

# Updated CSS for chat interface
st.markdown("""
<style>
    .bot-message {
        background: #4CAF50;
        color: white;
        border-radius: 15px 15px 15px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .user-message {
        background: #e0e0e0;
        color: black;
        border-radius: 15px 15px 0 15px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .message-content {
        line-height: 1.4;
        font-size: 15px;
    }
    .chat-container {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        min-height: 400px;
    }
    .disease-info {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def plant_selection():
    plants = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", 
             "Peach", "Pepper", "Potato", "Raspberry", "Soybean", 
             "Squash", "Strawberry", "Tomato"]
    
    # Bot prompt
    if not any(msg['content'] == "Please select a plant type:" for msg in st.session_state.chat_history):
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': "Please select the plant type you're interested in:"
        })
    
    # User selection buttons
    cols = st.columns(2)
    for idx, plant in enumerate(plants):
        with cols[idx % 2]:
            if st.button(plant, key=f"plant_{plant}"):
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': plant
                })
                st.session_state.selected_plant = plant
                st.session_state.chatbot_step = 'disease_selection'
                st.experimental_rerun()

def disease_selection():
    # Bot prompt
    if not any(msg['content'] == f"Select disease for {st.session_state.selected_plant}:" for msg in st.session_state.chat_history):
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': f"Now select the specific disease for {st.session_state.selected_plant}:"
        })
    
    diseases = [d for d in CLASS_NAMES if d.startswith(st.session_state.selected_plant)]
    diseases = [d.split('___')[1].replace('_', ' ') for d in diseases]
    
    # User selection buttons
    for disease in diseases:
        if st.button(disease, key=f"disease_{disease}"):
            st.session_state.chat_history.append({
                'type': 'user',
                'content': disease
            })
            st.session_state.selected_disease = f"{st.session_state.selected_plant}___{disease.replace(' ', '_')}"
            st.session_state.chatbot_step = 'disease_info'
            st.experimental_rerun()
    
    # Back button
    if st.button("← Back to Plant Selection"):
        st.session_state.chatbot_step = 'plant_selection'
        st.experimental_rerun()

# Updated CSS for `.disease-info`
st.markdown("""
<style>
    .disease-info {
        background: black; /* Black background for better visibility */
        color: white; /* White text for contrast */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disease-info h3 {
        color: #4CAF50; /* Optional: Green color for headings */
    }
    .disease-info ul {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

def display_disease_info():
    try:
        disease = st.session_state.selected_disease
        plant_type = disease.split('___')[0]
        healthy_key = f"{plant_type}___healthy"
        
        info = DISEASE_INFO.get(
            disease, 
            DISEASE_INFO.get(
                healthy_key, 
                {
                    'description': 'No information available',
                    'reasons': ['Unknown'],
                    'precautions': ['Consult an agricultural expert'],
                    'solution': 'Please contact plant health specialists for diagnosis'
                }
            )
        )

        disease_name = disease.split('___')[1].replace('_', ' ')
        content = f"""
        <div class='disease-info'>
            <h3>{disease_name}</h3>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>Common Causes:</strong></p>
            <ul>
                {''.join(f'<li>{reason}</li>' for reason in info['reasons'])}
            </ul>
            <p><strong>Prevention Tips:</strong></p>
            <ul>
                {''.join(f'<li>{precaution}</li>' for precaution in info['precautions'])}
            </ul>
            <p><strong>Treatment Solution:</strong> {info['solution']}</p>
        </div>
        """
        
        # Add to chat history only once
        if not any(msg['content'] == content for msg in st.session_state.chat_history):
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': content
            })
        
        # Navigation buttons with unique keys
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Diseases", key="back_to_diseases"):
                st.session_state.chatbot_step = 'disease_selection'
                st.experimental_rerun()
        with col2:
            if st.button("↩ New Analysis", key="new_analysis"):
                st.session_state.chatbot_step = 'plant_selection'
                st.session_state.chat_history = []
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Error displaying disease information: {str(e)}")
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': "⚠️ Oops! I encountered an error. Let's start over..."
        })
        st.session_state.chatbot_step = 'plant_selection'
        st.experimental_rerun()

    # Add to chat history if not already present
    if not any(msg['content'] == content for msg in st.session_state.chat_history):
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': content
        })
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Diseases"):
            st.session_state.chatbot_step = 'disease_selection'
            st.experimental_rerun()
    with col2:
        if st.button("↩ New Analysis"):
            st.session_state.chatbot_step = 'plant_selection'
            st.session_state.chat_history = []
            st.experimental_rerun()

# Database operations
def get_active_notifications():
    c.execute('''SELECT title, content FROM notifications 
               WHERE is_active = TRUE ORDER BY created_at DESC LIMIT 3''')
    return c.fetchall()

# Authentication functions
def create_user(name, email, password, phone):
    try:
        hashed_password = pbkdf2_sha256.hash(password)
        c.execute('INSERT INTO users (name, email, password, phone) VALUES (?, ?, ?, ?)',
                 (name, email, hashed_password, phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(email, password):
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    if user and pbkdf2_sha256.verify(password, user[3]):
        return user
    return None

# UI Components
def loading_spinner():
    return st.markdown("""
    <div class="fade-in" style="text-align: center; padding: 20px;">
        <div class="spinner"></div>
        <style>
            .spinner {
                border: 4px solid #f3f3f3;
                border-radius: 50%;
                border-top: 4px solid #4CAF50;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </div>
    """, unsafe_allow_html=True)

# Page navigation
def show_navigation():
    st.sidebar.title("🌱 Agrodoc Navigation")
    pages = ["Home", "Predictions", "Chatbot", "Reviews", "About", "Account"]
    if st.session_state.user:
        page = st.sidebar.radio("Menu", pages)
    else:
        page = st.sidebar.radio("Menu", ["Home", "About", "Account"])
    st.session_state.page = page

# Notification System
def dynamic_notifications():
    if st.session_state.user and not st.session_state.notification_shown:
        try:
            c.execute('''SELECT show_notifications FROM users WHERE id = ?''', 
                     (st.session_state.user[0],))
            show_notifications = c.fetchone()[0]
            
            if show_notifications:
                notifications = get_active_notifications()
                if notifications:
                    with st.expander("🔔 Latest Updates", expanded=True):
                        for title, content in notifications:
                            st.markdown(f"""
                            <div class="fade-in card">
                                <h4>{title}</h4>
                                <p>{content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        if st.button("Don't show these again"):
                            c.execute('''UPDATE users SET show_notifications = 0 WHERE id = ?''',
                                     (st.session_state.user[0],))
                            conn.commit()
                            st.session_state.notification_shown = True
                            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error loading notifications: {str(e)}")

# Modified Home Page
def home_page():
    st.title("🌱 Agrodoc - Plant Disease Detection")
    st.markdown("## Instantly identify plant disease")
    
    if st.session_state.user:
        if st.session_state.prediction_done:
            st.success("✅ Analysis complete! Click the button below to view predictions.")
            if st.button("View Predictions"):
                st.session_state.page = 'Predictions'
                st.session_state.prediction_done = False
                st.experimental_rerun()
            return

        uploaded_file = st.file_uploader(
            "Upload a plant image", 
            type=["jpg", "png", "jpeg"],
            disabled=st.session_state.processing
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyse", disabled=st.session_state.processing):
                st.session_state.processing = True
                st.session_state.uploaded_file = uploaded_file
                process_image(uploaded_file)

        if st.session_state.processing:
            loading_spinner()
            st.write("Analyzing image... Please wait.")

def process_image(uploaded_file):
    os.makedirs('uploads', exist_ok=True)
    
    try:
        st.session_state.latest_prediction = None
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join('uploads', filename)
        image.save(file_path)
        
        img = cv2.resize(img_array, (128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        
        with st.spinner('Analyzing plant health...'):
            prediction = model.predict(input_arr)
            time.sleep(1)
            
        confidence = np.max(prediction)
        result_index = np.argmax(prediction)
        disease = CLASS_NAMES[result_index]
        
        c.execute('''INSERT INTO predictions 
                   (user_id, image_path, prediction, confidence) 
                   VALUES (?, ?, ?, ?)''',
                 (st.session_state.user[0], file_path, disease, float(confidence)))
        conn.commit()
        
        st.session_state.latest_prediction = (image, disease, confidence)
        st.session_state.prediction_done = True

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        st.session_state.processing = False
        st.experimental_rerun()

def prediction_page():
    st.title("🔍 Prediction Results")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = None

    if st.session_state.latest_prediction:
        image, disease, confidence = st.session_state.latest_prediction
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"**Prediction:** {disease}")
        st.info(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning("No predictions available. Upload an image on the Home page.")

    st.subheader("Filter Predictions by Date")
    selected_date = st.date_input("Select a date to view historical predictions")
    
    if selected_date:
        c.execute('''SELECT timestamp, prediction, confidence 
                   FROM predictions 
                   WHERE user_id = ? AND DATE(timestamp) = ?
                   ORDER BY timestamp DESC''',
                 (st.session_state.user[0], selected_date))
        filtered_history = c.fetchall()
        
        if filtered_history:
            st.subheader(f"Predictions for {selected_date}")
            plot_history(filtered_history)
        else:
            st.info(f"No predictions found for {selected_date}")
        return

    st.subheader("📈 Prediction History")
    
    c.execute('''SELECT COUNT(*) FROM predictions 
               WHERE user_id = ?''',
             (st.session_state.user[0],))
    total_predictions = c.fetchone()[0]
    
    if total_predictions == 0:
        st.info("No prediction history found. Make your first prediction on the Home page!")
        return

    items_per_page = 8
    total_pages = (total_predictions + items_per_page - 1) // items_per_page
    
    col1, col2, col3 = st.columns([2, 4, 2])
    with col1:
        if st.button("Previous", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
    with col3:
        if st.button("Next", disabled=st.session_state.current_page >= total_pages-1):
            st.session_state.current_page += 1
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")

    offset = st.session_state.current_page * items_per_page
    c.execute('''SELECT timestamp, prediction, confidence 
               FROM predictions WHERE user_id = ? 
               ORDER BY timestamp DESC 
               LIMIT ? OFFSET ?''',
             (st.session_state.user[0], items_per_page, offset))
    history = c.fetchall()

    if history:
        plot_history(history)
    else:
        st.info("No predictions found for this page")

    if st.button("← Back to Home", key='back_home'):
        st.session_state.page = 'Home'
        st.session_state.uploaded_file = None
        st.experimental_rerun()

def plot_history(history):
    dates = [row[0] for row in history]
    confidences = [row[2] for row in history]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, confidences, marker='o', color='#4CAF50')
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=10)
    ax.set_title("Prediction Confidence History", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write("**Recent Predictions:**")
    for row in history:
        st.write(f"- {row[0][:10]}: {row[1]} ({row[2]:.2%})")

# Review Page
def review_page():
    st.title("⭐ User Reviews")
    
    st.subheader("Example Reviews")
    sample_reviews = [
        {"rating": 5, "text": "Excellent service! Accurate detection saved my crops!", "date": "2024-03-01"},
        {"rating": 4, "text": "Very user-friendly interface and fast results", "date": "2024-03-05"},
        {"rating": 5, "text": "Best plant disease detection app I've used", "date": "2024-03-10"}
    ]
    
    for review in sample_reviews:
        st.markdown(f"""
        <div class="fade-in card">
            <div style="display: flex; justify-content: space-between;">
                <div>{''.join(['★'] * review['rating'])}{''.join(['☆'] * (5 - review['rating']))}</div>
                <small>{review['date']}</small>
            </div>
            <p style="margin-top: 10px;"><em>{review['text']}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.user:
        st.subheader("Submit Your Review")
        with st.form("review_form"):
            rating = st.slider("Rate your experience", 1, 5, 5)
            review_text = st.text_area("Write your review")
            submitted = st.form_submit_button("Submit Review")
            
            if submitted:
                c.execute('''INSERT INTO reviews (user_id, rating, review)
                          VALUES (?, ?, ?)''',
                         (st.session_state.user[0], rating, review_text))
                conn.commit()
                st.success("Thank you for your review!")
    else:
        st.warning("Please login to submit a review")
    
    st.subheader("Recent Community Reviews")
    c.execute('''SELECT users.name, reviews.rating, reviews.review, reviews.timestamp
               FROM reviews JOIN users ON reviews.user_id = users.id
               ORDER BY timestamp DESC LIMIT 10''')
    user_reviews = c.fetchall()
    
    for name, rating, text, date in user_reviews:
        st.markdown(f"""
        <div class="fade-in card">
            <div style="display: flex; justify-content: space-between;">
                <strong>{name}</strong>
                <small>{date[:10]}</small>
            </div>
            <div style="margin: 10px 0;">{''.join(['★'] * rating)}{''.join(['☆'] * (5 - rating))}</div>
            <p>{text}</p>
        </div>
        """, unsafe_allow_html=True)

# About Page
def about_page():
    st.title("🌿 About Agrodoc")
    
    with st.container():
        st.header("Technology Deep Dive")
        tab1, tab2, tab3 = st.tabs(["CNN Architecture", "Workflow", "Case Studies"])
        
        with tab1:
            st.markdown("""
                <div class="tab-content">
                    <h4>Convolutional Neural Network Architecture</h4>
    <p>Our CNN architecture consists of:</p>
    <ul>
        <li><strong>Input Layer (128x128x3):</strong> 
            The input layer takes in images with a resolution of 128x128 pixels and 3 color channels (RGB). The input shape is (128, 128, 3), where:
            <ul>
                <li>128 is the height of the image</li>
                <li>128 is the width of the image</li>
                <li>3 represents the RGB channels of the image.</li>
            </ul>
        </li>
        <li><strong>Convolutional Blocks with ReLU activation:</strong> 
            Convolutional layers are used to detect patterns in the input image by applying filters (kernels) over the image. The convolutional layers learn features such as edges, shapes, and textures. ReLU (Rectified Linear Unit) activation is applied to the output of each convolutional layer to introduce non-linearity, which allows the network to learn more complex patterns.
            <ul>
                <li>Each convolutional block typically consists of multiple convolutional layers followed by a ReLU activation function.</li>
                <li>Filters, such as 3x3 or 5x5, are applied to the input image to extract spatial features, and the output is a set of feature maps.</li>
            </ul>
        </li>
        <li><strong>Max Pooling Layers:</strong> 
            Max pooling is used to down-sample the feature maps generated by the convolutional layers. This reduces the spatial dimensions (height and width) of the feature maps while retaining the most important features. A typical pooling operation uses a 2x2 window with a stride of 2, which means the output is halved in size.
            <ul>
                <li>Max pooling reduces the computational load and helps the network focus on the most significant features.</li>
                <li>This also helps in mitigating overfitting by making the model less sensitive to small variations in the input data.</li>
            </ul>
        </li>
        <li><strong>Batch Normalization:</strong> 
            Batch normalization helps in stabilizing and speeding up the training process. It normalizes the output of each layer by adjusting its mean and variance. This prevents the network from facing issues like vanishing or exploding gradients, which are common in deep networks.
            <ul>
                <li>It normalizes the activations of the layers so that they have a mean of 0 and variance of 1.</li>
                <li>By normalizing the data, the model trains faster and is less likely to get stuck in local minima during optimization.</li>
            </ul>
        </li>
        <li><strong>Dropout Regularization:</strong> 
            Dropout is a regularization technique used to prevent overfitting. During training, a fraction of the neurons (e.g., 25% to 50%) is randomly set to zero, effectively "dropping" them out of the network for a particular iteration. This forces the model to generalize better by making it less reliant on any specific neuron.
            <ul>
                <li>Dropout is applied after each convolutional block and/or fully connected layer to reduce the model's tendency to memorize the training data.</li>
                <li>This also improves the network’s ability to perform well on unseen data (generalization).</li>
            </ul>
        </li>
        <li><strong>Fully Connected Layers:</strong> 
            After the convolutional and pooling layers have extracted high-level features from the image, the fully connected (dense) layers are used to make final predictions. These layers connect every neuron to every other neuron in the previous layer. 
            <ul>
                <li>The fully connected layers combine the features learned in earlier layers and generate output predictions.</li>
                <li>Typically, a softmax activation function is applied in the final layer for multi-class classification tasks (i.e., when the model needs to predict one of several categories), or a sigmoid activation function for binary classification tasks.</li>
            </ul>
        </li>
    </ul>
</div>
<style>
    .tab-content {
        padding: 15px;
        border-radius: 8px;
        background: black; /* Light gray background for better contrast */
        color: white; /* Dark text for readability */
        margin-top: 10px;
    }
                </style>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="tab-content">
                <h4>Image Processing Pipeline</h4>
                <div style="display: grid; gap: 15px;">
                    <div class="card">
                        <h5>1. Image Preprocessing</h5>
                        <p>Resizing → Normalization → Augmentation</p>
                    </div>
                    <div class="card">
                        <h5>2. Feature Extraction</h5>
                        <p>Convolution → Activation → Pooling</p>
                    </div>
                    <div class="card">
                        <h5>3. Classification</h5>
                        <p>Flattening → Dense Layers → Softmax</p>
                    </div>
                </div>
            </div>
            <style>
    .tab-content {
        padding: 15px;
        border-radius: 8px;
        background: black; /* Light gray background for better contrast */
        color: white; /* Dark text for readability */
        margin-top: 10px;
    }
</style>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="tab-content">
                <h4>Real-World Applications</h4>
                <div class="card">
                    <h5>Case Study: Tomato Disease Detection</h5>
                    <p style="margin-top: 15px;">Achieved 98.2% accuracy in detecting 10 common tomato diseases</p>
                </div>
            </div>
            <style>
    .tab-content {
        padding: 15px;
        border-radius: 8px;
        background: black; /* Light gray background for better contrast */
        color: white; /* Dark text for readability */
        margin-top: 10px;
    }
</style>
            """, unsafe_allow_html=True)

    st.header("Development Team")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Leader")
        st.markdown("""
        <div class="card">
            <h4>Saurav Varshney</h4>
            <p><a href="https://github.com/Saurav-Lumfy" target="_blank">GitHub Profile</a></p>
            <p><a href="https://linkedin.com/in/sauravvarshney" target="_blank">LinkedIn Profile</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("ML Developer")
        st.markdown("""
        <div class="card">
            <h4>Yaashi Sharma</h4>
            <p>Deep Learning Specialist</p>
            <p><a href="https://github.com/Saurav-Lumfy" target="_blank">GitHub Profile</a></p>
            <p><a href="https://linkedin.com/in/sauravvarshney" target="_blank">LinkedIn Profile</a></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Team Members")
        st.markdown("""
        <div class="card">
            <h4>Shisha Rajput</h4>
            <p>Backend Developer</p>
            <p><a href="https://github.com/Saurav-Lumfy" target="_blank">GitHub Profile</a></p>
            <p><a href="https://linkedin.com/in/sauravvarshney" target="_blank">LinkedIn Profile</a></p>
        </div>
        <div class="card" style="margin-top: 15px;">
            <h4>Vashu</h4>
            <p>Frontend Developer</p>
            <p><a href="https://github.com/Saurav-Lumfy" target="_blank">GitHub Profile</a></p>
            <p><a href="https://linkedin.com/in/sauravvarshney" target="_blank">LinkedIn Profile</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="margin-top: 20px;">
        <h4>Institution</h4>
        <p>G.L Bajaj Institute of Technology and Management, Greater Noida</p>
        <p>B.Tech in Computer Science & Engineering</p>
    </div>
    """, unsafe_allow_html=True)

# Account Management
def account_page():
    st.title("🔐 Account Management")
    
    if st.session_state.user:
        user_id = st.session_state.user[0]
        
        # Account Deletion Section
        with st.expander("🗑️ Account Deletion", expanded=False):
            st.markdown("""
            <style>
                .danger-zone { border-left: 5px solid #ff4b4b; padding: 10px; background: #fff0f0; }
            </style>
            <div class="danger-zone">
                <h4>⚠️ Danger Zone</h4>
                <p>Permanently delete your account and all associated data. This action cannot be undone!</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.checkbox("I understand all consequences and wish to proceed"):
                if st.button("Delete My Account", key="delete_account"):
                    try:
                        # Delete all user-related data
                        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
                        c.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
                        c.execute('DELETE FROM reviews WHERE user_id = ?', (user_id,))
                        conn.commit()
                        st.session_state.user = None
                        st.success("Account deleted successfully. Redirecting to home page...")
                        time.sleep(2)
                        st.session_state.page = 'Home'
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting account: {str(e)}")

        # Password Reset Section
        with st.expander("🔒 Password Reset", expanded=False):
            with st.form("password_reset"):
                current_password = st.text_input("Current Password", type='password')
                new_password = st.text_input("New Password", type='password')
                confirm_password = st.text_input("Confirm New Password", type='password')
                
                if st.form_submit_button("Update Password"):
                    if pbkdf2_sha256.verify(current_password, st.session_state.user[3]):
                        if new_password == confirm_password:
                            hashed_password = pbkdf2_sha256.hash(new_password)
                            c.execute('UPDATE users SET password = ? WHERE id = ?', 
                                     (hashed_password, user_id))
                            conn.commit()
                            st.success("Password updated successfully!")
                        else:
                            st.error("New passwords do not match!")
                    else:
                        st.error("Incorrect current password!")

        # Username Update Section
        with st.expander("👤 Update Profile", expanded=False):
            with st.form("username_update"):
                new_name = st.text_input("New Display Name", value=st.session_state.user[1])
                if st.form_submit_button("Update Profile"):
                    c.execute('UPDATE users SET name = ? WHERE id = ?', 
                             (new_name, user_id))
                    conn.commit()
                    # Update session state
                    st.session_state.user = list(st.session_state.user)
                    st.session_state.user[1] = new_name
                    st.success("Profile updated successfully!")

        # Phone Number Verification Section
        with st.expander("📱 Phone Verification", expanded=False):
            if 'otp' not in st.session_state:
                st.session_state.otp = None
                st.session_state.otp_expiry = None
                
            current_phone = st.session_state.user[4] or "Not set"
            st.write(f"Current verified number: **{current_phone}**")
            
            with st.form("phone_verification"):
                new_phone = st.text_input("New Phone Number (with country code)", 
                                          value="+91")
                if st.form_submit_button("Send Verification Code"):
                    if new_phone != current_phone:
                        # Generate 6-digit OTP
                        st.session_state.otp = str(random.randint(100000, 999999))
                        st.session_state.otp_expiry = time.time() + 300  # 5 minutes
                        
                        # In real implementation, use SMS gateway like Twilio
                        st.success(f"OTP sent to {new_phone}: **{st.session_state.otp}**")
                        st.session_state.temp_phone = new_phone
                    else:
                        st.warning("This is already your current number!")

            if st.session_state.otp:
                with st.form("otp_verification"):
                    entered_otp = st.text_input("Enter 6-digit OTP")
                    if st.form_submit_button("Verify OTP"):
                        if time.time() > st.session_state.otp_expiry:
                            st.error("OTP has expired. Please request a new one.")
                        elif entered_otp == st.session_state.otp:
                            c.execute('UPDATE users SET phone = ? WHERE id = ?',
                                     (st.session_state.temp_phone, user_id))
                            conn.commit()
                            # Update session state
                            st.session_state.user = list(st.session_state.user)
                            st.session_state.user[4] = st.session_state.temp_phone
                            st.session_state.otp = None
                            st.success("Phone number verified and updated!")
                        else:
                            st.error("Invalid OTP. Please try again.")

        # Preferences Section
        with st.expander("⚙️ Preferences", expanded=True):
            c.execute('SELECT show_notifications FROM users WHERE id = ?', (user_id,))
            show_notifications = c.fetchone()[0]
            
            notifications = st.checkbox("Show update notifications", value=show_notifications)
            if notifications != show_notifications:
                c.execute('UPDATE users SET show_notifications = ? WHERE id = ?',
                         (notifications, user_id))
                conn.commit()
                st.success("Preferences updated!")

        # Logout Button
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.page = 'Home'
            st.experimental_rerun()
            
    else:
        login_register_tab()

def login_register_tab():
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            if st.form_submit_button("Login"):
                user = verify_user(email, password)
                if user:
                    st.session_state.user = user
                    st.session_state.page = 'Home'
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone Number")
            password = st.text_input("Password", type='password')
            if st.form_submit_button("Create Account"):
                if create_user(name, email, password, phone):
                    st.success("Account created! Please login")
                    st.session_state.page = 'Account'
                    st.experimental_rerun()
                else:
                    st.error("Email already exists")


# Main app flow
def main():
    show_navigation()
    dynamic_notifications()
    
    if st.session_state.page == 'Home':
        home_page()
    elif st.session_state.page == 'Predictions':
        prediction_page()
    elif st.session_state.page == 'Chatbot':
        disease_chatbot()
    elif st.session_state.page == 'Reviews':
        review_page()
    elif st.session_state.page == 'About':
        about_page()
    elif st.session_state.page == 'Account':
        account_page()

if __name__ == '__main__':
    main()