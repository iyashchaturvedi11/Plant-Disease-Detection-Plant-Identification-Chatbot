import os
import sys
import csv
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from datetime import datetime
import warnings
from fuzzywuzzy import (process)
warnings.filterwarnings("ignore")

DATA_PLANT = 'Data'
DATA_DISEASE = 'Plant_Disease_Data'
MODEL_FILE_PREFIX_PLANT = 'plant_model_v'
MODEL_FILE_PREFIX_DISEASE = 'disease_model_v'
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANT_DETAILS_CSV = 'Plant_Details.csv'

history = []

fun_facts = {
    "tomato": "Fun Fact: Tomatoes are technically fruits, not vegetables!",
    "potato": "Fun Fact: Potatoes were the first vegetable grown in space!",
    "almond": "Fun Fact: Almonds are seeds, not true nuts!",
    "banana": "Fun Fact: Bananas are berries, but strawberries aren't!",
    "cardamom": "Fun Fact: Cardamom is known as the 'Queen of Spices'",
    "cherry": "Fun Fact: Cherries can be used to make natural dye!",
    "chilli": "Fun Fact: The spiciness of chilli comes from capsaicin",
    "clove": "Fun Fact: Cloves have been used as a traditional dental remedy!",
    "coconut": "Fun Fact: Coconut water can be used as a natural IV in emergencies!",
    "coffee plant": "Fun Fact: Coffee beans are actually seeds of a fruit!",
    "cotton": "Fun Fact: Cotton fibers were first cultivated over 7,000 years ago!",
    "cucumber": "Fun Fact: Cucumbers are 95% water!",
    "fox_nut": "Fun Fact: Fox nuts are considered superfoods and used in fasting meals!",
    "gram": "Fun Fact: Chickpeas, or gram, are a staple in Indian cuisine!",
    "jowar": "Fun Fact: Jowar (sorghum) is naturally gluten-free!",
    "jute": "Fun Fact: Jute is called the 'Golden Fiber' due to its color and economic value!",
    "lemon": "Fun Fact: Lemons float in water because they are less dense!",
    "maize": "Fun Fact: Corn was domesticated over 9,000 years ago!",
    "mustard-oil": "Fun Fact: Mustard oil has been used in India for centuries for cooking and massage!",
    "olive-tree": "Fun Fact: Olive trees can live for over 1,000 years!",
    "papaya": "Fun Fact: Papaya contains an enzyme called papain which tenderizes meat!",
    "pearl_millet": "Fun Fact: Pearl millet (Bajra) is highly drought-resistant!",
    "pineapple": "Fun Fact: Pineapples take about 2 years to grow!",
    "rice": "Fun Fact: Rice is the staple food for more than half of the world's population!",
    "soyabean": "Fun Fact: Soybeans are a complete protein and a staple in plant-based diets!",
    "sugarcane": "Fun Fact: Sugarcane can grow up to 6 meters tall!",
    "sunflower": "Fun Fact: Sunflowers track the sun's movement throughout the day!",
    "tea": "Fun Fact: Tea is the second most consumed beverage in the world after water!",
    "tobacco-plant": "Fun Fact: Tobacco leaves were used by indigenous people for ceremonial purposes!",
    "vigna-radiati": "Fun Fact: Mung beans are rich in protein and commonly sprouted for salads!",
    "wheat": "Fun Fact: Wheat has been cultivated for over 10,000 years!"
}

disease_tips = {
    "American Bollworm on Cotton": "Remove affected bolls, use pheromone traps, and apply appropriate insecticides.",
    "RedRust sugarcane": "Remove infected leaves and apply recommended fungicides.",
    "Rice Blast": "Use resistant varieties, ensure proper spacing, and apply fungicides as needed.",
    "Anthracnose on Cotton": "Remove infected plant parts, avoid overhead watering, and use fungicide sprays.",
    "Army worm": "Handpick larvae if possible, use biological control or insecticides.",
    "Sugarcane Healthy": "No issues detected. Maintain proper care and irrigation.",
    "bacterial_blight in Cotton": "Remove infected areas, improve drainage, and apply copper-based bactericides.",
    "thirps on cotton": "Spray neem oil or appropriate insecticides; encourage natural predators.",
    "Becterial Blight in Rice": "Remove infected plants, improve water drainage, and apply copper sprays.",
    "Tungro": "Use disease-free seedlings and resistant varieties; control vector insects.",
    "Ibollrot on Cotton": "Remove infected bolls and maintain field sanitation.",
    "Wheat aphid": "Monitor fields, encourage ladybugs, and use approved insecticides if necessary.",
    "bollworm on Cotton": "Monitor regularly, use pheromone traps, and apply biological control or insecticides.",
    "Wheat black rust": "Use resistant varieties and apply fungicides if needed.",
    "Brownspot": "Apply recommended fungicides and maintain proper nutrition and irrigation.",
    "Wheat Brown leaf Rust": "Grow resistant wheat varieties and apply fungicide sprays.",
    "Common_Rust": "Apply fungicides and remove infected plant debris.",
    "Wheat leaf blight": "Use resistant varieties and fungicide sprays; avoid excess nitrogen.",
    "Cotton Aphid": "Encourage natural predators and apply insecticidal soaps if needed.",
    "Wheat mite": "Apply miticides and practice crop rotation.",
    "cotton mealy bug": "Use insecticidal soaps or neem oil; encourage predatory insects.",
    "Wheat powdery mildew": "Apply sulfur or fungicides; ensure proper spacing for airflow.",
    "cotton whitefly": "Use yellow sticky traps and insecticidal sprays if infestation is high.",
    "Wheat scab": "Plant resistant varieties, rotate crops, and avoid excessive nitrogen.",
    "Flag Smut": "Use resistant wheat varieties and treat seeds before planting.",
    "Wheat Stem fly": "Rotate crops, destroy residues, and apply insecticides if necessary.",
    "Gray_Leaf_Spot": "Apply fungicides and maintain field sanitation.",
    "Wheat_Yellow_Rust": "Use resistant varieties and fungicide treatments as needed.",
    "Healthy cotton": "No issues detected. Maintain proper watering and nutrition.",
    "Wilt": "Remove infected plants, avoid waterlogging, and apply fungicides if needed.",
    "Healthy Maize": "No issues detected. Ensure regular care and monitoring.",
    "Yellow Rust Sugarcane": "Use resistant varieties and apply fungicides if needed.",
    "Healthy Wheat": "No issues detected. Maintain proper irrigation and nutrition.",
    "Leaf Curl": "Remove affected leaves, control vector insects, and apply recommended sprays.",
    "Leaf smut": "Apply fungicides and use disease-free seeds.",
    "maize ear rot": "Remove infected ears and apply fungicides during tasseling.",
    "maize fall armyworm": "Handpick larvae if possible, use biopesticides or approved insecticides.",
    "Imaize stem borer": "Use resistant varieties, destroy residues, and apply biological control or insecticides.",
    "Mosaic sugarcane": "Use virus-free seedlings and control vector insects.",
    "pink bollworm in cotton": "Remove affected bolls and use pheromone traps or insecticides.",
    "red cotton bug": "Handpick insects or use approved insecticides; maintain field hygiene.",
    "RedRot sugarcane": "Remove infected parts, maintain proper drainage, and apply fungicides."
}


def type_print(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def colored_confidence(conf):
    if conf > 0.8:
        return f"\033[92m{conf * 100:.2f}%\033[0m"
    elif conf > 0.5:
        return f"\033[93m{conf * 100:.2f}%\033[0m"
    else:
        return f"\033[91m{conf * 100:.2f}%\033[0m"


plant_details = {}
if os.path.exists(PLANT_DETAILS_CSV):
    with open(PLANT_DETAILS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_details[row['CropName']] = row


def get_plant_info(plant_name):
    info = plant_details.get(plant_name)
    if info:
        lines = [f"{k}: {v}" for k, v in info.items() if k != 'CropName']
        return "\n".join(lines)
    else:
        return " Let me know if you want to know more about this plant"


def get_tip(name, mode):
    if mode == 'plant':
        return get_plant_info(name)
    else:
        return disease_tips.get(name, "Follow proper disease management")


def greet():
    hour = datetime.now().hour
    if hour < 12:
        type_print("Good morning! Let's take care of your plants today!")
    elif hour < 18:
        type_print("Good afternoon! Ready to check your plants?")
    else:
        type_print("Good evening! Let's see how your garden is doing!")


def prepare_plant_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=DATA_PLANT, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print(f"Plant dataset: {len(dataset)} images across {len(idx_to_class)} classes")
    return dataset, loader, idx_to_class


def prepare_disease_dataset():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DISEASE, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DISEASE, 'validation'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    print(f"Disease dataset: {len(train_dataset)} training images across {len(idx_to_class)} classes")
    return train_dataset, val_dataset, train_loader, val_loader, idx_to_class


def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    return model, criterion, optimizer


def load_latest_model(model_prefix, model_obj):
    existing_models = [f for f in os.listdir() if f.startswith(model_prefix)]
    if existing_models:
        latest_model = sorted(existing_models)[-1]
        model_obj.load_state_dict(torch.load(latest_model, map_location=DEVICE))
        print(f"\nLoaded existing model: {latest_model}")
        return True
    return False


def predict_image(model, idx_to_class, image_path, threshold=0.5):
    model.eval()
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return None, None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = nn.Softmax(dim=1)(outputs)
        conf, pred_idx = torch.max(probs, 1)
        conf_val = conf.item()
        label = idx_to_class[pred_idx.item()]
        if conf_val < threshold:
            return "Unknown", conf_val
        return label, conf_val


def handle_question(user_text):
    text = user_text.lower()
    response = "I'm not sure, but I can classify an image for you!"
    plant_names = list(plant_details.keys())
    match, score = process.extractOne(text, plant_names)
    if score > 70:
        response = f"Here's what I know about {match}:\n{get_plant_info(match)}"
    else:
        disease_names = list(disease_tips.keys())
        match2, score2 = process.extractOne(text, disease_names)
        if score2 > 70:
            response = f"Advice for {match2}: {disease_tips[match2]}"
        elif "care" in text or "tips" in text:
            response = "You can ask me to identify a plant and I will show care info"
        elif "hello" in text or "hi" in text:
            response = "Hello! How's your garden today?"
        elif "thanks" in text or "thank you" in text:
            response = "You're welcome! Happy gardening!"
        elif "quiz" in text:
            start_quiz()
    return response


quiz_questions = [
    {"question": "Which plant is known as the 'King of Fruits'?", "options": ["Banana", "Mango", "Guava", "Apple"],
     "answer": 2},
    {"question": "Which crop is a staple in India?", "options": ["Rice", "Wheat", "Oats", "Barley"], "answer": 1},
    {"question": "Which fruit floats in water?", "options": ["Apple", "Orange", "Lemon", "Banana"], "answer": 3},
    {"question": "Which nut is actually a seed, not a true nut?",
     "options": ["Cashew", "Pistachio", "Walnut", "Almond"], "answer": 4},
    {"question": "Which spice is known as the 'Queen of Spices'?",
     "options": ["Cinnamon", "Nutmeg", "Clove", "Cardamom"], "answer": 4},
    {"question": "Which crop is naturally gluten-free?", "options": ["Wheat", "Jowar", "Barley", "Rye"], "answer": 2},
    {"question": "Which plant's fibers are called 'Golden Fiber'?", "options": ["Cotton", "Hemp", "Jute", "Flax"],
     "answer": 3},
    {"question": "Which plant can live for over 1,000 years?",
     "options": ["Baobab", "Olive tree", "Coconut", "Sequoia"], "answer": 2},
    {"question": "Which plant tracks the sun's movement during the day?",
     "options": ["Rose", "Sunflower", "Daisy", "Tulip"], "answer": 2},
    {"question": "Which enzyme in papaya helps tenderize meat?", "options": ["Protease", "Papain", "Lipase", "Amylase"],
     "answer": 2},
    {"question": "Which legume is commonly sprouted for salads and rich in protein?",
     "options": ["Chickpeas", "Mung beans", "Soybeans", "Lentils"], "answer": 2},
    {"question": "Which crop was first grown in space?", "options": ["Tomato", "Potato", "Wheat", "Corn"], "answer": 2},
    {"question": "Which fruit is botanically a berry but often mistaken as non-berry?",
     "options": ["Strawberry", "Raspberry", "Banana", "Blueberry"], "answer": 3},
    {"question": "Which crop can grow up to 6 meters tall?", "options": ["Corn", "Sugarcane", "Bamboo", "Maize"],
     "answer": 2},
    {"question": "Which disease affects wheat and is controlled by resistant varieties?",
     "options": ["Wheat Aphid", "Wheat Stem Fly", "Wheat Yellow Rust", "Wheat Scab"], "answer": 3},
    {"question": "Which fruit’s water can be used as a natural IV in emergencies?",
     "options": ["Papaya", "Orange", "Coconut", "Watermelon"], "answer": 3},
    {"question": "Which crop is highly drought-resistant and used as fodder?",
     "options": ["Rice", "Pearl millet", "Barley", "Wheat"], "answer": 2},
    {"question": "Which spice has been traditionally used as a dental remedy?",
     "options": ["Cardamom", "Nutmeg", "Clove", "Cinnamon"], "answer": 3},
    {"question": "Which vegetable is 95% water?", "options": ["Spinach", "Tomato", "Cucumber", "Lettuce"], "answer": 3},
    {"question": "Which plant is affected by the pink bollworm?", "options": ["Tomato", "Rice", "Cotton", "Wheat"],
     "answer": 3},
]


def shuffle_quiz_options(questions):
    for q in questions:
        correct_option = q['options'][q['answer'] - 1]
        random.shuffle(q['options'])
        q['answer'] = q['options'].index(correct_option) + 1
    return questions


quiz_questions = shuffle_quiz_options(quiz_questions)

quiz_quotes = [
    "Great job! Keep growing your plant knowledge!",
    "Remember: Every mistake is a step to learning!",
    "Awesome effort! Nature loves curious minds!",
    "You're sprouting into a plant expert!",
    "Keep it up! Even trees start from tiny seeds!"
]


def start_quiz():
    type_print("Welcome to Plant Quiz! Answer by typing the number of your choice.\n")
    score = 0
    questions = random.sample(quiz_questions, 5)
    total = len(questions)

    for idx, q in enumerate(questions, 1):
        type_print(f"Q{idx}: {q['question']}")
        for i, opt in enumerate(q['options'], 1):
            type_print(f"{i}. {opt}")

        while True:
            ans = input("Your answer (1-4): ").strip()
            if ans.isdigit() and int(ans) in [1, 2, 3, 4]:
                ans = int(ans)
                break
            else:
                type_print("Please enter a valid number between 1-4!")

        if ans == q['answer']:
            type_print("Correct! +20 points")
            score += 20
        else:
            correct_ans = q['options'][q['answer'] - 1]
            type_print(f"Wrong! Correct answer: {correct_ans} +0 points")

        progress = int((idx / total) * 20)
        print("Progress: [" + "#" * progress + "-" * (20 - progress) + f"] {idx}/{total}")
        type_print("-" * 20)

    type_print(f"\nQuiz Completed! Your total marks: {score}/{total * 20}")
    type_print(f"{random.choice(quiz_quotes)}")


def chatbot_interface():
    greet()
    while True:
        type_print("\nChoose an option:")
        print("1 - Detect Plant Species")
        print("2 - Detect Plant Disease")
        print("3 - Detect Both")
        print("4 - Take a Plant Quiz")
        print("Type 'exit' to quit.\n")
        choice = input("Your choice: ").strip()
        if choice.lower() == 'exit':
            type_print("Goodbye! Happy gardening!")
            break
        if choice not in ['1', '2', '3', '4']:
            type_print("Invalid choice. Try again!")
            continue

        if choice in ['1', '3']:
            plant_dataset, plant_loader, plant_idx = prepare_plant_dataset()
            plant_model, plant_criterion, plant_optimizer = create_model(len(plant_idx))
            load_latest_model(MODEL_FILE_PREFIX_PLANT, plant_model)
        if choice in ['2', '3']:
            dis_train, dis_val, dis_loader, dis_val_loader, dis_idx = prepare_disease_dataset()
            dis_model, dis_criterion, dis_optimizer = create_model(len(dis_idx))
            load_latest_model(MODEL_FILE_PREFIX_DISEASE, dis_model)
        if choice == '4':
            start_quiz()
            continue

        type_print("\nYou can enter image paths (comma separated) or ask me questions in plain English.")
        type_print("Type 'back' to return to main menu, or 'history' to look at your previous queries.")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                type_print("Goodbye! Happy gardening!")
                return
            elif user_input.lower() == 'back':
                break
            elif user_input.lower() == 'history':
                for h in history[-5:]:
                    print(h)
                continue

            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            paths = [p.strip().replace("\\", "/").strip('"').strip("'") for p in user_input.split(',')]
            is_image = all(p.lower().endswith(image_extensions) for p in paths)

            if is_image:
                for path in paths:
                    if not os.path.exists(path):
                        type_print(f"File '{path}' not found")
                        continue
                    results = []
                    if choice in ['1', '3']:
                        label, conf = predict_image(plant_model, plant_idx, path)
                        tip = get_tip(label, 'plant')
                        results.append(f"Species: {label} ({colored_confidence(conf)})\nInfo:\n{tip}")
                        if label in fun_facts: type_print(fun_facts[label])
                    if choice in ['2', '3']:
                        label, conf = predict_image(dis_model, dis_idx, path)
                        tip = get_tip(label, 'disease')
                        results.append(f"Disease: {label} ({colored_confidence(conf)})\nAdvice: {tip}")
                    type_print("\n".join(results))
                    history.append(f"Image: {path} | " + " | ".join(results))
                    type_print("-" * 40)

            else:
                response = handle_question(user_input)
                type_print(f"Chatbot: {response}")


chatbot_interface()