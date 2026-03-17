# =========================
# Tkinter & GUI
# =========================
import tkinter as tk
import tkinter
from tkinter import *
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# =========================
# Core Python
# =========================
import os
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

# =========================
# Numerical & Data
# =========================
import numpy as np
import pandas as pd

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# =========================
# TensorFlow / Keras
# =========================
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    Conv2D, MaxPooling2D, Flatten,
    GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping

# =========================
# Scikit-learn
# =========================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

# =========================
# Model Persistence
# =========================
import joblib

# =========================
# Database
# =========================
import pymysql


# Load InceptionResNetV2 base model
base_model = InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',            
    input_shape=(128, 128, 3) 
)


base_model.trainable = False 


main = Tk('Ensemble Feature Learning for Accurate Recognition of Industrial Tools in Workshop Environments')
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window size to full screen
main.geometry(f"{screen_width}x{screen_height}")

global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"

import os
from tkinter import filedialog, END

def uploadDataset():
    global filename, categories
    text.delete('1.0', END)

    # select dataset folder
    filename = filedialog.askdirectory(initialdir=".")

    # class folders inside the dataset
    categories = [
        d for d in os.listdir(filename)
        if os.path.isdir(os.path.join(filename, d))
    ]
    categories.sort()

    text.insert(END, "Dataset loaded successfully\n")
    text.insert(END, f"Total classes: {len(categories)}\n")
    text.insert(END, f"Classes: {categories}\n")


    
def imageProcessing():
    global X, Y, categories, filename, base_model, model_folder

    text.delete('1.0', END)

    path = filename
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)

    X_file = os.path.join(model_folder, "X_features.npz")
    Y_file = os.path.join(model_folder, "Y_labels.npy")
    classes_file = os.path.join(model_folder, "classes.json")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)["X"]
        Y = np.load(Y_file)
        with open(classes_file, "r") as f:
            categories = json.load(f)

        text.insert(END, "Loaded features from model folder\n")
        return

    X, Y, categories = [], [], []
    print("Processing dataset from:", path)

    for class_name in sorted(os.listdir(path)):
        class_dir = os.path.join(path, class_name)
        if not os.path.isdir(class_dir):
            continue

        categories.append(class_name)
        label = len(categories) - 1

        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_file)

                img = image.load_img(img_path, target_size=(128, 128))
                arr = image.img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)

                feat = base_model.predict(arr, verbose=0)
                X.append(np.squeeze(feat))
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    np.savez_compressed(X_file, X=X)
    np.save(Y_file, Y)

    with open(classes_file, "w") as f:
        json.dump(categories, f)

    text.insert(END, "Features extracted & saved in model folder\n")
    text.insert(END, f"Images: {len(X)} | Classes: {len(categories)}\n")


def Train_Test_split():
    global X, Y, x_train, x_test, y_train, y_test

    print("Splitting ->", X.shape, Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    text.insert(END, f"Train samples: {x_train.shape}\n")
    text.insert(END, f"Test samples: {x_test.shape}\n")



def calculateMetrics(algorithm, categories, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    text.insert(END, algorithm + " Accuracy  :  " + str(a) + "\n")
    text.insert(END, algorithm + " Precision : " + str(p) + "\n")
    text.insert(END, algorithm + " Recall    : " + str(r) + "\n")
    text.insert(END, algorithm + " FScore    : " + str(f) + "\n")

    conf_matrix = confusion_matrix(y_test, predict)
    se = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) * 100 if conf_matrix.shape[0] > 1 else 0
    sp = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) * 100 if conf_matrix.shape[0] > 1 else 0

    text.insert(END, algorithm + ' Sensitivity : ' + str(se) + "\n")
    text.insert(END, algorithm + ' Specificity : ' + str(sp) + "\n\n")

    CR = classification_report(y_test, predict, target_names=categories)
    text.insert(END, algorithm + ' Classification Report \n')
    text.insert(END, algorithm + str(CR) + "\n\n")

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories,
                     annot=True, cmap="magma", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(algorithm + " Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

    # =======================
    # ROC Curve & AUC Score
    # =======================

    if len(categories) > 2:
        y_test_bin = label_binarize(y_test, classes=list(range(len(categories))))
        pred_bin = label_binarize(predict, classes=list(range(len(categories))))

        auc_score = roc_auc_score(y_test_bin, pred_bin, average="macro", multi_class="ovr")
        text.insert(END, algorithm + ' AUC Score (Macro Avg) : ' + str(auc_score * 100) + "\n\n")

        plt.figure(figsize=(8, 6))
        for i in range(len(categories)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], pred_bin[:, i])
            plt.plot(fpr, tpr, label=f"{categories[i]} (AUC = {roc_auc_score(y_test_bin[:, i], pred_bin[:, i]):.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(algorithm + " ROC Curve (Multi-class)")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        try:
            auc_score = roc_auc_score(y_test, predict)
            text.insert(END, algorithm + ' AUC Score : ' + str(auc_score * 100) + "\n\n")

            fpr, tpr, _ = roc_curve(y_test, predict)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % auc_score)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(algorithm + " ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        except ValueError:
            text.insert(END, algorithm + " AUC not available for this binary prediction.\n\n")


def Existing_DecisionTree():
    global x_train, x_test, y_train, y_test, model_folder, categories

    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "DecisionTree_model.pkl")

    # Flatten CNN/DNN feature inputs
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat  = x_test.reshape((x_test.shape[0], -1))

    # ----- Load model if already trained -----
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        text.insert(END, "Loaded saved Decision Tree model\n")
    else:
        # ----- Train Decision Tree -----
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )

        model.fit(x_train_flat, y_train)
        joblib.dump(model, model_file)
        text.insert(END, "Decision Tree model trained & saved\n")

    # ----- Predict & Evaluate -----
    y_pred = model.predict(x_test_flat)

    calculateMetrics("Decision Tree Classifier", categories, y_pred, y_test)



def Existing_Perceptron():
    global x_train, x_test, y_train, y_test, model_folder, categories

    text.delete('1.0', END)

    model_file = os.path.join(model_folder, "Perceptron_model.pkl")

    # Flatten CNN features
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat  = x_test.reshape((x_test.shape[0], -1))

    # Train or load model
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print("Loaded saved Perceptron model")
    else:
        model = Perceptron(
        max_iter=20,    
        eta0=0.001,      
        tol=1e-2,        
        shuffle=False
    )

        model.fit(x_train_flat, y_train)
        joblib.dump(model, model_file, compress=5)
        print("Perceptron model saved")

    # Predict
    y_pred = model.predict(x_test_flat)

    # Evaluate
    calculateMetrics("Perceptron Classifier", categories, y_pred, y_test)


def Hybrid_DNN_RF_Model():
    global x_train, x_test, y_train, y_test, model_folder, categories

    text.delete('1.0', END)

    dnn_dir = os.path.join(model_folder, "DNN_prob_model_tf")
    rf_file = os.path.join(model_folder, "RF_on_DNN_Probs.pkl")

    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat  = x_test.reshape((x_test.shape[0], -1))

    num_classes = len(np.unique(y_train))

    # -------- LOAD / TRAIN DNN --------
    if os.path.exists(dnn_dir):
        dnn = load_model(dnn_dir, compile=False)
        text.insert(END, "Loaded DNN (SavedModel)\n")
    else:
        inputs = Input(shape=(x_train_flat.shape[1],))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.6)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        dnn = Model(inputs, outputs)
        dnn.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

        dnn.fit(x_train_flat, y_train,
                epochs=5, batch_size=64,
                validation_split=0.1, verbose=1)

        dnn.save(dnn_dir)
        text.insert(END, "DNN trained & saved\n")

    # -------- DNN → probability features --------
    train_probs = dnn.predict(x_train_flat, verbose=0)
    test_probs  = dnn.predict(x_test_flat,  verbose=0)

    # -------- LOAD / TRAIN RF --------
    if os.path.exists(rf_file):
        rf = joblib.load(rf_file)
        text.insert(END, "Loaded RF Hybrid\n")
    else:
        rf = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        rf.fit(train_probs, y_train)
        joblib.dump(rf, rf_file)
        text.insert(END, "RF trained & saved\n")

    # -------- Evaluate --------
    y_pred = rf.predict(test_probs)
    calculateMetrics("Hybrid (DNN Probs → RF)", categories, y_pred, y_test)



def predict_Hybrid_DNN_RF():
    global model_folder, categories, base_model

    text.delete('1.0', END)

    dnn_dir = os.path.join(model_folder, "DNN_prob_model_tf")
    rf_file = os.path.join(model_folder, "RF_on_DNN_Probs.pkl")

    if not os.path.exists(dnn_dir):
        text.insert(END, "DNN model not found. Train Hybrid model first.\n")
        return
    if not os.path.exists(rf_file):
        text.insert(END, "RF model not found. Train Hybrid model first.\n")
        return

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    dnn = load_model(dnn_dir, compile=False)
    rf  = joblib.load(rf_file)

    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        text.insert(END, "No image selected.\n")
        return

    # ---- Preprocess image ----
    img = image.load_img(filename, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ---- CNN features ----
    cnn_feat = base_model.predict(arr, verbose=0).reshape(1, -1)

    # ---- DNN probs → RF ----
    probs = dnn.predict(cnn_feat, verbose=0)
    y_pred = rf.predict(probs)[0]
    class_name = categories[y_pred]

    # ---- Show result in text box ----
    text.insert(END, f"Hybrid Prediction: {class_name}\n")

    # ---- Display popup image with label ----
    img_disp = cv2.imread(filename)
    img_disp = cv2.resize(img_disp, (500, 500))

    label = f"Predicted: {class_name}"
    cv2.putText(
        img_disp, label,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255), 2
    )

    cv2.imshow("Hybrid Prediction", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def graph():
    global accuracy, precision, recall, fscore

    # Create a DataFrame with all metrics
    df = pd.DataFrame([
        ['Accuracy', 'Existing', accuracy[0]],
        ['Precision', 'Existing', precision[0]],
        ['Recall', 'Existing', recall[0]],
        ['F1 Score', 'Existing', fscore[0]],
        ['Accuracy', 'Proposed', accuracy[1]],
        ['Precision', 'Proposed', precision[1]],
        ['Recall', 'Proposed', recall[1]],
        ['F1 Score', 'Proposed', fscore[1]]
    ], columns=['Metric', 'Model', 'Score'])

    # Pivot for bar plot
    pivot_df = df.pivot(index='Metric', columns='Model', values='Score')
    pivot_df.plot(kind='bar', figsize=(8, 6), colormap='Set2')

    # Graph properties
    plt.title('Comparison of Classifier Performance (KNN vs CNN+ETC)')
    plt.ylabel('Score (%)')
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

def close():
    main.destroy()
    

def setBackground():
    global bg_photo
    image_path = r"BG_image\images.jpeg" 
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()

def connect_db():
    return pymysql.connect(host='localhost', user='root', password='root', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


# Clear buttons function
def clear_buttons():
    for widget in main.place_slaves():
        if isinstance(widget, tkinter.Button):
            widget.destroy()

# Admin Button Functions
def show_admin_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()

    tk.Button(main, text="Upload Dataset",
              command=uploadDataset,
              font=font1).place(x=80, y=150)

    tk.Button(main, text="FE with InceptionResNetV2",
              command=imageProcessing,
              font=font1).place(x=350, y=150)

    tk.Button(main, text="Dataset Splitting",
              command=Train_Test_split,
              font=font1).place(x=700, y=150)

    tk.Button(main, text="Train Decision Tree Classifier",
              command=Existing_DecisionTree,
              font=font1).place(x=80, y=230)

    tk.Button(main, text="Train Perceptron Classifier",
              command=Existing_Perceptron,
              font=font1).place(x=350, y=230)

    tk.Button(main, text="Train Hybrid Classifier",
              command=Hybrid_DNN_RF_Model,
              font=font1).place(x=700, y=230)

    #  New Logout button
    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=600)

# User Button Functions
def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    tk.Button(main, text="Prediction from Test Image",
              command=predict_Hybrid_DNN_RF,
              font=font1).place(x=550, y=200)
    tk.Button(main, text="Exit", command=close, font=font1).place(x=980, y=200)

    # New Logout button
    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=600)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightgreen').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightgreen').place(x=1000, y=100)



def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(
    main,
    text="Ensemble Feature Learning for Accurate Recognition of Industrial Tools in Workshop Environments",
    bg='#003366',
    fg='white',
    font=font,
    height=3,
    width=120
)
title.pack(pady=10)
                     
font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=300)
text.config(font=font1) 


# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=100, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=400, y=100)


admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightgreen')
admin_button.place(x=700, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightgreen')
user_button.place(x=1000, y=100)

main.mainloop()
