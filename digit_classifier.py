import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(knn, "knn_model.pkl", compress=3)
print("Compressed model saved as knn_model.pkl")

def preprocess_image(image):
    from PIL import ImageOps
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    image = image.convert("L")
    image = ImageOps.invert(image)
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = img_array / 255.0
    return img_array
