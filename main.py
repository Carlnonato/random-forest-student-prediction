import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print("================================")
print(" STUDENT PASS OR FAIL PREDICTION ")
print(" RANDOM FORREST SHIT")
print("================================")

data = pd.read_csv("datasets.csv")

X = data[["study_hours" , "attendance", "previous_score"]]
y = data ["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Training completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model accuracy:", accuracy * 100, "%")


print("\n====================================")
print(" NEW STUDENT PREDICTION")
print("====================================")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))


new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score]],
    columns=["study_hours", "attendance", "previous_score"]
)

result = model.predict(new_student)

print("\n====================================")
print(" PREDICTION RESULT")
print("====================================")

print("Student Result:", result[0])