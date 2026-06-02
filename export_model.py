import joblib
from RandomForestModel import build_dataset_from_csv, train_random_forest
from config import TRAINING_DATA_ROOT, VLA_BRUTAL_ROOT

def main():
    #Re-build the dataset one last time to make sure it's fresh
    print("Loading data for final export...")
    X, y = build_dataset_from_csv()

    #Train the model on ALL available data 
    #(We use 100% of the data here so the model is as smart as possible for the leaderboard)
    print("Training final model...")
    clf = train_random_forest(X, y)

    #Save the model to a file
    joblib.dump(clf, 'model.pkl')
    
    print("\nSuccess! 'model.pkl' has been created.")
    print("This file contains your trained Random Forest weights.")

if __name__ == "__main__":
    main()