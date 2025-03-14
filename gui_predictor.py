import tkinter as tk
from tkinter import ttk
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

class TennisPredictor:
    def __init__(self):
        # Load models using FIXED filenames (no timestamp)
        try:
            self.ml_model = CatBoostClassifier().load_model('moneyline_model.cbm')
            self.spread_model = CatBoostRegressor().load_model('spread_model.cbm')
            self.total_model = CatBoostRegressor().load_model('total_model.cbm')
        except Exception as e:
            raise RuntimeError(f"Model loading failed - ensure models are trained first. Error: {str(e)}")
        
        # Get feature names from the model
        self.model_features = self.ml_model.feature_names_
        
        # Descriptions and categorical options (updated to match model requirements)
        self.descriptions = {
            "Tournament": "Tournament Name (e.g., Wimbledon)",
            "Series": "Tournament Level (Grand Slam/Masters)",
            "Court": "Court Type (Indoor/Outdoor)",
            "Surface": "Playing Surface (Hard/Clay/Grass)",
            "Round": "Match Stage (e.g., 3rd Round/Final)",
            "Best of": "Match Format (3 or 5 sets)",
            "Rank_1": "Player 1 World Ranking (1 = best)",
            "Rank_2": "Player 2 World Ranking",
            "Pts_1": "Player 1 ATP Points",
            "Pts_2": "Player 2 ATP Points",
            "Odd_1": "Player 1 Betting Odds",
            "Odd_2": "Player 2 Betting Odds"
        }
        
        self.categorical_options = {
            "Tournament": ["Australian Open", "French Open", "Wimbledon", "US Open", "Masters 1000"],
            "Series": ["Grand Slam", "ATP Masters 1000", "ATP 500", "ATP 250"],
            "Court": ["Indoor", "Outdoor"],
            "Surface": ["Hard", "Clay", "Grass", "Carpet"],
            "Round": ["1st Round", "2nd Round", "3rd Round", "Quarterfinals", "Semifinals", "Final"],
            "Best of": ["3", "5"]
        }

        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Tennis Match Predictor")
        self.root.geometry("600x800")
        self.root.configure(bg="#f0f0f0")
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        self.create_widgets()

    def create_widgets(self):
        row = 0
        self.inputs = {}
        title = ttk.Label(self.main_frame, text="Enter Match Details", font=('Arial', 14, 'bold'))
        title.grid(row=row, column=0, columnspan=2, pady=(0, 20))
        row += 1

        # Create input fields for all features
        for feature in self.model_features:
            if feature not in self.descriptions:
                continue
                
            label = ttk.Label(self.main_frame, text=self.descriptions[feature], wraplength=250, justify='right')
            label.grid(row=row, column=0, padx=10, pady=5, sticky='e')
            
            if feature in self.categorical_options:
                var = tk.StringVar(value=self.categorical_options[feature][0])
                dropdown = ttk.Combobox(self.main_frame, textvariable=var, values=self.categorical_options[feature], width=25)
                dropdown.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.inputs[feature] = var
            else:
                var = tk.DoubleVar()
                entry = ttk.Entry(self.main_frame, textvariable=var, validate="key", validatecommand=(self.root.register(self.validate_numeric), "%P"), width=28)
                entry.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.inputs[feature] = var
            row += 1

        self.result_label = ttk.Label(self.main_frame, text="", font=('Arial', 12, 'bold'), justify='center')
        self.result_label.grid(row=row, column=0, columnspan=2, pady=20)
        
        predict_btn = ttk.Button(self.main_frame, text="Predict Match Odds", command=self.predict)
        predict_btn.grid(row=row+1, column=0, columnspan=2, pady=10, ipadx=20, ipady=5)

    def predict(self):
        try:
            input_data = {feat: var.get() for feat, var in self.inputs.items()}
            input_data["Best of"] = int(input_data["Best of"])
            
            # Calculate derived features
            input_data["Rank_Diff"] = input_data["Rank_1"] - input_data["Rank_2"]
            input_data["Pts_Diff"] = input_data["Pts_1"] - input_data["Pts_2"]
            input_data["Odds_Ratio"] = input_data["Odd_1"] / input_data["Odd_2"]
            
            # Remove any features not in the model's expected features
            df = pd.DataFrame([input_data])[self.model_features]
            
            # Get predictions from all models
            ml_prob = self.ml_model.predict_proba(df)[0][1]
            spread = self.spread_model.predict(df)[0]
            total = self.total_model.predict(df)[0]
            
            # Calculate fair odds
            ml_odds_p1 = round(1/ml_prob, 2) if ml_prob > 0 else "N/A"
            ml_odds_p2 = round(1/(1-ml_prob), 2) if ml_prob < 1 else "N/A"
            
            # Determine spread favor
            if spread > 0:
                spread_text = f"Player 1: -{abs(spread):.1f}"
            else:
                spread_text = f"Player 1: +{abs(spread):.1f}"
            
            result_text = (
                f"Moneyline Odds:\n"
                f"Player 1: {ml_odds_p1}\nPlayer 2: {ml_odds_p2}\n\n"
                f"Spread Prediction: {spread_text}\n\n"
                f"Total Prediction: {round(total,1)} sets"
            )
            
            self.result_label.config(text=result_text)
            
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")

    def validate_numeric(self, value):
        return value == "" or self.is_float(value)

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    predictor = TennisPredictor()
    predictor.run()