"""
Main entry point for House Price Prediction Project.
Uses modular structure with separate files for different functionalities.
"""

import sys
import os

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CaliforniaHousePricePredictor


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("ElasticNet Regression Model")
    print("="*70 + "\n")
    
    try:
        # Initialize predictor
        predictor = CaliforniaHousePricePredictor(alpha=0.1, l1_ratio=0.5)
        
        # Display dataset information
        predictor.explore_data()
        
        # Preprocess data
        predictor.split_and_preprocess()
        
        # Train model
        predictor.train_model()
        
        # Evaluate model
        predictor.evaluate_model()
        
        # Visualize results
        predictor.visualize_results()
        
        # Interactive user input prediction
        print("\n" + "="*70)
        print("Would you like to enter your own house values? (y/n): ")
        
        try:
            choice = input().strip().lower()
            
            if choice == 'y':
                print("\n" + "="*70)
                print("ENTER YOUR HOUSE VALUES")
                print("="*70)
                
                # Get user input for each feature
                medinc = float(input("Median Income (in $10k, e.g., 8.5 for $85k): "))
                houseage = float(input("House Age (in years): "))
                avelrooms = float(input("Average Rooms per household: "))
                avebedrms = float(input("Average Bedrooms per household: "))
                population = float(input("Block Population: "))
                aveoccup = float(input("Average Occupancy per household: "))
                latitude = float(input("Latitude (e.g., 37.8 for SF): "))
                longitude = float(input("Longitude (e.g., -122.4 for SF): "))
                
                # Predict with user values
                predictor.predict_sample_house(
                    medinc=medinc, houseage=houseage, avelrooms=avelrooms,
                    avebedrms=avebedrms, population=population, aveoccup=aveoccup,
                    latitude=latitude, longitude=longitude
                )
            else:
                print("Using default sample values above.")
                
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default sample values above.")
        except ValueError:
            print("\nInvalid input! Please enter valid numbers. Using default sample values above.")
        except Exception as e:
            print(f"Input error: {str(e)}. Using default sample values above.")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your environment and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
