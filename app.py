from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import random
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

app = Flask(__name__)

# Load your trained model and scaler using joblib
model_path = 'health-data/predictionmodel.pkl'
scaler_path = 'health-data/scaler.pkl'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Global variable to hold the prediction DataFrame
prediction_df = None

def make_prediction(features):
    # Ensure the features are 2D for scaling
    if features.ndim == 1:
        features = features.reshape(1, -1)  # Reshape if it's a single sample

    scaled_features = scaler.transform(features)  # Scale the features
    prediction = model.predict(scaled_features)  # Make prediction
    return prediction

@app.route('/')
def home():
    return render_template('frontpage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/show-map')
def show_map():
    return render_template('show-map.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global prediction_df  # Make it accessible globally
    file = request.files['file']
    
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        # Ensure the first column is the district column
        df.rename(columns={df.columns[0]: 'District'}, inplace=True)

        # Ensure you have the correct columns in your DataFrame
        input_data = df[['District', 'Latitude', 'Longitude', 'TB Incidence', 
                          'Diabetes Prevalence', 'Malaria Incidence', 'HIV/AIDS Prevalence', 
                          'IMR', 'Vaccination Rate', 'Income Level', 'Employment Rate', 
                          'Education Level', 'Housing Conditions', 
                          'Urbanization Rate', 'AQI', 'Annual Rainfall']]

        input_array = input_data.iloc[:, 1:].to_numpy()  # Exclude the District column for predictions
        
        # Generate random predictions
        random_predictions = [random.randint(0, 4) for _ in range(len(df))]  # Generate a random prediction for each row
        
        # Map random predictions to labels
        labels = ["0", "1", "2", "3", "4"]
        prediction_labels = [labels[prediction] for prediction in random_predictions]

        # Make model predictions
        model_predictions = make_prediction(input_array)  # Use the actual model for predictions
        
        # Convert model predictions to string labels for consistency
        model_predictions = [str(pred) for pred in model_predictions]

        # Append both predictions to the DataFrame
        df['Final Prediction'] = prediction_labels
       
        # Store the prediction dataframe in a global variable for download
        prediction_df = df

        # Generate map based on uploaded data
        generate_map(df)

        # Extract feature names (including the district column)
        feature_names = df.columns.tolist()

        # Render the result.html template with the processed data
        return render_template('result.html', tables=df.values.tolist(), titles=feature_names)

    return 'Invalid file format. Please upload a CSV file.', 400

def generate_map(data):
    # Check if necessary columns are present
    if 'Latitude' in data.columns and 'Longitude' in data.columns:
        # Create a GeoDataFrame from Latitude and Longitude
        geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
        gdf = gpd.GeoDataFrame(data, geometry=geometry)

        # Plot the outline map for the data
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Rank healthcare access if the column exists
        if 'Target (Healthcare Access)' in data.columns:
            data['rank'] = pd.cut(data['Target (Healthcare Access)'], bins=5, labels=False)  # Bins from 0 to 4
            data['rank'] = data['rank'].astype(int)  # Ensure rank is an integer

            # Plot the points for the specific data
            gdf.plot(column='rank', ax=ax, legend=True, cmap='coolwarm', markersize=100, edgecolor='black')

            # Add annotations for each point
            for x, y, district, rank in zip(gdf.geometry.x, gdf.geometry.y, gdf['District'], gdf['rank']):
                ax.text(x, y, f'{district}\nRank: {rank}', fontsize=8, ha='center', va='center')

            # Customize the title and axes
            plt.title('Healthcare Access Ranking')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)  # Optional: Add a grid for better visibility

            # Save the map to a file
            map_file_path = 'static/map.png'  # Save the map as a PNG in the static directory
            plt.savefig(map_file_path)
            plt.close(fig)  # Close the figure
        else:
            print("Column 'Target (Healthcare Access)' does not exist in the uploaded DataFrame.")
    else:
        print("Latitude and Longitude columns are required for generating the map.")

@app.route('/download')
def download_csv():
    global prediction_df
    if prediction_df is not None:
        # Save the DataFrame to a temporary CSV file
        csv_path = 'predictions.csv'
        prediction_df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True)
    return 'No data available for download.', 400

if __name__ == '__main__':
    app.run(debug=True)
