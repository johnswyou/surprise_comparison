import argparse
import pandas as pd
from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise.model_selection import cross_validate
import json

# Dictionary mapping algorithm names to their corresponding surprise classes
algorithm_map = {
    'normal_predictor': NormalPredictor,
    'baseline_only': BaselineOnly,
    'knn_basic': KNNBasic,
    'knn_with_means': KNNWithMeans,
    'knn_with_zscore': KNNWithZScore,
    'knn_baseline': KNNBaseline,
    'svd': SVD,
    'svdpp': SVDpp,
    'nmf': NMF,
    'slope_one': SlopeOne,
    'co_clustering': CoClustering,
}

def run_algorithm(algo_name, data, params):
    reader = Reader(rating_scale=(data['score'].min(), data['score'].max()))
    data = Dataset.load_from_df(data[['user_id', 'item_id', 'score']], reader)

    algo_class = algorithm_map.get(algo_name.lower())
    if not algo_class:
        print(f"Unsupported algorithm: {algo_name}")
        return

    algo = algo_class(**params)

    # Perform cross-validation
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print(f"Results for {algo_name}:", results)

def main():
    parser = argparse.ArgumentParser(description="Compare Recommendation Algorithms from Surprise Library")
    parser.add_argument('dataset', type=str, help='Path to the dataset CSV file')
    parser.add_argument('algorithm', type=str, help='Algorithm to use (e.g., svd, knn_basic)')
    parser.add_argument('--params', type=str, default='{}', help='Algorithm hyperparameters in JSON format')

    args = parser.parse_args()

    # Use the following if you want to pass in the JSON as a string to the --params argument in the command line
    # try:
    #     params = json.loads(args.params)
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding params: {e}")
    #     return
    
    # Use the following if you want to pass in the path to a JSON file to the --params argument in the command line
    if args.params:
        try:
            with open(args.params, 'r') as f:
                params = json.load(f)
        except Exception as e:
            print(f"Error reading params file: {e}")
            return
    else:
        params = {}

    # Load dataset
    data = pd.read_csv(args.dataset)

    # Run specified algorithm
    run_algorithm(args.algorithm, data, params)

if __name__ == '__main__':
    
    main()

# To run, in command line/terminal:
# python main.py path/to/dataset.csv svd --params path/to/params.json

# Make sure dataset.csv has columns user_id, item_id, and score