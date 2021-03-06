import pandas as pd
import Normal_Predictor as np
import Collaborative_item_item as cii
import Collaborative_user_user as cuu
import ContentBased as cb
import SVD_MatrixFactorization as svd_mf
import SVDplusplus as svdpp
import SlopeOne as slopeone
import CoClustering as coclust
import Hybrid as hyd
import os
import time
start_time = time.time()

recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
train_rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
user_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))

benchmark = []
#cb.ComputeContentBasedFiltering(recipe_df, train_rating_df, pd)
#np.Normalpredictor(recipe_df, train_rating_df, pd, benchmark)
#cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
#cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
#cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
#cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
svd_mf.ComputeSVD_MatrixFactorization(recipe_df, train_rating_df, pd, benchmark)
print("--- Total Surprise SVD recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

svdpp.SVDplusplus(recipe_df, train_rating_df, pd, benchmark)
print("--- Total Surprise SVD++ recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

slopeone.Slopeone(recipe_df, train_rating_df, pd, benchmark)
print("--- Total Surprise Slopeone recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

coclust.Coclustering(recipe_df, train_rating_df, pd, benchmark)
print("--- Total Surprise Coclustering recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

#hybrid
#hyd.ComputeHybrid(recipe_df, train_rating_df, pd, benchmark)
#print("--- Total Surprise Hybrid recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

pd.set_option("display.max_rows", None, "display.max_columns", None)
results = pd.DataFrame.from_records(benchmark, exclude=['MSE', 'FCP'],
                                    columns=['RMSE', 'MAE', 'MSE', 'FCP', 'PrecisionAt5', 'RecallAt5', 'PrecisionAt10', 'RecallAt10', 'PrecisionAt20', 'RecallAt20'],
                                    index=['SVD++', 'SVD', 'CoClustering', 'SlopeOne'])
print(results)