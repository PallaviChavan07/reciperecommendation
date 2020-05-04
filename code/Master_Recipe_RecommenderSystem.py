import pandas as pd
import code.Normal_Predictor as np
import code.Collaborative_item_item as cii
import code.Collaborative_user_user as cuu
import code.ContentBased as cb
import code.SVD_MatrixFactorization as svd_mf
import code.SVDplusplus as svdpp
import code.SlopeOne as slopeone
import code.CoClustering as coclust
import code.Hybrid as hyd
import os

recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
train_rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
user_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))

benchmark = []
#cb.ComputeContentBasedFiltering(recipe_df, train_rating_df, pd)
#np.Normalpredictor(recipe_df, train_rating_df, pd, benchmark)
cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
#cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
#cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
svd_mf.ComputeSVD_MatrixFactorization(recipe_df, train_rating_df, pd, benchmark)
svdpp.SVDplusplus(recipe_df, train_rating_df, pd, benchmark)
slopeone.Slopeone(recipe_df, train_rating_df, pd, benchmark)
coclust.Coclustering(recipe_df, train_rating_df, pd, benchmark)

#hybrid
hyd.ComputeHybrid(recipe_df, train_rating_df, pd, benchmark)

pd.set_option("display.max_rows", None, "display.max_columns", None)
results = pd.DataFrame.from_records(benchmark, exclude=['MSE', 'FCP'],
                                    columns=['RMSE', 'MAE', 'MSE', 'FCP', 'PrecisionAt10', 'RecallAt10'],
                                    index=['KNNBasic_Item_Item', 'KNNWithMeans_Item_Item', 'SVD', 'SVD++', 'SlopeOne', 'CoClustering', 'Hybrid'])
print(results)
