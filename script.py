from library.data_analysis import *
from library.customer_segmentation import *
from library.product_recommendation import *

# parameters
data_path = 'Data/'
input_folder = 'raw_data/'
transaction_file_name = 'Transaction.csv'
product_file_name = 'Product.csv'
output_folder = 'result/'
log_folder='logs/'


### -------------------------- Data Analysis  -------------------------- ###
print('Analyzing Data ... ')
da = DataAnalysis(data_path, transaction_file_name, product_file_name, input_folder, output_folder, log_folder)
da.run()


### -------------------------- Customer Segmentation  -------------------------- ###
print('Performing Customer Segmentation based on Purchase Behavior of customers ... ')
cs_buy_pattern = Based_on_Purchase_Behavior(data_path, transaction_file_name, input_folder, output_folder, log_folder)
cs_buy_pattern.run()

print('Performing Customer Segmentation using K-means Clustering ... ')
cs_kmeans_clustering = Based_on_Kmeans_Clustering(data_path, transaction_file_name, input_folder, output_folder, log_folder)
cs_kmeans_clustering.run()



### -------------------------- Product Recommendation  -------------------------- ###
prod_reco_cf = CollaborativeFiltering(data_path, transaction_file_name, input_folder, output_folder, log_folder)
prod_reco_cf.run()

prod_reco_apriori = MarketBasketAnalysis(data_path, transaction_file_name, product_file_name, input_folder, output_folder, log_folder)
prod_reco_apriori.run()
        
        