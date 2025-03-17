import os
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Based_on_Purchase_Behavior:
    def __init__(self, path, transaction_file, in_folder, out_folder, log_folder):
        self.path = path
        self.transaction_file = transaction_file
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.log_folder = log_folder

        # Ensure log folder exists
        log_path = os.path.join(self.path, self.log_folder)
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, "Customer_Segmentation-Purchase_Behavior.txt")

    def log(self, message):
        with open(self.log_file, "a") as file:
            file.write(message + "\n")

    def load_data(self, filename):
        df = pd.read_csv(self.path + self.in_folder + filename)
        return df

    def write_data(self, df, filename):
        output_path = os.path.join(self.path, self.out_folder)
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        df.to_csv(full_path, index=False)

    def run(self):
         # Clear log file at start
        open(self.log_file, 'w').close()

        ## --------------------- Read Raw tables --------------------------------- ##
        df = self.load_data(self.transaction_file)
        
        # Convert columns
        df['Purchase Amount'] = pd.to_numeric(df['Purchase Amount'], errors='coerce')
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        
        
        ## --------------------- Segmentation based on Purchase Behavior --------------------------------- ##
        self.log('--> SEGMENTATION BASED ON PURCHASE BEHAVIOR\n')
        self.log('Performing below segmentation:')
        self.log('1. Segmentation based on Total Spending: Low, Medium and High Spender')
        self.log('2. Segmentation based on Average Order Value: Budget Buyer, Value Shopper, Big Buyer')
        self.log('3. Segmentation based on Frequency of Transaction: One-time, Occasional, Frequent Buyer')
        self.log('4. Segmentation based on Recency: Active, At-risk, Lapsed Buyer')
        self.log('\n')
        
        # ---------------------- 1. Total Spend Segmentation ----------------------
        total_spend = df.groupby('Customer ID')['Purchase Amount'].sum().reset_index()
        total_spend.columns = ['Customer ID', 'Total Spend']
        total_spend_q = total_spend['Total Spend'].quantile([0.2, 0.8]).to_dict()

        def spend_category(spend):
            if spend <= total_spend_q[0.2]:
                return 'Low Spender'
            elif spend <= total_spend_q[0.8]:
                return 'Medium Spender'
            else:
                return 'High Spender'

        total_spend['Spend Category'] = total_spend['Total Spend'].apply(spend_category)
        
        # ---------------------- 2. Average Order Value Segmentation ----------------------
        customer_txn = df.groupby(['Customer ID', 'Transaction ID'])['Purchase Amount'].sum().reset_index()
        avg_order = customer_txn.groupby('Customer ID')['Purchase Amount'].mean().reset_index()
        avg_order.columns = ['Customer ID', 'Average Order Value']
        avg_order['Average Order Value'] = avg_order['Average Order Value'].round(2)
        avg_order_q = avg_order['Average Order Value'].quantile([0.2, 0.8]).to_dict()

        def order_value_category(avg):
            if avg <= avg_order_q[0.2]:
                return 'Budget Buyer'
            elif avg <= avg_order_q[0.8]:
                return 'Value Shopper'
            else:
                return 'Big Buyer'

        avg_order['Order Value Category'] = avg_order['Average Order Value'].apply(order_value_category)
        
        # ---------------------- 3. Frequency of Purchases Segmentation ----------------------
        purchase_frequency = df.groupby('Customer ID')['Transaction ID'].nunique().reset_index()
        purchase_frequency.columns = ['Customer ID', 'Purchase Frequency']

        def frequency_category(freq):
            if freq == 1:
                return 'One-time Buyer'
            elif freq <= 3:
                return 'Occasional Buyer'
            else:
                return 'Frequent Buyer'

        purchase_frequency['Frequency Category'] = purchase_frequency['Purchase Frequency'].apply(frequency_category)
        
        # ---------------------- 4. Recency Segmentation ----------------------
        latest_date = df['Purchase Date'].max()
        recency_df = df.groupby('Customer ID')['Purchase Date'].max().reset_index()
        recency_df['Days Since Last Purchase'] = (latest_date - recency_df['Purchase Date']).dt.days

        def recency_category(days):
            if days <= 30:
                return 'Active Buyer'
            elif days <= 90:
                return 'At-risk Buyer'
            else:
                return 'Lapsed Buyer'

        recency_df['Recency Category'] = recency_df['Days Since Last Purchase'].apply(recency_category)
        
        # ---------------------- Merge All Segmentations ----------------------
        segmentation = total_spend.merge(avg_order, on='Customer ID')
        segmentation = segmentation.merge(purchase_frequency, on='Customer ID')
        segmentation = segmentation.merge(recency_df[['Customer ID', 'Recency Category']], on='Customer ID')
        
        self.log('Here is customer segmentation results based on purchase behavior for first 10 customers:')
        self.log(segmentation.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Customer_Segmentation_based_on_Purchase_Behavior.csv'
        self.write_data(segmentation, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')
        
        
        ## --------------------- Segmentation based on Purchase Pattern --------------------------------- ##
        self.log('--> SEGMENTATION BASED ON PURCHASE PATTERN\n')
        self.log('Performing below segmentation:')
        self.log('1. Segmentation based on Purchase Pattern: New, Loyal, Returning Customer')
        self.log('2. Segmentation based on Buyer Seasonality: Holiday Shopper, Regular Season Buyer, Off-season Shopper')
        self.log('\n')
        
        # Group by Customer ID and count number of unique transactions
        purchase_counts = df.groupby('Customer ID')['Transaction ID'].nunique().reset_index()
        purchase_counts.columns = ['Customer ID', 'Purchase Count']
        
        # Define purchase pattern category
        def purchase_pattern_category(count):
            if count == 1:
                return 'New Customer'
            elif count >= 5:
                return 'Loyal Customer'
            else:
                return 'Returning Customer'

        purchase_counts['Purchase Pattern'] = purchase_counts['Purchase Count'].apply(purchase_pattern_category)
        
        # ------------------- Seasonal Buyers Segmentation -------------------
        # Extract month from Purchase Date
        df['Purchase Month'] = df['Purchase Date'].dt.month

        # Count number of purchases in each month per customer
        monthly_purchase = df.groupby(['Customer ID', 'Purchase Month']).size().unstack(fill_value=0)
        
        # Define holiday month(s), here assumed December (12) for this dataset
        holiday_months = [12]
        
        # Determine buyer seasonality
        def seasonality_category(row):
            total_purchases = row.sum()
            holiday_purchases = row[holiday_months].sum()
            if holiday_purchases / total_purchases >= 0.6:
                return 'Holiday Shopper'
            elif row[row > 0].count() >= 6:
                return 'Regular Season Buyer'
            else:
                return 'Off-season Shopper'
            
        monthly_purchase['Seasonality Category'] = monthly_purchase.apply(seasonality_category, axis=1)
        
        # ------------------- Merge both segmentations -------------------
        pattern_segmentation = purchase_counts.merge(monthly_purchase[['Seasonality Category']], on='Customer ID')
        
        self.log('Here is customer segmentation results based on purchase pattern for first 10 customers:')
        self.log(pattern_segmentation.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Customer_Segmentation_based_on_Purchase_Pattern.csv'
        self.write_data(pattern_segmentation, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')
        
        
        ## --------------------- Segmentation based on Basket Composition --------------------------------- ##
        self.log('--> SEGMENTATION BASED ON BASKET COMPOSITION\n')
        self.log('Performing below segmentation:')
        self.log('1. Segmentation based on Basket Value Segmentation: Low, Moderate, High Basket Buyer')
        self.log('2. Segmentation based on Items per Transaction: Selective, Bulk, Mixed Buyer')
        self.log('\n')
        
        # Calculate basket value: total amount per transaction
        basket_value = df.groupby(['Customer ID', 'Transaction ID'])['Purchase Amount'].sum().reset_index()
        basket_avg = basket_value.groupby('Customer ID')['Purchase Amount'].mean().reset_index()
        basket_avg.columns = ['Customer ID', 'Average Basket Value']
        
        # --------------------- Basket Value Segmentation ---------------------
        # Quantiles for basket value segmentation
        basket_quantiles = basket_avg['Average Basket Value'].quantile([0.2, 0.8]).to_dict()

        # Categorize customers by basket value
        def basket_value_category(value):
            if value <= basket_quantiles[0.2]:
                return 'Low Basket Buyer'
            elif value <= basket_quantiles[0.8]:
                return 'Moderate Basket Buyer'
            else:
                return 'High Basket Buyer'

        basket_avg['Basket Value Category'] = basket_avg['Average Basket Value'].apply(basket_value_category)
        
        # Count items per transaction per customer
        items_per_txn = df.groupby(['Customer ID', 'Transaction ID']).size().reset_index(name='Item Count')
        avg_items = items_per_txn.groupby('Customer ID')['Item Count'].mean().reset_index()
        avg_items.columns = ['Customer ID', 'Average Items per Transaction']
        
        # --------------------- Items per Transaction Segmentation ---------------------
        # Categorize customers by item count
        def item_count_category(avg):
            if avg <= 2:
                return 'Selective Buyer'
            elif avg >= 5:
                return 'Bulk Buyer'
            else:
                return 'Mixed Buyer'

        avg_items['Item Count Category'] = avg_items['Average Items per Transaction'].apply(item_count_category)
        
        # --------------------- Merge Segmentations ---------------------
        basket_segmentation = basket_avg.merge(avg_items, on='Customer ID')
        basket_segmentation['Average Basket Value'] = basket_segmentation['Average Basket Value'].round(2)
        basket_segmentation['Average Items per Transaction'] = basket_segmentation['Average Items per Transaction'].round(2)
        
        self.log('Here is customer segmentation results based on Basket Composition for first 10 customers:')
        self.log(basket_segmentation.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Customer_Segmentation_based_on_Basket_Composition.csv'
        self.write_data(basket_segmentation, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')
        
        
        
class Based_on_Kmeans_Clustering:
    def __init__(self, path, transaction_file, in_folder, out_folder, log_folder):
        self.path = path
        self.transaction_file = transaction_file
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.log_folder = log_folder

        # Ensure log folder exists
        log_path = os.path.join(self.path, self.log_folder)
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, "Customer_Segmentation-K-means_Clustering.txt")

    def log(self, message):
        with open(self.log_file, "a") as file:
            file.write(message + "\n")

    def load_data(self, filename):
        df = pd.read_csv(self.path + self.in_folder + filename)
        return df

    def write_data(self, df, filename):
        output_path = os.path.join(self.path, self.out_folder)
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        df.to_csv(full_path, index=False)

    def run(self):
         # Clear log file at start
        open(self.log_file, 'w').close()

        ## --------------------- Read Raw tables --------------------------------- ##
        df = self.load_data(self.transaction_file)
        
        # Aggregate customer data
        customer_data = df.groupby('Customer ID').agg({'Purchase Amount': ['sum', 'mean', 'count']}).reset_index()
        
        # Rename columns for clarity
        customer_data.columns = ['Customer ID', 'Total Spend', 'Avg Spend', 'Purchase Count']
        
        # Features to use for clustering
        X = customer_data[['Total Spend', 'Avg Spend', 'Purchase Count']]
        
        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        
        ## --------------------- K-means Clustering --------------------------------- ##
        self.log('--> CUSTOMER SEGMENTATION BASED USING K-MEANS CLUSTERING\n')
        self.log('Performing K-means clustering with k=4:')
        self.log('\n')
        
        # Apply KMeans with the optimal number of clusters
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        customer_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster centers (in original scale)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_summary = pd.DataFrame(cluster_centers, columns=['Total Spend', 'Avg Spend', 'Purchase Count'])
        cluster_summary['Cluster'] = cluster_summary.index
        
        # Merge cluster labels to analyze statistics
        clustered_customers = customer_data.merge(cluster_summary, on='Cluster', suffixes=('', '_Centroid'))
        
        # Show the cluster summary sorted by Total Spend
        cluster_summary_sorted = cluster_summary.sort_values(by='Total Spend', ascending=False)
        cluster_summary_sorted.reset_index(drop=True, inplace=True)
        
        # Assign intuitive labels based on spending behavior
        labels = ['High Spenders', 'Moderate Buyers', 'Frequent Bargain Shoppers', 'Occasional Buyers']
        cluster_summary_sorted['Label'] = labels
        
        # Map these labels back to the customer data
        label_mapping = cluster_summary_sorted.set_index('Cluster')['Label'].to_dict()
        customer_data['Segment'] = customer_data['Cluster'].map(label_mapping)
        
        self.log('Here is customer segmentation results based on K-means clustering first 10 customers:')
        self.log(customer_data.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Customer_Segmentation_based_on_K-means_Clustering.csv'
        self.write_data(customer_data, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')
        
        cluster_summary_sorted[['Cluster', 'Total Spend', 'Avg Spend', 'Purchase Count', 'Label']]
        self.log('Here is summary of the clusters:')
        self.log(cluster_summary_sorted.to_string(index=False))
        self.log('')
        
        
        ## --------------------- PCA for better Clustering --------------------------------- ##
        self.log('-->PCA FOR BETTER CLUSTERING \n')
        self.log('Performing PCA to visualize better clusters:')
        self.log('\n')
        
        # Apply PCA to reduce to 2 components for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA results to the dataframe
        customer_data['PCA1'] = X_pca[:, 0]
        customer_data['PCA2'] = X_pca[:, 1]
        
        # Plot PCA-transformed data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=customer_data, x='PCA1', y='PCA2', hue='Segment', palette='Set2', s=100, alpha=0.7)
        plt.title('Customer Segments (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Customer Segment')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot as a PNG file
        plot_file = "PCA_Cluster_Visualization.png"
        plt.savefig(os.path.join(self.path, self.out_folder, plot_file), dpi=300, bbox_inches='tight')
        self.log(f'Saved Customer Segmentation visualization at: {self.path}{self.out_folder}{plot_file}')
        self.log('\n')