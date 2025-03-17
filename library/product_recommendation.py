import os
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules


class CollaborativeFiltering:
    def __init__(self, path, transaction_file, in_folder, out_folder, log_folder):
        self.path = path
        self.transaction_file = transaction_file
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.log_folder = log_folder

        # Ensure log folder exists
        log_path = os.path.join(self.path, self.log_folder)
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, "Product_Recommendation-Collaborative_Filtering.txt")

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
        
        
        ## --------------------- Collaborative Filtering --------------------------------- ##
        self.log('--> PRODUCT RECOMMENDATION USING COLLABORATIVE FILTERING\n')
        self.log('Recommending Products using Collaborative Filtering method:')
        self.log('\n')
        
        # Step 1: Create a customer-product matrix (purchase count per product)
        customer_product_matrix = df.pivot_table(index='Customer ID', columns='Product ID', values='Purchase Amount', aggfunc='count', fill_value=0)
        
        # Step 2: Compute cosine similarity between customers
        customer_similarity = cosine_similarity(customer_product_matrix)
        similarity_df = pd.DataFrame(customer_similarity, index=customer_product_matrix.index, columns=customer_product_matrix.index)
        
        # Step 3: Generate recommendations
        recommendations = {}

        for customer_id in customer_product_matrix.index:
            # Find top 5 similar customers (excluding the customer themself)
            similar_customers = similarity_df[customer_id].drop(customer_id).sort_values(ascending=False).head(5).index

            # Get products purchased by similar customers
            similar_customers_purchases = customer_product_matrix.loc[similar_customers]
            summed_purchases = similar_customers_purchases.sum(axis=0)

            # Remove products already purchased by the customer
            already_purchased = customer_product_matrix.loc[customer_id]
            products_to_recommend = summed_purchases[already_purchased == 0]

            # Get top 5 recommended product IDs
            top_5_products = products_to_recommend.sort_values(ascending=False).head(5).index.tolist()
            recommendations[customer_id] = top_5_products
            
        # Convert to DataFrame for display
        recommendation_df = pd.DataFrame([{'Customer ID': customer, 'Recommended Products': products} for customer, products in recommendations.items()])
        
        # Get past purchases for each customer
        customer_purchases = df.groupby('Customer ID')['Product ID'].apply(lambda x: list(sorted(set(x)))).reset_index()
        customer_purchases.columns = ['Customer ID', 'Past Purchases']

        # Merge with recommendations
        full_recommendation_df = customer_purchases.merge(recommendation_df, on='Customer ID')
        
        self.log('Here is product recommendation using collaborative filtering for first 10 customers:')
        self.log(full_recommendation_df.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Product_Recommendation-Collaborative_Filtering.csv'
        self.write_data(full_recommendation_df, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')

        
        
class MarketBasketAnalysis:
    def __init__(self, path, transaction_file, product_file, in_folder, out_folder, log_folder):
        self.path = path
        self.transaction_file = transaction_file
        self.product_file = product_file
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.log_folder = log_folder

        # Ensure log folder exists
        log_path = os.path.join(self.path, self.log_folder)
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, "Product_Recommendation-Market_Basket_Analysis.txt")

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
        product_df = self.load_data(self.product_file)
        
        
        ## --------------------- Collaborative Filtering --------------------------------- ##
        self.log('--> PRODUCT RECOMMENDATION USING COLLABORATIVE FILTERING\n')
        self.log('Recommending Products using Collaborative Filtering method:')
        self.log('\n')
        
        # Prepare transaction data
        basket_data = df.groupby("Transaction ID")["Product ID"].apply(list).tolist()
        
        # Encode transaction list
        te = TransactionEncoder()
        te_ary = te.fit(basket_data).transform(basket_data)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        # Display sorted rules
        rules.sort_values(by="lift", ascending=False, inplace=True)
#         print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        
        ## --------------------- Looking at Rules at SKU level --------------------------------- ##
        # Step 1: Flatten antecedents and consequents
        rules['antecedent_ids'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequent_ids'] = rules['consequents'].apply(lambda x: list(x))
        
        # Step 2: Explode lists for mapping
        exploded_ante = rules[['antecedent_ids']].explode('antecedent_ids')
        exploded_cons = rules[['consequent_ids']].explode('consequent_ids')

        # Step 3: Map Product ID → SKU using a dictionary or merge
        id_to_sku = product_df.set_index('Product ID')['SKU'].to_dict()
        exploded_ante['SKU'] = exploded_ante['antecedent_ids'].map(id_to_sku)
        exploded_cons['SKU'] = exploded_cons['consequent_ids'].map(id_to_sku)
        
        # Step 4: Re-aggregate SKUs
        ante_sku = exploded_ante.groupby(level=0)['SKU'].apply(list)
        cons_sku = exploded_cons.groupby(level=0)['SKU'].apply(list)
        
        # Step 5: Add back to rules DataFrame
        rules['antecedent_skus'] = ante_sku
        rules['consequent_skus'] = cons_sku
        
        # Display sorted rules
        rules.sort_values(by="lift", ascending=False, inplace=True)
        rules[['antecedent_skus', 'consequent_skus', 'support', 'confidence', 'lift']].head(10)
        
        ## --------------------- Making Recommendation --------------------------------- ##
        # Step 1: Build simple co-occurrence based "rules" using product pair frequenc
        # Create product pair frequency dictionary
        pair_counts = defaultdict(int)

        # Iterate through transactions and count pairs
        transactions = df.groupby("Transaction ID")["Product ID"].apply(set)
        for items in transactions:
            for item_pair in combinations(sorted(items), 2):
                pair_counts[item_pair] += 1
                
        # Step 2: Create a mapping from each product to others often bought together
        product_to_related = defaultdict(set)
        for (prod1, prod2), count in pair_counts.items():
            if count >= 5:  # threshold to filter meaningful associations
                product_to_related[prod1].add(prod2)
                product_to_related[prod2].add(prod1)

        # Step 3: Prepare customer-level product history
        customer_products = df.groupby("Customer ID")["Product ID"].apply(set).reset_index()
        
        # Step 4: Recommend products not already bought but often co-purchased
        def recommend_from_pairs(purchased):
            recommendations = set()
            for pid in purchased:
                related = product_to_related.get(pid, set())
                recommendations.update(related)
            return list(recommendations - purchased)

        customer_products["Previous Purchases"] = customer_products["Product ID"].apply(list)
        customer_products["Recommended Products"] = customer_products["Product ID"].apply(recommend_from_pairs)
        
        # Limit recommendations to top 5 (already sorted or just first 5 from the list)
        customer_products["Recommended Products"] = customer_products["Recommended Products"].apply(lambda x: x[:5])

        # Final formatting and export
        recommendation_result = customer_products[["Customer ID", "Previous Purchases", "Recommended Products"]]
        
        self.log('Here is product recommendation using market basket analysis for first 10 customers:')
        self.log(recommendation_result.head(10).to_string(index=False))
        self.log('')
        
        file_name = 'Product_Recommendation-Market_Basket_Analysis.csv'
        self.write_data(recommendation_result, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')
        self.log('\n')