import os
import pandas as pd

class DataAnalysis:
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
        self.log_file = os.path.join(log_path, "Data_Analysis.txt")

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

        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        merged_df = df.merge(product_df[['Product ID', 'SKU']], on='Product ID', how='left')

        ## --------------------- Top Selling Products & Categories --------------------------------- ##
        self.log('--> TOP SELLING PRODUCTS & CATEGORIES\n')

        top_products = merged_df['Product ID'].value_counts().head(10)
        top_product_ids = top_products.index.tolist()
        top_selling_with_sku = product_df[product_df['Product ID'].isin(top_product_ids)][['Product ID', 'SKU']]
        top_selling_with_sku = top_selling_with_sku.set_index('Product ID').loc[top_product_ids].reset_index()
        top_selling_with_sku['Sales Count'] = top_products.values

        self.log('(i)  Here are the details about the top 10 selling products:')
        self.log(top_selling_with_sku.head(10).to_string(index=False))
        self.log('')

        top_category_counts = df['Product Category'].value_counts().head(10).reset_index()
        top_category_counts.columns = ['Product Category', 'Count']

        self.log('(ii) Here are the details about the top 3 selling product categories:')
        self.log(top_category_counts.head(3).to_string(index=False))
        self.log('\n')

        ## --------------------- Average Spending per Customer per Transaction --------------------------------- ##
        self.log('--> AVERAGE SPENDING PER CUSTOMER PER TRANSACTION\n')

        transaction_spending = df.groupby('Transaction ID').agg({'Customer ID': 'first', 'Purchase Amount': 'sum'}).reset_index()
        customer_avg_spending = transaction_spending.groupby('Customer ID')['Purchase Amount'].agg(
            Total_Spending='sum',
            Number_of_Transactions='count',
            Average_Spending_Per_Transaction='mean').reset_index()
        customer_avg_spending['Average_Spending_Per_Transaction'] = customer_avg_spending['Average_Spending_Per_Transaction'].round(2)

        self.log('Here are the details about the average spending per customer per transaction for first 5 customers:')
        self.log(customer_avg_spending.head(10).to_string(index=False))
        self.log('')

        file_name = 'Average_spending_per_customer_per_transaction.csv'
        self.write_data(customer_avg_spending, file_name)
        self.log(f'Find full file at: {self.path}{self.out_folder}{file_name}\n')

        ## --------------------- Spending Statistics --------------------------------- ##
        self.log('--> SPENDING STATISTICS\n')

        transaction_summary = df.groupby('Transaction ID').agg(
            Customer_ID=('Customer ID', 'first'),
            Transaction_Spending=('Purchase Amount', 'sum')).reset_index()
        average_transaction_spending = round(transaction_summary['Transaction_Spending'].mean(), 2)
        min_transaction_spending = transaction_summary['Transaction_Spending'].min()
        max_transaction_spending = transaction_summary['Transaction_Spending'].max()

        self.log('Here are the details about the Spending Statistics:')
        self.log(f'1. Minimum Spending in a transaction: {min_transaction_spending}')
        self.log(f'2. Maximum Spending in a transaction: {max_transaction_spending}')
        self.log(f'3. Average Spending per transaction: {average_transaction_spending}')
        self.log('\n')