# BookedBy: Customer Segmentation & Product Recommendation

**BookedBy** is a data-driven application designed for a retail company to enhance the customer experience using purchase behavior insights. It analyzes transaction data to identify patterns, segment customers into meaningful groups, and recommend personalized products.

---

## 🚀 Project Overview

This project performs the following tasks:

1. **Data Analysis** – Understand customer purchase trends and product demand.
2. **Customer Segmentation** – Group customers based on their buying behavior.
3. **Product Recommendation** – Suggest relevant products to customers based on their purchase history.

---

## 📂 Project Structure

```
BookedBy/
├── Data/ 
│   ├── logs/                      # Task-specific logs (e.g., data_analysis_log.txt)
│   ├── result/                    # Output files: customer segments, recommendations
│   ├── raw_data/ 
│   │   ├── Transaction.csv        # Transactional purchase data (Jan 2024 – Feb 2025)
│   │   ├── Product.csv            # Product metadata (50 products, 15 categories)
│   │   └── Customer.py            # Customer data and processing logic
├── library/
│   ├── customer_segmentation.py  # Clustering logic to group customers
│   ├── data_analysis.py          # Insights from transaction data
│   └── product_recommendation.py # Personalized product recommendation engine
├── script.py                     # Main runner script
├── Report.pdf                    # Report summarizing work done and results for the project
└── README.md
```

---

## 📊 Data Summary

- **Number of Customers:** 500  
- **Number of Product Categories:** 15  
- **Categories:**  
  Cleansers, Exfoliators, Toners, Serums, Moisturizers, Sunscreens (SPF), Face Masks, Eye Care, Lip Care, Acne Treatment, Anti-Aging Treatment, Sensitive Skin Treatment, Skin Repair & Healing, Pore Care  
- **Number of Products:** 50  
- **Transaction Date Range:** January 1, 2024 – February 28, 2025

---

## 🛠 How to Run

1. Clone or download this repository.
2. Open your terminal and navigate to the `BookedBy/` directory.
3. Run the script:

```bash
python script.py
```

---

## 📁 Output

- **Logs:** Task execution logs will be saved in the `Data/logs/` directory.
- **Results:**  
  - `customer_segments.csv` – Output from the segmentation process  
  - `product_recommendations.csv` – Personalized recommendations for each customer  
  These can be found in the `Data/result/` directory.

---

## 📌 Dependencies

Ensure you have Python 3.x installed. Required packages include (but are not limited to):

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install all dependencies via:

```bash
pip install -r requirements.txt
``` 
