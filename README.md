# Smart Customer Management Portal with AI-Driven Insights

## Overview
This project implements a lightweight customer management portal for a fictional networking hardware company.
It includes:
- Synthetic data generation for 200+ customer accounts
- CRUD workflows for customers, support tickets, and device inventory
- Natural language analytics query interface with follow-up context handling
- Account health scoring
- Churn prediction with precision/recall reporting
- Weekly account review email summary generator

## Tech Stack
- UI: Streamlit
- Data store: SQLite (`portal.db`)
- ML: scikit-learn (logistic regression)
- Data generation: Faker + custom heuristics

## Setup
1. Create and activate your Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run app:
   ```bash
   streamlit run app/main.py
   ```

## Demo Flow
1. Open the app and use sidebar action **Seed / Reseed synthetic data** (min 200 records).
2. Visit **Customers**, **Tickets**, and **Devices** pages to show CRUD operations.
3. In **AI Insights**, click:
   - **Refresh Account Health Scores**
   - **Run Churn Prediction Training**
4. Run natural language queries in increasing complexity.
5. Open **Email Agent** and generate weekly summary for any customer.

## Natural Language Demo Prompts (10+)
1. Show all customers
2. Show all customers in EMEA
3. only enterprise plans
4. Show open tickets
5. Show open tickets in APAC
6. Which accounts have high churn risk?
7. Only in North America
8. Show account health for enterprise customers
9. Show usage trends for APAC
10. List customers with professional plans in LATAM
11. Show churn risk in EMEA enterprise

## Deliverable Evidence
- Churn metrics displayed in-app after model training.
- 200+ records can be validated from overview metric and seeded summary.
- Design details are documented in `docs/design.md`.
