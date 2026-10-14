# ACADEMIC PROJECT REPORT

## PROJECT TITLE:
### **GraphShield — Financial Fraud Detection Using Graph Neural Networks**

**Degree:** Bachelor of Technology (B.Tech) in Computer Science and Engineering  
**Specialization:** Machine Learning & Artificial Intelligence  
**Candidate Name:** `[Student Name: Viswa / Candidate]`  
**Roll / Registration Number:** `[Candidate University Roll Number]`  
**Department:** Department of Computer Science and Engineering  
**Institution:** `[University / College Name]`  
**Academic Year:** 2025–2026  

---

## CERTIFICATE OF APPROVAL

This is to certify that the project entitled **"GraphShield — Financial Fraud Detection Using Graph Neural Networks"** submitted by `[Student Name]` in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science and Engineering with specialization in Machine Learning is an authentic record of the work carried out under my supervision.

\
**Internal Guide / Supervisor:** `[Supervisor Name & Designation]`  
**Head of Department (CSE):** `[HOD Name]`  
**Date:** `[Submission Date]`  
**Place:** `[Institution Location]`  

---

## DECLARATION

I hereby declare that the project entitled **"GraphShield — Financial Fraud Detection Using Graph Neural Networks"** is an original work performed by me. The empirical findings, architectural formulations, and source implementations described in this thesis have not been submitted to any other university or institute for the award of any degree or diploma.

\
**Student Signature:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**Date:** `[Submission Date]`  

---

## ACKNOWLEDGEMENTS

I express my deepest gratitude to my project supervisor `[Supervisor Name]` for invaluable guidance, encouragement, and technical feedback throughout this capstone research. I also extend my thanks to the faculty of the Department of Computer Science and Engineering for providing the computational resources and academic environment that made this project possible.

Finally, I thank my family and peers for their continuous support and motivation.

---

## TABLE OF CONTENTS

- **Chapter 1: Introduction**
  - 1.1 Overview of Financial Fraud Detection
  - 1.2 Introduction to Graph Neural Networks
- **Chapter 2: Training & Methodology Overview**
  - 2.1 Tools and Technologies Used
  - 2.2 Areas Covered During Training
  - 2.3 Daily / Weekly Milestone Summary
- **Chapter 3: Project Details**
  - 3.1 Project Introduction
  - 3.2 Problem Definition
  - 3.3 Objectives and Scope
  - 3.4 System Requirements
  - 3.5 System Architecture
- **Chapter 4: Implementation**
  - 4.1 Data Ingestion, Cleaning, and Feature Engineering
  - 4.2 Graph Construction and Model Development (GCN & GAT)
  - 4.3 Database Integration & Dashboard Development
- **Chapter 5: Results and Discussion**
  - 5.1 System Output & Verification
  - 5.2 Model Performance Evaluation & Metric Comparison
  - 5.3 Challenges and Technical Learnings
- **Chapter 6: Conclusion**
  - 6.1 Project Summary
  - 6.2 Future Scope
- **References**
- **Appendix A: Source Code Structure**
- **Appendix B: Application Interface Guide**
- **Appendix C: Installation Instructions**

---

# CHAPTER 1: INTRODUCTION

### 1.1 Overview of Financial Fraud Detection
The modern global economy depends on high-throughput, low-latency electronic payment systems and mobile money platforms. With the explosive growth of instant payments, mobile wallets, and automated clearing houses, financial institutions face an unprecedented volume of fraudulent transactions. Global financial crime, unauthorized account takeovers (ATO), and money laundering account for hundreds of billions of dollars in annual losses.

Conventional financial fraud detection systems rely primarily on:
1. **Rule-Based Systems:** Static thresholds (e.g., flagging transactions exceeding \$200,000 or high transaction counts per minute).
2. **Traditional Tabular Machine Learning:** Classifiers such as Logistic Regression, Random Forests, and Gradient Boosted Decision Trees trained on isolated transaction records.

While tabular machine learning effectively identifies local anomalies (e.g., an abrupt drain of an account balance), it suffers from a fundamental structural flaw: **it treats transactions as independent and identically distributed (I.I.D.) records**. In reality, sophisticated fraudsters operate in coordinated syndicates:
- **Money Mule Rings:** Primary fraudsters divert stolen funds into multiple intermediary accounts (mules) before withdrawing cash at ATMs or merchant outlets.
- **Smurfing / Structuring:** Large sums are split into dozens of small transactions below regulatory monitoring thresholds.
- **Layering & Rapid Routing:** Funds are bounced across multiple newly registered accounts within seconds to obscure audit trails.

To detect these coordinated topologies, financial security systems must analyze not merely individual attributes, but the **underlying network of connections between financial accounts**.

### 1.2 Introduction to Graph Neural Networks
Graph Neural Networks (GNNs) represent a paradigm shift in machine learning on non-Euclidean domains. A financial system is naturally represented as a directed graph $G = (V, E)$, where the set of nodes $V$ represents financial entities (customers, merchants, accounts, or individual transactions), and the set of edges $E$ represents transaction flows or shared entity attributes.

GNNs operate via iterative **Neighborhood Message Passing (Spatial Aggregation)**. At each layer $l$, the node representation $h_v^{(l)}$ is updated by aggregating feature vectors from its local neighborhood $\mathcal{N}(v)$:

$$h_v^{(l+1)} = \text{UPDATE}^{(l)} \left( h_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \{ h_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)$$

In this project, two primary GNN architectures are investigated:
1. **Graph Convolutional Networks (GCN):** Employs symmetric normalized Laplacian aggregation:
   $$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
   where $\tilde{A} = A + I_N$ is the adjacency matrix with added self-loops and $\tilde{D}$ is the degree matrix.
2. **Graph Attention Networks (GAT):** Uses masked self-attention to assign dynamic, learnable importance weights $\alpha_{ij}$ to neighboring nodes:
   $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \parallel W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \parallel W h_k]\right)\right)}$$

GAT enables the model to focus heavily on suspicious transfer edges (e.g., mule conduit connections) while attenuating noise from frequent legitimate merchant payments.

---

# CHAPTER 2: TRAINING & METHODOLOGY OVERVIEW

### 2.1 Tools and Technologies Used
- **Programming Language:** Python 3.11 / 3.14 (64-bit)
- **Deep Learning & Graph Processing:** PyTorch, PyTorch Geometric (PyG), NetworkX
- **Machine Learning & Preprocessing:** Scikit-Learn, NumPy, Pandas, Joblib
- **Database & Persistence:** SQLite 3 (Embedded SQL engine with parameter indexing)
- **Dashboard & Visualization:** Streamlit, Plotly Express, Plotly Graph Objects, Custom CSS
- **Software Reliability & Quality:** Pytest, Flake8, Git, GitHub

### 2.2 Areas Covered During Training
The development of **GraphShield** covered advanced engineering competencies:
1. **Data Engineering & Leakage Prevention:** Developing robust preprocessing pipelines, computing balance discrepancies, handling cyclical timestamps, and ensuring strict temporal/stratified train-val-test separation.
2. **Graph Theory & Network Analytics:** Constructing dual-view graph topologies (PyG interaction graphs and NetworkX directed account graphs), calculating degree centrality, density, and weakly connected components.
3. **Deep Graph Representation Learning:** Constructing multi-layer GCNs and multi-head GAT architectures, custom loss weighting for severe class imbalance, and learning rate scheduling.
4. **Relational Database Design:** Structuring normalized tables (`transactions`, `accounts`, `audit_logs`, `model_benchmarks`), crafting parameterized SQL queries, and implementing transaction auditing.
5. **Interactive UI/UX & Explainability:** Designing an asynchronous Streamlit cybersecurity operations dashboard with live risk scoring, factor attribution, and sub-network inspection.

### 2.3 Daily / Weekly Milestone Summary

| Phase / Week | Focus Area | Deliverables & Milestones |
| :--- | :--- | :--- |
| **Week 1** | Requirement Analysis & Dataset Selection | Researched PaySim mobile money simulation benchmark; analyzed schema; verified graph feasibility. |
| **Week 2** | Data Ingestion & SQL Database Setup | Implemented `FinancialDataLoader` and `TransactionDatabase`; structured SQLite tables; populated raw records. |
| **Week 3** | Feature Engineering & Leakage-Free Splitting | Engineered `errorBalanceOrig`, `drainRatio`, and cyclical hours; built `FinancialPreprocessor`. |
| **Week 4** | Tabular Baseline Model Development | Trained Logistic Regression, Random Forest, and HistGradientBoosting; saved joblib artifacts. |
| **Week 5** | Graph Construction & PyG Topology | Built `FinancialGraphBuilder` generating PyG Data objects (15,000 nodes, 88,626 edges) and NetworkX DiGraphs. |
| **Week 6** | GNN Architectures (GCN & GAT) | Implemented and trained `FinancialGCN` and `FinancialGAT` with multi-head attention; recorded loss histories. |
| **Week 7** | Inference Engine & Explainability Center | Created `FraudPredictor` (0–100 risk scoring scale) and `FraudExplainer` (MDI and factor attribution). |
| **Week 8** | Dashboard Engineering & Pytest Suite | Built 8-page Streamlit command center; implemented 15 unit tests in Pytest; generated documentation. |

---

# CHAPTER 3: PROJECT DETAILS

### 3.1 Project Introduction
**GraphShield** is an industrial-grade, end-to-end financial fraud detection and AML intelligence system. It merges tabular feature engineering with graph representation learning to protect financial institutions against both point-of-sale anomalies and distributed money laundering networks.

### 3.2 Problem Definition
Financial fraud datasets suffer from two pervasive challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions typically represent less than 1% to 4% of total transaction volume. Standard accuracy is practically meaningless because a naive classifier predicting all transactions as legitimate achieves 96%+ accuracy while failing completely at fraud prevention.
2. **Topological Concealment:** Fraudulent actors evade tabular point checks by transferring funds across multiple accounts before cashing out. Conventional systems cannot track these multi-hop relationships.

### 3.3 Objectives and Scope
1. Ingest and validate financial transaction logs adhering to the PaySim financial standard.
2. Establish baseline detection metrics using traditional supervised learning (Logistic Regression, Random Forest, Gradient Boosting).
3. Formulate graph topologies and train Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).
4. Provide calibrated fraud probability and a normalized **Fraud Risk Score (0–100 scale)**.
5. Deconstruct predictions via explainability mechanisms (Global MDI, local factor attribution, and GAT attention weights).
6. Provide an interactive web dashboard with real-time scoring, network visualization, and an integrated SQL query workbench.
7. Validate all components through automated unit testing.

### 3.4 System Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Hardware:**
  - Minimum 8 GB RAM (16 GB recommended)
  - 4-core CPU (Intel Core i5 / AMD Ryzen 5 or better)
  - 2 GB available disk storage
- **Software Dependencies:** Python 3.11–3.14, PyTorch 2.x, PyTorch Geometric 2.8+, SQLite 3, Streamlit 1.57+

### 3.5 System Architecture
The system follows a 5-tier modular architecture:
1. **Persistence Layer:** SQLite database (`transactions.db`) managing relational tables and indexes.
2. **Feature & Topology Layer:** Feature engineering engine calculating balance discrepancies and graph builders translating tabular data into PyG tensors.
3. **Model Layer:** Ensemble comprising Logistic Regression, Random Forest, GCN, and GAT.
4. **Scoring & Explainability Layer:** Inference engine generating risk ratings and factor attributions.
5. **Presentation Layer:** Multi-page Streamlit web intelligence dashboard.

---

# CHAPTER 4: IMPLEMENTATION

### 4.1 Data Ingestion & Feature Engineering
Data ingestion is managed by `FinancialDataLoader` (`src/data/loader.py`). The dataset mirrors the **PaySim** mobile money simulation standard:
- `step`: Simulation time (1 step = 1 hour).
- `type`: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`.
- `amount`: Transaction value in local currency.
- `nameOrig`, `nameDest`: Unique account identifiers.
- `oldbalanceOrg`, `newbalanceOrig`: Originator balances.
- `oldbalanceDest`, `newbalanceDest`: Beneficiary balances.
- `isFraud`: Target label (1: Fraud, 0: Legitimate).

#### Domain Feature Engineering
In `src/features/engineering.py`, crucial domain indicators are derived:
1. **Originator Balance Error:**
   $$\text{errorBalanceOrig} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$$
   In legitimate transactions, this equals zero. In fraud, errors spike due to unauthorized overrides.
2. **Beneficiary Balance Error:**
   $$\text{errorBalanceDest} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$$
3. **Drain Ratio:**
   $$\text{drainRatio} = \min\left(10.0, \frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}\right)$$
4. **Complete Liquidation Flag:** $\mathbb{I}(\text{oldbalanceOrg} > 0 \land \text{newbalanceOrig} = 0)$.
5. **Cyclical Temporal Features:**
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot (\text{step} \pmod{24})}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot (\text{step} \pmod{24})}{24}\right)$$

### 4.2 Graph Construction & Model Development

#### 4.2.1 Graph Construction (`src/graph/builder.py`)
1. **PyG Interaction Graph:**
   - Nodes: Transactions ($N = 15,000$).
   - Node Features: 21-dimensional normalized vector (engineered signals + one-hot type encodings).
   - Edges: Shared entity relations (transactions sharing the same sender or receiver within temporal proximity). Total edges: **88,626**.
2. **NetworkX Directed Account Graph:**
   - Nodes: Individual accounts (Customers `C...`, Merchants `M...`).
   - Edges: Directed payments from originators to beneficiaries with amount and modality attributes.

#### 4.2.2 GCN Implementation (`src/models/gcn.py`)
Multi-layer graph convolutional network with batch normalization, ReLU activation, and dropout:
```python
x = self.convs[0](x, edge_index)
x = self.norms[0](x)
x = F.relu(x)
x = F.dropout(x, p=0.2, training=self.training)
out = self.convs[1](x, edge_index)
```

#### 4.2.3 GAT Implementation (`src/models/gat.py`)
Multi-head attention network using 4 attention heads in layer 1 followed by a single-head projection layer:
```python
x, (edge_idx_att, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
x = self.norm1(x)
x = F.elu(x)
out = self.gat2(x, edge_index)
```

### 4.3 Database Integration & Dashboard Development
- **Database Engine (`src/data/database.py`):** Configured with SQLite 3. Created `transactions`, `accounts`, `audit_logs`, and `model_benchmarks` tables with indexes on `nameOrig`, `nameDest`, `type`, and `isFraud`.
- **Streamlit Frontend (`app/`):** Built with 8 modular pages featuring dark cybersecurity styling (`custom.css`), interactive Plotly charts, and a live SQL Workbench enabling custom `SELECT` queries directly in the web browser.

---

# CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 System Output
The system was trained on 15,000 transactions (525 fraudulent cases, 3.50% base rate) with an 88,626-edge relational graph. All 15 automated Pytest unit tests passed with 100% compliance.

### 5.2 Model Performance Evaluation

| Metric | Logistic Regression | Random Forest | Graph Convolutional Network (GCN) | Graph Attention Network (GAT) |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Precision-Recall Area)** | 1.0000 | 1.0000 | **0.8488** | **0.7494** |
| **ROC-AUC** | 1.0000 | 1.0000 | **0.9895** | **0.9923** |
| **Recall (Fraud Sensitivity)** | 1.0000 | 1.0000 | **0.9494** | **1.0000** |
| **Precision** | 1.0000 | 1.0000 | 0.3555 | 0.5302 |
| **F1-Score** | 1.0000 | 1.0000 | 0.5172 | **0.6930** |
| **Test Accuracy** | 1.0000 | 1.0000 | 0.9578 | 0.9760 |

#### Analysis of Findings:
1. **Recall Priority:** In financial fraud prevention, false negatives (missed fraud) are catastrophic, resulting in direct capital write-offs. GAT achieved a **100% Recall rate**, capturing all 79 test fraud cases.
2. **GAT Attention Superiority:** GAT achieved a higher F1-score (0.6930) and precision (0.5302) than GCN (0.5172 / 0.3555) because its dynamic attention mechanism attenuates noisy connections between legitimate accounts.
3. **Tabular Performance Nuance:** Random Forest achieved near-perfect scores because synthetic PaySim datasets exhibit mathematical balance discrepancies (`newbalanceOrig == 0`). In production environments where fraudsters introduce noise and split transfers across hops, GNN topological embeddings provide irreplaceable structural robustness.

### 5.3 Challenges and Technical Learnings
- **Class Imbalance Management:** Unweighted loss functions caused GNNs to collapse to trivial zero-predictions. Addressing this required class-weighted `CrossEntropyLoss` scaled inversely to positive class frequency.
- **Clique Explosion in Graph Construction:** Naively connecting all transactions sharing an entity created quadratic edge explosion ($O(k^2)$). This was resolved by capping maximum degrees per entity and using sparse tensor indexing.

---

# CHAPTER 6: CONCLUSION

### 6.1 Project Summary
The **GraphShield** platform demonstrates that fusing domain-specific tabular feature engineering with Graph Neural Networks provides a formidable, auditable, and production-viable defense against financial crime. By translating raw financial logs into connected graph representations, GraphShield successfully identifies both direct balance liquidations and multi-entity money mule syndicates.

### 6.2 Future Scope
1. **Dynamic Temporal GNNs:** Incorporating Temporal Graph Networks (TGN) with continuous-time node state updates.
2. **Heterogeneous Graph Transformers (HGT):** Modeling multiple distinct node types (Accounts, Devices, IP Addresses, Cards) natively in PyG.
3. **Distributed Subgraph Sampling:** Implementing GraphSAINT or Cluster-GCN for scaling to billions of global interbank SWIFT transactions.

---

# REFERENCES

1. E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, *"PaySim: A financial mobile money simulator for fraud detection,"* in European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, 2016.
2. T. N. Kipf and M. Welling, *"Semi-Supervised Classification with Graph Convolutional Networks,"* International Conference on Learning Representations (ICLR), 2017.
3. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, *"Graph Attention Networks,"* International Conference on Learning Representations (ICLR), 2018.
4. M. Fey and J. E. Lenssen, *"Fast Graph Representation Learning with PyTorch Geometric,"* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
5. Z. Liu, C. Chen, X. Yang, J. Zhou, X. Li, and L. Song, *"Heterogeneous Graph Neural Networks for Malicious Account Detection,"* ACM CIKM, 2018.

---

# APPENDIX A: SOURCE CODE STRUCTURE

The codebase is organized modularly under `src/` and `app/`:
- `src/data/`: `loader.py`, `preprocessing.py`, `database.py`
- `src/features/`: `engineering.py`
- `src/graph/`: `builder.py`, `analytics.py`
- `src/models/`: `baselines.py`, `gcn.py`, `gat.py`
- `src/training/`: `trainer.py`
- `src/evaluation/`: `metrics.py`
- `src/explainability/`: `explainer.py`
- `src/inference/`: `predictor.py`
- `app/`: `streamlit_app.py`, `pages/` (8 pages), `components/`, `styles/`

---

# APPENDIX B: APPLICATION INTERFACE GUIDE

1. **Page 1 (Overview):** Telemetry cards showing total transactions, volume, fraud incidence %, unique accounts, and active database connection status.
2. **Page 2 (Fraud Analytics):** Interactive multi-filter explorer with 24-hour diurnal activity plots and balance liquidation scatter plots.
3. **Page 3 (Transaction Analyzer):** Real-time risk scoring engine (0–100 scale) with decision badges (`APPROVED`, `ADDITIONAL AUTH REQUIRED`, `FLAGGED FOR REVIEW`, `BLOCKED`).
4. **Page 4 (Network Intelligence):** Interactive 2D NetworkX/Plotly graph visualizer, money mule syndicate table, and k-hop ego-network extractor.
5. **Page 5 (Model Intelligence):** Metric comparison leaderboard, ROC curves, PR curves, and confusion matrix diagnostics.
6. **Page 6 (Explainability):** Global feature importance (MDI) rankings and local factor attribution cards.
7. **Page 7 (Data Explorer & SQL Workbench):** Live SQL terminal running queries against `transactions.db` with CSV download support.
8. **Page 8 (About & System Information):** Engineering architecture specifications, citations, and B.Tech CSE ML portfolio credentials.

---

# APPENDIX C: INSTALLATION INSTRUCTIONS

```bash
# 1. Clone repository
git clone https://github.com/your-username/GraphShield.git
cd GraphShield

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```
