import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import Draw
import pandas as pd
import base64
from io import BytesIO

st.set_page_config(page_title="AI Drug Generator Demo", layout="wide")
st.title("💊 AI-Driven De Novo Drug Design - Molecule Screening Demo")

st.markdown("""
This app demonstrates a simplified AI-driven drug design platform.
- Upload a CSV of SMILES
- Compute QED, MW, logP, and Lipinski Rule
- Visualize molecules and export results
""")

def validate_smiles(smiles_list):
    valid = []
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid.append(smi)
            mols.append(mol)
    return valid, mols

def simulated_sa_score(mol):
    return Descriptors.NumRotatableBonds(mol) + 2

def draw_molecule_grid(mols, legends):
    return Draw.MolsToGridImage(mols, molsPerRow=4, legends=legends, subImgSize=(200,200))

uploaded_file = st.file_uploader("📁 Upload CSV with SMILES column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    smiles_column = st.selectbox("Select SMILES column", df.columns.tolist())
    smiles_list = df[smiles_column].dropna().astype(str).tolist()

    valid_smiles, mols = validate_smiles(smiles_list)
    st.success(f"✅ Found {len(valid_smiles)} valid molecules.")

    results = []
    for smi, mol in zip(valid_smiles, mols):
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = QED.qed(mol)
        sa_score = simulated_sa_score(mol)
        lipinski = all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
        results.append({
            "SMILES": smi,
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "QED": round(qed, 2),
            "SA Score (simulated)": sa_score,
            "Lipinski Pass": lipinski
        })

    result_df = pd.DataFrame(results)
    st.subheader("📊 Molecule Properties Table")
    st.dataframe(result_df)

    st.subheader("🧪 Molecule Visualization")
    legends = [f"QED={row['QED']}" for row in results[:8]]
    grid_img = draw_molecule_grid(mols[:8], legends)
    st.image(grid_img)

    csv_buffer = BytesIO()
    result_df.to_csv(csv_buffer, index=False)
    b64 = base64.b64encode(csv_buffer.getvalue()).decode()
    st.markdown(f"📥 [Download Results CSV](data:file/csv;base64,{b64})", unsafe_allow_html=True)

else:
    st.info("👆 Upload a CSV with SMILES to begin.")
