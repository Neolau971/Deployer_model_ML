from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
import pytest
from fastapi import UploadFile, HTTPException

import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.main import load_data_from_upload

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "dataCentral.csv"

def make_upload_file():
    csv_text = CSV_PATH.read_text(encoding="utf-8")
    return SimpleNamespace(file=BytesIO(csv_text.encode("utf-8")))

def test_empty_csv_raises_400():
    csv_content = ""
    file_like = BytesIO(csv_content.encode("utf-8"))
    upload = UploadFile(filename="empty.csv", file=file_like)
    
    with pytest.raises(HTTPException) as exc_info:
        load_data_from_upload(upload)
    
    assert exc_info.value.status_code == 400
    assert "Erreur de lecture CSV" in exc_info.value.detail

def test_load_data_from_upload_returns_dataframe():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_expected_columns_present():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    expected = {
    "satisfaction_employee_environnement",       
    "note_evaluation_precedente",                
    "niveau_hierarchique_poste",                  
    "satisfaction_employee_nature_travail",      
    "satisfaction_employee_equipe",              
    "satisfaction_employee_equilibre_pro_perso", 
    "note_evaluation_actuelle", 
    "heure_supplementaires", 
    "augementation_salaire_precedente", 
    "nombre_participation_pee",                   
    "nb_formations_suivies",    
    "distance_domicile_travail",  
    "niveau_education",                          
    "annees_depuis_la_derniere_promotion",        
    "annes_sous_responsable_actuel",    
    "age",                                                                             
    "nombre_experiences_precedentes",             
    "annee_experience_totale",                    
    "annees_dans_l_entreprise",                   
    "annees_dans_le_poste_actuel" 
    }
    assert expected.issubset(df.columns)

def test_valid_csv_multiple_rows():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    assert len(df) > 1

def test_invalid_age_second_row_validation():
    df = pd.read_csv(CSV_PATH, dtype=str)
    
    df.loc[1, 'age'] = "abc"
    
    csv_content = df.to_csv(index=False)
    file_like = BytesIO(csv_content.encode("utf-8"))
    upload = UploadFile(filename="modified.csv", file=file_like)
    
    with pytest.raises(HTTPException) as exc_info:
        df = load_data_from_upload(upload)
    
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["row"] == 1

def test_invalid_column_name_age_raises_422():
    df = pd.read_csv(CSV_PATH, dtype=str)
    
    df = df.rename(columns={'age': 'ages'})
    
    csv_content = df.to_csv(index=False)
    file_like = BytesIO(csv_content.encode("utf-8"))
    upload = UploadFile(filename="modified.csv", file=file_like)
    
    with pytest.raises(HTTPException) as exc_info:
        df = load_data_from_upload(upload)
    
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["row"] == 0  
    assert "age" in str(exc_info.value.detail[0]["errors"]).lower()