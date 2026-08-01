import sys
from pathlib import Path
import numpy as np

# Ajouter le répertoire racine au path Python
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Maintenant les imports fonctionneront
import pytest
import pandas as pd
from io import BytesIO
from fastapi import UploadFile, HTTPException
from model.main import load_data_from_upload, predict_with_model


class TestPredictWithModel:
    @pytest.fixture
    def mock_app_state(self, mocker):
        """Fixture pour mocker le modèle et les feature_columns."""
        # Mock du modèle
        mock_model = mocker.Mock()
        mock_model.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.8, 0.2],
            ])
        
        # Mock du request avec feature_columns
        mock_request = mocker.Mock()
        mock_request.app.state.model = mock_model
        mock_request.app.state.feature_columns = [
            'age',
            'niveau_hierarchique_poste',
            'heure_supplementaires',
            'satisfaction_employee_environnement',
            'note_evaluation_precedente',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso',
            'note_evaluation_actuelle',
            'augementation_salaire_precedente',
            'nombre_participation_pee',
            'nb_formations_suivies',
            'distance_domicile_travail',
            'niveau_education',
            'annees_depuis_la_derniere_promotion',
            'annes_sous_responsable_actuel',
            'nombre_experiences_precedentes',
            'annee_experience_totale',
            'annees_dans_l_entreprise',
            'annees_dans_le_poste_actuel'
        ]
        
        return mock_request

    def test_prediction_with_target_column(self, mock_app_state):
        data = pd.DataFrame({
            'age': [30, 35],
            'niveau_hierarchique_poste': [2, 3],
            'heure_supplementaires': [10, 15],
            'satisfaction_employee_environnement': [3.5, 4.0],
            'note_evaluation_precedente': [3.0, 3.5],
            'satisfaction_employee_nature_travail': [4.0, 3.8],
            'satisfaction_employee_equipe': [4.2, 3.9],
            'satisfaction_employee_equilibre_pro_perso': [3.7, 4.1],
            'note_evaluation_actuelle': [3.3, 3.6],
            'augementation_salaire_precedente': [5.0, 7.5],
            'nombre_participation_pee': [5, 3],
            'nb_formations_suivies': [2, 4],
            'distance_domicile_travail': [15, 25],
            'niveau_education': [16.0, 18.0],
            'annees_depuis_la_derniere_promotion': [2, 3],
            'annes_sous_responsable_actuel': [1, 2],
            'nombre_experiences_precedentes': [3, 5],
            'annee_experience_totale': [8, 12],
            'annees_dans_l_entreprise': [4, 7],
            'annees_dans_le_poste_actuel': [2, 3],
            'a_quitte_l_entreprise': [0, 1]
        })

        results = predict_with_model(mock_app_state, data)

        assert isinstance(results, pd.DataFrame)
        assert "row_id" in results.columns
        assert "prediction" in results.columns
        assert "probability" in results.columns
        assert len(results) == 2
        assert "a_quitte_l_entreprise" not in results.columns

    def test_prediction_without_target_column(self, mock_app_state):
        data = pd.DataFrame({
            'age': [30, 35],
            'niveau_hierarchique_poste': [2, 3],
            'heure_supplementaires': [10, 15],
            'satisfaction_employee_environnement': [3.5, 4.0],
            'note_evaluation_precedente': [3.0, 3.5],
            'satisfaction_employee_nature_travail': [4.0, 3.8],
            'satisfaction_employee_equipe': [4.2, 3.9],
            'satisfaction_employee_equilibre_pro_perso': [3.7, 4.1],
            'note_evaluation_actuelle': [3.3, 3.6],
            'augementation_salaire_precedente': [5.0, 7.5],
            'nombre_participation_pee': [5, 3],
            'nb_formations_suivies': [2, 4],
            'distance_domicile_travail': [15, 25],
            'niveau_education': [16.0, 18.0],
            'annees_depuis_la_derniere_promotion': [2, 3],
            'annes_sous_responsable_actuel': [1, 2],
            'nombre_experiences_precedentes': [3, 5],
            'annee_experience_totale': [8, 12],
            'annees_dans_l_entreprise': [4, 7],
            'annees_dans_le_poste_actuel': [2, 3]
        })

        results = predict_with_model(mock_app_state, data)

        assert isinstance(results, pd.DataFrame)
        assert "row_id" in results.columns
        assert "prediction" in results.columns
        assert "probability" in results.columns
        assert len(results) == 2
        assert results.loc[0, "prediction"] == 1  
        assert results.loc[1, "prediction"] == 0 

    def test_model_not_loaded_raises_500(self, mocker):
        mock_request = mocker.Mock()
        mock_request.app.state.model = None
        mock_request.app.state.feature_columns = None

        data = pd.DataFrame({"age": [30]})

        with pytest.raises(HTTPException) as exc_info:
            predict_with_model(mock_request, data)

        assert exc_info.value.status_code == 500
        assert "Modèle non chargé" in exc_info.value.detail

