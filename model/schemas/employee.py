from pydantic import BaseModel, Field, ConfigDict

class EmployeeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    a_quitte_l_entreprise: int = Field(ge=0, le=1)   
    satisfaction_employee_environnement: float = Field(ge=0)       
    note_evaluation_precedente: float = Field(ge=0)                
    niveau_hierarchique_poste: int = Field(ge=1)                 
    satisfaction_employee_nature_travail: float = Field(ge=0)      
    satisfaction_employee_equipe: float = Field(ge=0)              
    satisfaction_employee_equilibre_pro_perso: float = Field(ge=0) 
    note_evaluation_actuelle: float = Field(ge=0)
    heure_supplementaires: int = Field(ge=0)
    augementation_salaire_precedente: float = Field(ge=0)
    nombre_participation_pee: int = Field(ge=0)                  
    nb_formations_suivies: int = Field(ge=0)   
    distance_domicile_travail: int = Field(ge=0) 
    niveau_education: float = Field(ge=0)                          
    annees_depuis_la_derniere_promotion: int = Field(ge=0)       
    annes_sous_responsable_actuel: int = Field(ge=0)   
    age: int = Field(ge=18, le=70)                                                                            
    nombre_experiences_precedentes: int = Field(ge=0)            
    annee_experience_totale: int = Field(ge=0)                   
    annees_dans_l_entreprise: int = Field(ge=0)                  
    annees_dans_le_poste_actuel: int = Field(ge=0)

      
