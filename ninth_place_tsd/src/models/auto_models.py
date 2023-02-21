from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from src.utils.mapper import configmapper

configmapper.map("models", "autotoken")(AutoModelForTokenClassification)
configmapper.map("models", "autotoken_3cls")(AutoModelForTokenClassification)
configmapper.map("models", "autotoken_4cls")(AutoModelForTokenClassification)
configmapper.map("models", "autospans")(AutoModelForQuestionAnswering)

