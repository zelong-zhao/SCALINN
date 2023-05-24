from ..model_predict import model_evaluate
from .benchmark_tools import evaluate_model
import contextlib

def worst_and_best(args):
    with contextlib.redirect_stdout(None):
        Predicted_G_dict=model_evaluate(args,args.file)
    evaluate_model(args,Predicted_G_dict)