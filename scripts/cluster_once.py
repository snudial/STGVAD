import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.generation_runner import run_pipeline_and_save

if __name__ == "__main__":
    run_pipeline_and_save("storage/nx_graphs_original.pkl")
